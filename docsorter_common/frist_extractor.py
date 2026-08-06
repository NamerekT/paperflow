"""
frist_extractor.py — извлечение срока (Frist) из уже готового текста
документа: Widerspruchsfrist, срок оплаты, "ответить до...", "в течение N
недель с момента получения" и т.п.

Узкий, отдельный от classification.py вызов — намеренно: другая модель
(config.FRIST_LLM_MODEL, отдельный бюджет квоты от OCR-эскалации и от
основного классификатора, см. config.py), другой промт, единственная
задача. Вызывается ОТДЕЛЬНЫМ шагом уже ПОСЛЕ того, как документ обработан
основным пайплайном (см. вызывающий код в apply_ocr_cascade.py/
DocSorter.py/ocr_webhook_service) — читает content уже из Paperless через
API, не требует скачанного файла.
"""

import json
import time

import requests

from . import config
from .http_utils import post_with_deadline

PROMPT_SYSTEM = """Ты помогаешь семье мигрантов в Германии не пропускать сроки \
(Fristen) из официальных писем (Jobcenter, Familienkasse, Agentur für \
Arbeit, банк, арендодатель и т.п.).

Тебе дан текст письма и дата САМОГО ПИСЬМА (когда оно составлено/отправлено \
— НЕ путай эту дату со сроком, который нужно найти).

Найди в тексте явно указанный СРОК — дату (или период), до которой \
получатель должен что-то сделать: ответить, подать возражение \
(Widerspruch), оплатить, предоставить документы и т.п. Ищи формулировки \
вроде "Frist", "Widerspruchsfrist", "bis zum", "spätestens", "innerhalb \
von N Wochen/Tagen nach Erhalt/Zugang dieses Schreibens", "срок до", \
"протягом N днів".

ПРАВИЛА:
- Если срок указан как ОТНОСИТЕЛЬНЫЙ период (например "in 4 Wochen nach \
Zugang dieses Schreibens") — посчитай абсолютную дату ОТ ДАТЫ ПИСЬМА \
(не от сегодняшнего дня — ты не знаешь, когда именно это читают).
- НЕ путай срок с датой самого письма, с датой рождения, с "gültig bis" \
на удостоверении личности, с датой уже прошедшего события. Ищи именно \
дедлайн на БУДУЩЕЕ действие получателя.
- Если срок в тексте не указан ЯВНО — не выдумывай и не угадывай, верни null.
- Если сроков несколько — верни САМЫЙ БЛИЖАЙШИЙ (первый, который \
реально нужно успеть).

Ответь СТРОГО JSON, без markdown и пояснений:
{"frist": "YYYY-MM-DD"} или {"frist": null}"""


def _build_messages(content_text: str, document_date: str) -> list:
    return [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": f"ДАТА ПИСЬМА: {document_date}\n\nТЕКСТ ПИСЬМА:\n{content_text[:10000]}"},
    ]


def _call_llm(messages: list, log_fn, max_retries: int = 4, timeout: int = 90) -> dict | None:
    """Тот же паттерн ретраев (429/5xx/timeout), что
    classification.py::classify_with_llm — тот же класс проблем
    (self-hosted LiteLLM-прокси может зависнуть/зарейтлимитить), тот же
    self-hosted прокси (VISION_PROXY_URL), другая модель."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = post_with_deadline(
                f"{config.VISION_PROXY_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.VISION_PROXY_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.FRIST_LLM_MODEL,
                    "messages": messages,
                    "temperature": 0,
                },
                timeout=timeout,
            )
        except requests.exceptions.ReadTimeout:
            log_fn(f"  ..  Frist LLM не ответил за {timeout}с, попытка {attempt}/{max_retries}")
            continue
        except Exception as e:
            log_fn(f"  ..  Frist LLM: {e} — попытка {attempt}/{max_retries}")
            if attempt < max_retries:
                time.sleep(5 * attempt)
                continue
            return None

        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 10 * attempt))
            log_fn(f"  ..  429 Too Many Requests (Frist LLM), жду {wait:.0f}с (попытка {attempt}/{max_retries})")
            if attempt < max_retries:
                time.sleep(wait)
                continue
            return None

        if resp.status_code >= 500:
            log_fn(f"  ..  Frist LLM HTTP {resp.status_code}, попытка {attempt}/{max_retries}")
            if attempt < max_retries:
                time.sleep(5 * attempt)
                continue
            return None

        if resp.status_code >= 400:
            log_fn(f"  !!  Frist LLM: HTTP {resp.status_code} — {resp.text[:300]}")
            return None

        data = resp.json()
        content = data["choices"][0]["message"].get("content")
        if not content:
            log_fn(f"  !!  Frist LLM вернул пустой ответ")
            return None

        raw = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            log_fn(f"  !!  Frist LLM: невалидный JSON ({e}) | raw: {raw[:300]}")
            return None

    return None


def extract_frist(content_text: str, document_date: str, log_fn=print) -> str | None:
    """Возвращает 'YYYY-MM-DD', если в тексте явно указан срок (или его
    можно посчитать от document_date по указанному периоду), иначе None.
    Никогда не бросает исключение сама по себе на "не нашла срок" или
    "модель недоступна" — это ожидаемые исходы, не ошибки; вызывающий код
    просто пропускает шаг, если результат None."""
    if not content_text or not content_text.strip():
        return None

    result = _call_llm(_build_messages(content_text, document_date), log_fn)
    if result is None:
        return None

    frist = result.get("frist")
    if not frist:
        return None

    # Простая защита от переданной моделью ерунды вместо даты — вызывающий
    # код пишет это значение напрямую в Paperless custom field типа date,
    # он сам отклонит невалидный формат, но лучше не пытаться и не
    # затирать поле явным мусором.
    if not isinstance(frist, str) or len(frist) != 10 or frist[4] != "-" or frist[7] != "-":
        log_fn(f"  !!  Frist LLM вернула значение не в формате YYYY-MM-DD: {frist!r} — игнорирую")
        return None

    return frist
