"""
vision_ocr_test.py
────────────────────────────────────────────────────────────────────────────
Тест OCR через vision-LLM (модель, которая "смотрит" на картинку целиком и
понимает текст семантически) вместо посимвольного OCR (RapidOCR/Tesseract)
или layout-модели (Docling). Идея: и Docling, и RapidOCR за сегодня
облажались каждый по-своему — Docling путает украшенный аттестат с
картинкой и теряет текст, RapidOCR путает кириллицу с похожей латиницей на
словах-омоглифах (УКРАЇНА → VKPAIHA) и не читает рукописный текст вообще.
Vision-LLM (я сам, читая скриншот глазами) ни на одной из этих трёх вещей
не спотыкается — не путает картинку с текстом, понимает, что весь документ
на украинском (а не гадает по одной строке/движку), и справляется с почерком.

БЕСПЛАТНЫЕ МОДЕЛИ (проверено на openrouter.ai/models, отфильтровано по
image-input + text-output + price=0, на 1 августа 2026):

  nvidia/nemotron-nano-12b-v2-vl:free   — 12B, заявлен как заточенный именно
                                           под document intelligence и OCR
                                           (лидер OCRBench v2). Дефолт здесь.
  google/gemma-4-31b-it:free            — 31B, мультиязычный (140+ языков),
                                           тоже с акцентом на document
                                           understanding. Запасной вариант.

Бесплатные модели на OpenRouter лимитированы по запросам в день (обычно
десятки в сутки без пополнения баланса) — для теста на паре документов
более чем достаточно.

ЛОКАЛЬНО, НА ТВОЁМ ПК (Ryzen 5 5600 + 32 ГБ RAM + RTX 5060) — это реально:
видеокарты с 8 ГБ VRAM достаточно для vision-моделей в 3-8 млрд параметров
в квантованном виде. Через Ollama (ollama.com), варианты на выбор:

  ollama pull glm-ocr          — 0.9B, всего 2.2 ГБ, специально под OCR
                                  документов (таблицы, печати, сложная
                                  вёрстка), #1 на бенчмарке OmniDocBench.
                                  Очень быстрый. МИНУС: не проверял, насколько
                                  хорошо знает украинский — тренирован явно
                                  с акцентом на китайский/английский.
  ollama pull deepseek-ocr      — 3B, 6.7 ГБ, тоже заточен под OCR.
  ollama pull qwen3-vl:8b       — 8B, более безопасный выбор по языкам:
                                  Qwen исторически силён в многоязычном OCR,
                                  включая кириллицу.
  ollama pull minicpm-v4.5:8b   — 8B, тоже сильный на многоязычных сканах.

Ollama не держит модель в VRAM постоянно — выгружает через некоторое время
простоя (по умолчанию ~5 минут после последнего запроса), так что это не
"вечно висящий Docker", а ближе к твоей исходной идее с ленивой загрузкой.
Но: сама модель весит несколько ГБ на диске, и одноразовая закачка (один
раз через интернет) неизбежна что для облачного варианта (не нужна), что
для локального.

РЕКОМЕНДАЦИЯ: сначала прогони бесплатную облачную модель (ничего не грузить
не надо, результат через 10 секунд) — если качество устроит, локальная
модель вообще может не понадобиться.

ЗАПУСК:
    python vision_ocr_test.py 4
    python vision_ocr_test.py 4 --model gemma-4-31b-it
    python vision_ocr_test.py 4 --patch
    python vision_ocr_test.py 29 --patch
"""

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests


def log(msg: str) -> None:
    """print() с принудительным flush — иначе на Windows вывод может
    буферизоваться, и кажется, что скрипт завис, хотя он просто ждёт сети."""
    print(msg, flush=True)

# Все скачанные/сгенерированные файлы теста лежат в test/results/, а не
# рядом со скриптом — чтобы не мусорить папку тестов.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Секреты только из окружения — см. .env.example в корне проекта.
PAPERLESS_URL       = os.environ.get("PAPERLESS_URL", "https://paperless.boreray-balance.ts.net")
PAPERLESS_API_TOKEN = os.environ["PAPERLESS_API_TOKEN"]

LITELLM_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1")
LITELLM_KEY = os.environ.get("OPENROUTER_API_KEY", "")

RENDER_DPI = 200

PROMPT = (
    "Распознай ВЕСЬ текст на этом изображении документа — печатный и "
    "рукописный (документ может быть на украинском или немецком). Выведи "
    "результат как Markdown, сохраняя структуру таблиц (используй таблицы "
    "Markdown, если в документе есть таблица) и порядок блоков на странице. "
    "Ничего не придумывай и не пропускай — если что-то неразборчиво, "
    "отметь это как [нерозбірливо], но не оставляй пропуски молча. "
    "Не добавляй собственных комментариев или пояснений — только "
    "распознанный текст."
)


def _paperless_headers() -> dict:
    return {"Authorization": f"Token {PAPERLESS_API_TOKEN}"}


def get_or_download(doc_id: int) -> str:
    path = str(RESULTS_DIR / f"doc_{doc_id}_original.pdf")
    if os.path.exists(path):
        log(f"  OK  уже скачан ранее: {path}")
        return path
    resp = requests.get(
        f"{PAPERLESS_URL}/api/documents/{doc_id}/download/",
        headers=_paperless_headers(), params={"original": "true"}, timeout=60,
    )
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    log(f"  OK  скачан оригинал: {path}  ({len(resp.content)} байт)")
    return path


def render_pages_base64(pdf_path: str) -> list[str]:
    """
    Рендерит каждую страницу PDF в JPEG (не PNG!) и возвращает base64-строки.
    JPEG с качеством 85 весит в разы меньше PNG на цветном скане — меньше
    данных на аплоад/даунлоад и меньше vision-токенов на вход модели, что
    напрямую влияет на скорость ответа у медленных бесплатных бэкендов.
    """
    images = []
    doc = fitz.open(pdf_path)
    try:
        zoom = RENDER_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            jpeg_bytes = pix.tobytes("jpeg", jpg_quality=85)
            images.append(base64.b64encode(jpeg_bytes).decode("ascii"))
    finally:
        doc.close()
    return images


def ocr_page_with_vision_llm(image_b64: str, model: str, timeout: int, max_retries: int = 4) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0,
    }

    for attempt in range(1, max_retries + 1):
        started = time.monotonic()
        log(f"       -> попытка {attempt}/{max_retries}: отправляю запрос "
            f"({len(image_b64) // 1024} КБ base64), жду ответ (таймаут {timeout}с)...")
        try:
            resp = requests.post(
                f"{LITELLM_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {LITELLM_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
        except requests.exceptions.ReadTimeout:
            elapsed = time.monotonic() - started
            log(f"       <- нет ответа за {elapsed:.0f}с (таймаут). Бесплатный бэкенд "
                f"перегружен и просто не отвечает, а не медленно генерирует — "
                f"это подтверждается быстрым 429 у другой модели. Пробую снова...")
            continue

        elapsed = time.monotonic() - started
        log(f"       <- ответ за {elapsed:.1f}с, статус {resp.status_code}")

        if resp.status_code == 429:
            # Печатаем тело ответа — там обычно ясно написано, что именно
            # уперлось в лимит (дневная квота :free модели, RPM, и т.п.)
            log(f"       !! 429 Too Many Requests: {resp.text[:500]}")
            wait = float(resp.headers.get("Retry-After", 5 * attempt))
            if attempt < max_retries:
                log(f"       .. жду {wait:.0f}с и пробую снова")
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"{model}: упёрлись в rate limit после {max_retries} попыток. "
                f"Это дневной/минутный лимит бесплатных моделей на твоём ключе OpenRouter "
                f"(общий с обычными вызовами классификации из DocSorter — та же LITELLM_KEY). "
                f"Проверь usage/лимиты на openrouter.ai/activity. Известный обходной путь: "
                f"после разового пополнения баланса на $10 OpenRouter поднимает дневной лимит "
                f"бесплатных моделей с ~50 до ~1000 запросов в день, даже если продолжаешь "
                f"пользоваться исключительно :free моделями."
            )

        if resp.status_code >= 400:
            raise RuntimeError(f"{model}: HTTP {resp.status_code} — {resp.text[:500]}")

        data = resp.json()
        if "choices" not in data:
            raise RuntimeError(f"Неожиданный ответ от {model}: {data}")
        return data["choices"][0]["message"]["content"].strip()

    raise RuntimeError(f"{model}: не удалось получить ответ за {max_retries} попыток")


def patch_content(doc_id: int, text: str) -> None:
    resp = requests.patch(
        f"{PAPERLESS_URL}/api/documents/{doc_id}/",
        headers=_paperless_headers(), json={"content": text}, timeout=30,
    )
    resp.raise_for_status()
    log(f"  OK  content документа {doc_id} обновлён ({len(text)} символов)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc_id", type=int)
    ap.add_argument("--model", default="nvidia/nemotron-nano-12b-v2-vl:free")
    ap.add_argument("--timeout", type=int, default=90,
                     help="сек ожидания ответа от модели на ОДНУ страницу (по умолчанию 90)")
    ap.add_argument("--patch", action="store_true")
    args = ap.parse_args()

    pdf_path = get_or_download(args.doc_id)

    log("  ..  рендерим страницы PDF в картинки")
    images = render_pages_base64(pdf_path)
    log(f"       {len(images)} стр., модель: {args.model}")

    page_texts = []
    for i, img in enumerate(images):
        log(f"  ..  OCR страницы {i + 1}/{len(images)}")
        page_texts.append(ocr_page_with_vision_llm(img, args.model, args.timeout))

    full_text = "\n\n---\n\n".join(page_texts)

    log("\n" + "=" * 70)
    log(full_text)
    log("=" * 70 + "\n")

    out_path = str(RESULTS_DIR / f"doc_{args.doc_id}_vision.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    log(f"  OK  сохранено локально: {out_path}")

    if args.patch:
        patch_content(args.doc_id, full_text)
    else:
        log("  i   content НЕ записан — перезапусти с --patch, если результат устраивает")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        log(__doc__)
        sys.exit(0)
    main()
