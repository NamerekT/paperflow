"""
gemini_vision_ocr_test.py
────────────────────────────────────────────────────────────────────────────
То же самое, что vision_ocr_test.py, но напрямую через Gemini API (Google AI
Studio), в обход OpenRouter — OpenRouter, судя по всему, роутит Gemma криво
и/или урезал лимиты. Прямой доступ к Gemma через Gemini API — отдельный
бесплатный пул лимитов, никак не связанный с тем, что показывал OpenRouter.

Источник API (проверено по офиц. доке 1 августа 2026):
https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api

Поддерживаемые модели Gemma на Gemini API:
    gemma-4-31b-it        — дефолт здесь, крупнее, обычно точнее
    gemma-4-26b-a4b-it     — MoE, легче, тоже вариант

Thinking-режим у Gemma 4 по умолчанию ВЫКЛЮЧЕН (включается только явным
"thinkingLevel": "high") — в отличие от Gemini-моделей, здесь ничего
специально отключать не нужно.

УСТАНОВКА:
    1. Получи бесплатный API-ключ: https://aistudio.google.com/apikey
       (обычный Google-аккаунт, не нужен OpenRouter/биллинг)
    2. Положи его в GEMINI_API_KEY в .env (корень проекта) или экспортируй
       в окружение — в коде ключей больше нет.
    3. Актуальные лимиты по своему ключу смотри на
       https://aistudio.google.com/rate-limit — это отдельный пул от
       того, что показывал OpenRouter.

ЗАПУСК:
    python gemini_vision_ocr_test.py 1
    python gemini_vision_ocr_test.py 1 --model gemma-4-26b-a4b-it
    python gemini_vision_ocr_test.py 1 --patch
"""

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Секреты только из окружения — см. .env.example в корне проекта.
PAPERLESS_URL       = os.environ.get("PAPERLESS_URL", "https://paperless.boreray-balance.ts.net")
PAPERLESS_API_TOKEN = os.environ["PAPERLESS_API_TOKEN"]

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/models"

RENDER_DPI = 200

PROMPT = (
    "Распознай ВЕСЬ текст на этом изображении документа — печатный и "
    "рукописный (документ может быть на украинском или немецком). Выведи "
    "результат простым текстом, сохраняя порядок блоков на странице и "
    "структуру таблиц (можно через табуляцию или ' | '). "
    "Ничего не придумывай и не пропускай — если что-то неразборчиво, "
    "отметь это как [нерозбірливо], но не оставляй пропуски молча и не "
    "подставляй правдоподобные, но выдуманные слова вместо реального текста. "
    "Не добавляй собственных комментариев или пояснений — только "
    "распознанный текст."
)


def log(msg: str) -> None:
    print(msg, flush=True)


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


def ocr_page_with_gemini(image_b64: str, model: str, timeout: int, max_retries: int = 4) -> str:
    url = f"{GEMINI_URL}/{model}:generateContent"
    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                {"text": PROMPT},
            ]
        }],
        "generationConfig": {"temperature": 0},
    }

    for attempt in range(1, max_retries + 1):
        started = time.monotonic()
        log(f"       -> попытка {attempt}/{max_retries}: отправляю запрос "
            f"({len(image_b64) // 1024} КБ base64), таймаут {timeout}с...")
        try:
            resp = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": GEMINI_API_KEY,
                },
                json=payload,
                timeout=timeout,
            )
        except requests.exceptions.ReadTimeout:
            elapsed = time.monotonic() - started
            log(f"       <- нет ответа за {elapsed:.0f}с (таймаут). Пробую снова...")
            continue

        elapsed = time.monotonic() - started
        log(f"       <- ответ за {elapsed:.1f}с, статус {resp.status_code}")

        if resp.status_code == 429:
            log(f"       !! 429: {resp.text[:500]}")
            wait = 10 * attempt
            if attempt < max_retries:
                log(f"       .. жду {wait}с и пробую снова")
                time.sleep(wait)
                continue
            raise RuntimeError(f"{model}: rate limit после {max_retries} попыток")

        if resp.status_code >= 400:
            raise RuntimeError(f"{model}: HTTP {resp.status_code} — {resp.text[:800]}")

        data = resp.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            # ВАЖНО: части с "thought": true — это внутренние рассуждения модели
            # вслух ("Wait, looking closer...", черновики, самокоррекции), а не
            # финальный ответ. Раньше это склеивалось вместе с текстом — отсюда
            # разросшийся в 4 раза файл с видимыми колебаниями модели. Берём
            # только не-thought части — то, что модель реально выдаёт как ответ.
            final_parts = [p.get("text", "") for p in parts if not p.get("thought")]
            text = "".join(final_parts).strip()
            if not text:
                # На случай, если модель ответила ТОЛЬКО thought-частями —
                # лучше вернуть их, чем пустую строку молча.
                text = "".join(p.get("text", "") for p in parts).strip()
            return text
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Неожиданный ответ от {model}: {data}") from e

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
    ap.add_argument("--model", default="gemma-4-31b-it")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--patch", action="store_true")
    args = ap.parse_args()

    if "ВСТАВЬ_СЮДА" in GEMINI_API_KEY:
        log("  !!  Сначала впиши свой ключ с https://aistudio.google.com/apikey в GEMINI_API_KEY")
        sys.exit(1)

    pdf_path = get_or_download(args.doc_id)

    log("  ..  рендерим страницы PDF в картинки")
    images = render_pages_base64(pdf_path)
    log(f"       {len(images)} стр., модель: {args.model}")

    page_texts = []
    for i, img in enumerate(images):
        log(f"  ..  OCR страницы {i + 1}/{len(images)}")
        page_texts.append(ocr_page_with_gemini(img, args.model, args.timeout))

    full_text = "\n\n---\n\n".join(page_texts)

    log("\n" + "=" * 70)
    log(full_text)
    log("=" * 70 + "\n")

    out_path = str(RESULTS_DIR / f"doc_{args.doc_id}_gemini.md")
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
