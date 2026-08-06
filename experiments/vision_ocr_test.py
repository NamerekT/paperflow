"""
vision_ocr_test.py
────────────────────────────────────────────────────────────────────────────
Tests OCR via a vision LLM (a model that "looks" at the whole image and
understands the text semantically) instead of character-by-character OCR
(RapidOCR/Tesseract) or a layout model (Docling). The idea: both Docling and
RapidOCR messed up today, each in its own way — Docling confuses a
decorated certificate for a picture and drops text, RapidOCR confuses
Cyrillic with similar-looking Latin on homoglyph words (УКРАЇНА →
VKPAIHA) and doesn't read handwritten text at all. A vision LLM (me,
reading a screenshot with my own eyes) doesn't trip up on any of these
three things — it doesn't mistake a picture for text, it understands the
whole document is in Ukrainian (rather than guessing per line/engine), and
it handles handwriting fine.

FREE MODELS (checked on openrouter.ai/models, filtered by image-input +
text-output + price=0, on August 1, 2026):

  nvidia/nemotron-nano-12b-v2-vl:free   — 12B, advertised as specifically
                                           tuned for document intelligence
                                           and OCR (leads OCRBench v2).
                                           Default here.
  google/gemma-4-31b-it:free            — 31B, multilingual (140+
                                           languages), also with an
                                           emphasis on document
                                           understanding. Fallback option.

Free models on OpenRouter are rate-limited per day (usually tens of
requests per day without topping up the balance) — more than enough for
testing on a couple of documents.

LOCALLY, ON YOUR PC (Ryzen 5 5600 + 32 GB RAM + RTX 5060) — this is
realistic: a GPU with 8 GB VRAM is enough for vision models in the 3-8B
parameter range in quantized form. Via Ollama (ollama.com), options to
choose from:

  ollama pull glm-ocr          — 0.9B, only 2.2 GB, purpose-built for
                                  document OCR (tables, stamps, complex
                                  layouts), #1 on the OmniDocBench
                                  benchmark. Very fast. DOWNSIDE: haven't
                                  checked how well it knows Ukrainian — it
                                  was clearly trained with an emphasis on
                                  Chinese/English.
  ollama pull deepseek-ocr      — 3B, 6.7 GB, also tuned for OCR.
  ollama pull qwen3-vl:8b       — 8B, a safer choice on languages: Qwen
                                  has historically been strong at
                                  multilingual OCR, including Cyrillic.
  ollama pull minicpm-v4.5:8b   — 8B, also strong on multilingual scans.

Ollama doesn't keep the model in VRAM permanently — it unloads it after
some idle time (~5 minutes after the last request by default), so this
isn't "a Docker container hanging around forever", it's closer to your
original idea of lazy loading. But: the model itself weighs several GB on
disk, and a one-time download (once, over the internet) is unavoidable
whether for the cloud option (not needed) or the local one.

RECOMMENDATION: run the free cloud model first (nothing to download,
result in 10 seconds) — if the quality is good enough, you may not need
the local model at all.

USAGE:
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
    """print() with a forced flush — otherwise output can get buffered on
    Windows, making it look like the script has hung when it's just waiting on the network."""
    print(msg, flush=True)

# All downloaded/generated test files live in experiments/results/, not
# next to the script — to avoid cluttering the experiments folder.
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Secrets come from the environment only — see .env.example in the project root.
PAPERLESS_URL       = os.environ.get("PAPERLESS_URL", "https://paperless.example.ts.net")
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
    Renders each PDF page to JPEG (not PNG!) and returns base64 strings.
    JPEG at quality 85 is many times smaller than PNG for a color scan —
    less data to upload/download and fewer vision tokens on the model's
    input, which directly affects response speed on slow free backends.
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
            # Print the response body — it usually spells out exactly what
            # hit the limit (the :free model's daily quota, RPM, etc.)
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
