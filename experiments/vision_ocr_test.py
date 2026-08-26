"""
vision_ocr_test.py
───────────────────────────────────────────────────────────────────────────
Tests OCR via a vision LLM (a model that "looks" at the whole image and
understands the text semantically) instead of character-by-character OCR
(RapidOCR/Tesseract) or a layout model (Docling). The idea: both Docling and
RapidOCR messed up today, each in its own way — Docling confuses a
decorated certificate for a picture and drops text, RapidOCR confuses
Cyrillic with similar-looking Latin on homoglyph words (Ukraine →
vkpaIha) and doesn't read handwritten text at all. A vision LLM (me,
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
    "Recognize ALL text in this document image — printed and handwritten "
    "(document may be in Ukrainian or German). Output the result as Markdown, "
    "preserving table structure (use Markdown tables if the document contains "
    "tables) and block order on the page. Do not invent or skip anything — "
    "if something is unreadable, mark it as [unreadable], but do not leave "
    "silent skips. Do not add your own comments or explanations — only "
    "recognized text."
)


def _paperless_headers() -> dict:
    return {"Authorization": f"Token {PAPERLESS_API_TOKEN}"}


def get_or_download(doc_id: int) -> str:
    path = str(RESULTS_DIR / f"doc_{doc_id}_original.pdf")
    if os.path.exists(path):
        log(f"  OK  already downloaded: {path}")
        return path
    resp = requests.get(
        f"{PAPERLESS_URL}/api/documents/{doc_id}/download/",
        headers=_paperless_headers(), params={"original": "true"}, timeout=60,
    )
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    log(f"  OK  downloaded original: {path}  ({len(resp.content)} bytes")
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
        log(f"       -> attempt {attempt}/{max_retries}: sending request "
            f"({len(image_b64) // 1024} KB base64), waiting for response (timeout {timeout}s)...")
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
            log(f"       <- no response for {elapsed:.0f}s (timeout). Free backend "
                f"is overloaded and just not responding, not generating slowly — "
                f"this is confirmed by a quick 429 from another model. Trying again...")
            continue

        elapsed = time.monotonic() - started
        log(f"       <- response for {elapsed:.1f}s, status {resp.status_code}")

        if resp.status_code == 429:
            # Print the response body — it usually spells out exactly what
            # hit the limit (the :free model's daily quota, RPM, etc.)
            log(f"       !! 429 Too Many Requests: {resp.text[:500]}")
            wait = float(resp.headers.get("Retry-After", 5 * attempt))
            if attempt < max_retries:
                log(f"       .. waiting {wait:.0f}s and trying again")
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"{model}: hit rate limit after {max_retries} attempts. "
                f"This is the daily/minute limit for free models on your OpenRouter key "
                f"(shared with regular classification calls from DocSorter — same LITELLM_KEY). "
                f"Check usage/limits at openrouter.ai/activity. Known workaround: "
                f"after topping up balance by $10 once, OpenRouter raises daily free model limit "
                f"from ~50 to ~1000 requests per day, even if you keep using :free models only."
            )

        if resp.status_code >= 400:
            raise RuntimeError(f"{model}: HTTP {resp.status_code} — {resp.text[:500]}")

        data = resp.json()
        if "choices" not in data:
            raise RuntimeError(f"Unexpected response from {model}: {data}")
        return data["choices"][0]["message"]["content"].strip()

    raise RuntimeError(f"{model}: failed to get response after {max_retries} attempts")


def patch_content(doc_id: int, text: str) -> None:
    resp = requests.patch(
        f"{PAPERLESS_URL}/api/documents/{doc_id}/",
        headers=_paperless_headers(), json={"content": text}, timeout=30,
    )
    resp.raise_for_status()
    log(f"  OK  document {doc_id} content updated ({len(text)} characters)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc_id", type=int)
    ap.add_argument("--model", default="nvidia/nemotron-nano-12b-v2-vl:free")
    ap.add_argument("--timeout", type=int, default=90,
                     help="seconds waiting for model response on ONE page (default: 90)")
    ap.add_argument("--patch", action="store_true")
    args = ap.parse_args()

    pdf_path = get_or_download(args.doc_id)

    log("  ..  rendering PDF pages to images")
    images = render_pages_base64(pdf_path)
    log(f"       {len(images)} pages, model: {args.model}")

    page_texts = []
    for i, img in enumerate(images):
        log(f"  ..  OCR page {i + 1}/{len(images)}")
        page_texts.append(ocr_page_with_vision_llm(img, args.model, args.timeout))

    full_text = "\n\n---\n\n".join(page_texts)

    log("\n" + "=" * 70)
    log(full_text)
    log("=" * 70 + "\n")

    out_path = str(RESULTS_DIR / f"doc_{args.doc_id}_vision.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    log(f"  OK  saved locally: {out_path}")

    if args.patch:
        patch_content(args.doc_id, full_text)
    else:
        log("  i   content NOT written — re-run with --patch if the result is acceptable")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        log(__doc__)
        sys.exit(0)
    main()
