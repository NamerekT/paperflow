"""
local_vision_ocr_test.py
────────────────────────────────────────────────────────────────────────────
The same thing as vision_ocr_test.py, but without OpenRouter/the cloud —
runs the vision model LOCALLY via Ollama on your RTX 5060. No limits, no
429s, no hangs on someone else's free backend — the model either fits on
your GPU or it doesn't, and you see that immediately.

INSTALLATION (one-time):
    1. Download and install Ollama: https://ollama.com/download/windows
       (after installation an icon appears in the tray — the server listens
       on http://localhost:11434 on its own, no need to start it separately)
    2. Download the model (a few GB, one-time):
           ollama pull qwen3-vl:8b
       This is the main option. Checked against independent OCR-accuracy
       comparisons on Cyrillic: Qwen came out as the most accurate on
       Cyrillic text among open models in those comparisons, Gemma is
       noticeably weaker (~70% Cyrillic accuracy in the same comparison),
       and MiniCPM-V overall showed weak OCR results (I recommended it as
       the top choice earlier — that was a mistake, downgrading its
       priority). For German/Latin script Qwen is also fine — it's one of
       the major languages in its training data. The 8B model fits into
       the RTX 5060's 8 GB VRAM in quantized form.

    Alternatives if qwen3-vl doesn't work out on quality or speed:
           ollama pull gemma4:12b            — stronger than Qwen on
                                                German (Google's home
                                                language — one of the
                                                best-covered), weaker on
                                                Ukrainian (see above), but
                                                still a workable option
           ollama pull glm-ocr                — 0.9B, 2.2 GB, VERY fast,
                                                #1 on the OmniDocBench
                                                benchmark (tables/stamps/
                                                complex layouts), but
                                                trained with an emphasis on
                                                Chinese/English — Ukrainian
                                                support is questionable
           ollama pull maternion/lightonocr-2 — 1B, 1.5 GB, very fast, but
                                                built by its authors
                                                (LightOn, France) with a
                                                focus on French and
                                                scientific papers/LaTeX —
                                                Ukrainian support is also
                                                questionable
           ollama pull deepseek-ocr           — 3B, 6.7 GB, general-purpose
                                                OCR, Cyrillic coverage not
                                                confirmed
           ollama pull minicpm-v4.5:8b        — keep as a backup, but based
                                                on comparison data it's not
                                                the best choice for this
                                                task

Ollama unloads the model from VRAM on its own after ~5 minutes of
inactivity — it doesn't sit in memory permanently unless you're actively
running the script.

IF THE MODEL "DIES" ANYWAY (usually means it didn't fit in VRAM):
    1. Already done in this script: context is capped at 4096 tokens
       (--num-ctx) and rendering is reduced to 150 DPI — both parameters
       directly reduce the required VRAM without any loss in the model
       weights' quality.
    2. If it still doesn't fit — take a smaller model:
           ollama pull qwen3-vl:4b   (3.3 GB instead of 6.1 GB for 8b)
           ollama pull qwen3-vl:2b   (1.9 GB — almost certain to fit)
       Qwen3-VL is one architecture at different sizes; the OCR
       improvements (32 languages) are present across the whole lineup,
       not just in the 8B.
    3. Another lever — quantize not the weights (already Q4 by default),
       but the context's KV cache: the environment variable
       OLLAMA_KV_CACHE_TYPE=q8_0 (or q4_0 for even more savings) before
       starting Ollama. On Windows, set it via "Edit environment variables
       for your account", then restart Ollama (tray icon -> Quit -> start
       again).
    4. Check that VRAM isn't shared with something else (games, a browser
       with hardware acceleration, a second monitor with a heavy desktop)
       — `ollama ps` and Task Manager -> "Performance" tab -> GPU will show
       how much memory is actually free before you run this.

USAGE:
    python local_vision_ocr_test.py 4
    python local_vision_ocr_test.py 4 --model minicpm-v4.5:8b
    python local_vision_ocr_test.py 4 --patch
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

# Secrets come from the environment only — see .env.example in the project root.
PAPERLESS_URL       = os.environ.get("PAPERLESS_URL", "https://paperless.example.ts.net")
PAPERLESS_API_TOKEN = os.environ["PAPERLESS_API_TOKEN"]

OLLAMA_URL  = "http://localhost:11434"
RENDER_DPI  = 150   # lowered from 200: smaller image -> fewer vision tokens -> less VRAM
NUM_CTX     = 4096  # Qwen3-VL's "native" window is 256K — Ollama reserves VRAM for it
                     # by default even for a single page. 4096 tokens is plenty for us
                     # (image + prompt + a reasonable response); this sharply cuts
                     # memory requirements without any loss in recognition quality.

PROMPT = (
    "Recognize ALL text in this document image — printed and handwritten "
    "(document may be in Ukrainian or German). Output the result as plain text, "
    "preserving the order of blocks on the page. Do not invent or skip anything — "
    "if something is illegible, mark it as [unreadable] or [unleserlich], but do not leave "
    "silent skips. Do not add your own comments or explanations — only "
    "recognized text."
)


def log(msg: str) -> None:
    print(msg, flush=True)


def _paperless_headers() -> dict:
    return {"Authorization": f"Token {PAPERLESS_API_TOKEN}"}


def check_ollama_running() -> None:
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama on {OLLAMA_URL} ({e}). "
            f"Is Ollama installed and running (tray icon)? "
            f"See https://ollama.com/download/windows"
        )


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


def ocr_page_with_ollama(image_b64: str, model: str, timeout: int, num_ctx: int) -> str:
    started = time.monotonic()
    log(f"       -> sending page to model {model} (might load into VRAM "
        f"first time — don't worry if the first page is slower)...")
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": PROMPT, "images": [image_b64]}
            ],
            "stream": False,
            "options": {"num_ctx": num_ctx},
        },
        timeout=timeout,
    )
    elapsed = time.monotonic() - started
    log(f"       <- response for {elapsed:.1f}s, status {resp.status_code}")

    if resp.status_code == 404:
        raise RuntimeError(
            f"Model '{model}' not found. Download it first: ollama pull {model}"
        )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"].strip()


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
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--timeout", type=int, default=180,
                     help="seconds to wait per page (first run of the model takes longer — loads weights into VRAM)")
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX,
                     help="size of the context for Ollama (smaller -> less VRAM, default 4096)")
    ap.add_argument("--patch", action="store_true")
    args = ap.parse_args()

    check_ollama_running()

    pdf_path = get_or_download(args.doc_id)

    log("  ..  rendering PDF pages to images")
    images = render_pages_base64(pdf_path)
    log(f"       {len(images)} pages, model: {args.model}")

    page_texts = []
    for i, img in enumerate(images):
        log(f"  ..  OCR page {i + 1}/{len(images)}")
        page_texts.append(ocr_page_with_ollama(img, args.model, args.timeout, args.num_ctx))

    full_text = "\n\n---\n\n".join(page_texts)

    log("\n" + "=" * 70)
    log(full_text)
    log("=" * 70 + "\n")

    out_path = str(RESULTS_DIR / f"doc_{args.doc_id}_local_vision.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    log(f"  OK  saved locally: {out_path}")

    if args.patch:
        patch_content(args.doc_id, full_text)
    else:
        log("  i   content NOT written — re-run with --patch if the result is satisfactory")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        log(__doc__)
        sys.exit(0)
    main()
