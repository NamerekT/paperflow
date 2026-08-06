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
PAPERLESS_URL       = os.environ.get("PAPERLESS_URL", "https://paperless.boreray-balance.ts.net")
PAPERLESS_API_TOKEN = os.environ["PAPERLESS_API_TOKEN"]

OLLAMA_URL  = "http://localhost:11434"
RENDER_DPI  = 150   # lowered from 200: smaller image -> fewer vision tokens -> less VRAM
NUM_CTX     = 4096  # Qwen3-VL's "native" window is 256K — Ollama reserves VRAM for it
                     # by default even for a single page. 4096 tokens is plenty for us
                     # (image + prompt + a reasonable response); this sharply cuts
                     # memory requirements without any loss in recognition quality.

PROMPT = (
    "Распознай ВЕСЬ текст на этом изображении документа — печатный и "
    "рукописный (документ может быть на украинском или немецком). Выведи "
    "результат как plain text, сохраняя порядок блоков на странице. "
    "Ничего не придумывай и не пропускай — если что-то неразборчиво, "
    "отметь это как [нерозбірливо] или [unleserlich], но не оставляй пропуски молча. "
    "Не добавляй собственных комментариев или пояснений — только "
    "распознанный текст."
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
            f"Не могу достучаться до Ollama на {OLLAMA_URL} ({e}). "
            f"Установлена ли Ollama и запущена ли она (иконка в трее)? "
            f"См. https://ollama.com/download/windows"
        )


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


def ocr_page_with_ollama(image_b64: str, model: str, timeout: int, num_ctx: int) -> str:
    started = time.monotonic()
    log(f"       -> отправляю страницу модели {model} (может грузиться в VRAM "
        f"первый раз — не пугайся, если первая страница дольше остальных)...")
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
    log(f"       <- ответ за {elapsed:.1f}с, статус {resp.status_code}")

    if resp.status_code == 404:
        raise RuntimeError(
            f"Модель '{model}' не найдена. Скачай её сначала: ollama pull {model}"
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
    log(f"  OK  content документа {doc_id} обновлён ({len(text)} символов)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc_id", type=int)
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--timeout", type=int, default=180,
                     help="сек ожидания на страницу (первый запуск модели дольше — грузит веса в VRAM)")
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX,
                     help="размер контекста для Ollama (меньше -> меньше VRAM, по умолчанию 4096)")
    ap.add_argument("--patch", action="store_true")
    args = ap.parse_args()

    check_ollama_running()

    pdf_path = get_or_download(args.doc_id)

    log("  ..  рендерим страницы PDF в картинки")
    images = render_pages_base64(pdf_path)
    log(f"       {len(images)} стр., модель: {args.model}")

    page_texts = []
    for i, img in enumerate(images):
        log(f"  ..  OCR страницы {i + 1}/{len(images)}")
        page_texts.append(ocr_page_with_ollama(img, args.model, args.timeout, args.num_ctx))

    full_text = "\n\n---\n\n".join(page_texts)

    log("\n" + "=" * 70)
    log(full_text)
    log("=" * 70 + "\n")

    out_path = str(RESULTS_DIR / f"doc_{args.doc_id}_local_vision.md")
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
