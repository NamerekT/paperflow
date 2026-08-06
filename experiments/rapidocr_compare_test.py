"""
rapidocr_compare_test.py
────────────────────────────────────────────────────────────────────────────
A direct comparison: the same RapidOCR engine (latin + eslav, duplicated
from DocSorter.py — the code isn't imported directly because DocSorter.py
starts a watchdog observer immediately on import) on the same files where
Docling messed up (see doc_4_docling.md — "АОААТОК 4О АТЕСТАТА" [garbled OCR
output] and dropped chunks of text under "<!-- image -->").

Why this is worth checking: Docling has a separate page-layout model that
decides what on the page is "text" and what is a "picture" (decorative
element/photo) — and doesn't run OCR over regions marked as "picture". It
seems to be confusing the compact, decorated certificate with its border
and coat of arms for a photograph. RapidOCR doesn't do this — it simply
looks for all text blocks in the image and recognizes each one, with no
semantic "text vs picture" classification. So in theory it shouldn't hit
the same snag.

Works on the doc_4_original.pdf / doc_29_original.pdf files already sitting
in this folder (downloaded by docling_ocr_test.py). If they're missing, the
script downloads them itself.

USAGE:
    python rapidocr_compare_test.py 4
    python rapidocr_compare_test.py 4 --patch
    python rapidocr_compare_test.py 29 --patch
"""

import argparse
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import requests

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

try:
    from rapidocr import RapidOCR, LangRec, OCRVersion, ModelType
except ImportError:
    from rapidocr_onnxruntime import RapidOCR
    LangRec = OCRVersion = ModelType = None

# Secrets come from the environment only — see .env.example in the project root.
PAPERLESS_URL       = os.environ.get("PAPERLESS_URL", "https://paperless.boreray-balance.ts.net")
PAPERLESS_API_TOKEN = os.environ["PAPERLESS_API_TOKEN"]
OCR_RENDER_DPI      = 200


def _headers() -> dict:
    return {"Authorization": f"Token {PAPERLESS_API_TOKEN}"}


def get_or_download(doc_id: int) -> str:
    path = str(RESULTS_DIR / f"doc_{doc_id}_original.pdf")
    if os.path.exists(path):
        print(f"  OK  уже скачан ранее: {path}")
        return path
    resp = requests.get(
        f"{PAPERLESS_URL}/api/documents/{doc_id}/download/",
        headers=_headers(), params={"original": "true"}, timeout=60,
    )
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    print(f"  OK  скачан оригинал: {path}  ({len(resp.content)} байт)")
    return path


def _build_ocr_engine(lang_rec, label: str):
    if LangRec is None:
        return RapidOCR()
    try:
        return RapidOCR(params={
            "Rec.lang_type": lang_rec,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
        })
    except Exception as e:
        print(f"  !!  Не удалось загрузить модель '{label}' ({e}) — используем дефолтную")
        return RapidOCR()


def _ocr_lines(result):
    if result is None:
        return []
    if isinstance(result, tuple):
        result = result[0]
    if hasattr(result, "txts"):
        txts = list(result.txts or [])
        scores = list(getattr(result, "scores", None) or [])
        if len(scores) != len(txts):
            scores = [1.0] * len(txts)
        return list(zip(txts, scores))
    if isinstance(result, list):
        return [(item[1], item[2] if len(item) > 2 else 1.0) for item in result]
    return []


def _merge_best_per_line(latin_lines, eslav_lines):
    if latin_lines and len(latin_lines) == len(eslav_lines):
        return [lt if ls >= es else et for (lt, ls), (et, es) in zip(latin_lines, eslav_lines)]
    return [t for t, _ in latin_lines] + [t for t, _ in eslav_lines]


def extract_text_rapidocr(pdf_path: str, engine_latin, engine_eslav) -> str:
    texts = []
    doc = fitz.open(pdf_path)
    try:
        zoom = OCR_RENDER_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = img_rgb[:, :, ::-1]

            latin_lines = _ocr_lines(engine_latin(img_bgr))
            eslav_lines = _ocr_lines(engine_eslav(img_bgr))

            page_texts = _merge_best_per_line(latin_lines, eslav_lines)
            if page_texts:
                texts.append("\n".join(page_texts))
    finally:
        doc.close()
    return "\n\n".join(texts).strip()


def patch_content(doc_id: int, text: str) -> None:
    resp = requests.patch(
        f"{PAPERLESS_URL}/api/documents/{doc_id}/",
        headers=_headers(), json={"content": text}, timeout=30,
    )
    resp.raise_for_status()
    print(f"  OK  content документа {doc_id} обновлён ({len(text)} символов)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc_id", type=int)
    ap.add_argument("--patch", action="store_true")
    args = ap.parse_args()

    pdf_path = get_or_download(args.doc_id)

    print("  ..  загружаем модели RapidOCR (latin + eslav)")
    engine_latin = _build_ocr_engine(LangRec.LATIN if LangRec else None, "latin")
    engine_eslav = _build_ocr_engine(LangRec.ESLAV if LangRec else None, "eslav")

    print("  ..  распознаём")
    text = extract_text_rapidocr(pdf_path, engine_latin, engine_eslav)

    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")

    out_path = str(RESULTS_DIR / f"doc_{args.doc_id}_rapidocr.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  OK  сохранено локально: {out_path}")

    if args.patch:
        patch_content(args.doc_id, text)
    else:
        print("  i   content НЕ записан — перезапусти с --patch, если результат устраивает")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        sys.exit(0)
    main()
