"""
docling_ocr_test.py
────────────────────────────────────────────────────────────────────────────
One-off test script (NOT wired into DocSorter.py): downloads the original
document from Paperless, runs it through Docling (layout-aware OCR), and
either prints the resulting Markdown or (with --patch) writes it straight
into the document's content field via PATCH /api/documents/{id}/.

Why this lives outside DocSorter: we first need to see with our own eyes
whether Docling actually fixes what NAPS2/Tesseract currently gets wrong
(dropped text, Cyrillic/Latin mix-ups) before wiring it into the permanent
pipeline.

WHY YOU NEED TO RUN THIS SCRIPT YOURSELF, RATHER THAN ME RUNNING IT:
I tried to run the test in my cloud sandbox — it wasn't physically
possible: it only has ~9.8 GB of disk, and `pip install docling` pulls in
the CUDA build of torch by default (the standard PyPI wheel on Linux,
~2-3 GB of GPU libraries). The install failed with "No space left on
device". The official CPU-only PyTorch index (download.pytorch.org) is
also unreachable — the sandbox proxy blocks it. On your PC with normal
internet access this isn't a problem, as long as you install torch
correctly (see below).

INSTALLATION (one-time; Docling itself doesn't keep ANYTHING in memory
between runs afterwards — it loads and runs only while this script is
executing, exactly as you originally asked):

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install docling

The first line MATTERS: without it, pip will install the CUDA build of
torch (an extra ~2 GB of GPU libraries for a card this PC doesn't have).
The command above installs the CPU-only build — smaller, and it doesn't
pull in the CUDA runtime. Install in exactly this order (torch first), so
Docling sees torch is already present and doesn't try to replace it with
the GPU version.

IMPORTANT ABOUT LANGUAGES (verified against the Docling/EasyOCR source,
not guessed): EasyOCR (Docling's default OCR engine) does NOT allow mixing
German and Ukrainian in a single call — its code has a check that raises
ValueError when Latin and Cyrillic language groups are mixed. Documents 29
and 4 are purely Ukrainian, so this isn't an issue here (lang=["uk"]). But
if in the future you want to run a mixed document (German + Ukrainian on
the same page) through Docling, you'll need two passes with different
lang values and a merge of the results, the same way it's already done
for RapidOCR in DocSorter.py (_merge_best_per_line).

USAGE:
    python docling_ocr_test.py 29                # just show the result
    python docling_ocr_test.py 29 --patch         # show and write to content
    python docling_ocr_test.py 4 --patch --lang uk
    python docling_ocr_test.py <id> --lang de     # for a German document
"""

import argparse
import os
import sys
from pathlib import Path

import requests

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Secrets come from the environment only — see .env.example in the project root.
PAPERLESS_URL       = os.environ.get("PAPERLESS_URL", "https://paperless.example.ts.net")
PAPERLESS_API_TOKEN = os.environ["PAPERLESS_API_TOKEN"]


def _headers() -> dict:
    return {"Authorization": f"Token {PAPERLESS_API_TOKEN}"}


def download_original(doc_id: int) -> str:
    """Downloads the ORIGINAL (not the archived copy, which already has a broken text layer)."""
    resp = requests.get(
        f"{PAPERLESS_URL}/api/documents/{doc_id}/download/",
        headers=_headers(),
        params={"original": "true"},
        timeout=60,
    )
    resp.raise_for_status()
    path = str(RESULTS_DIR / f"doc_{doc_id}_original.pdf")
    with open(path, "wb") as f:
        f.write(resp.content)
    print(f"  OK  скачан оригинал: {path}  ({len(resp.content)} байт)")
    return path


def run_docling(pdf_path: str, lang: str) -> str:
    """
    force_full_page_ocr=True — mandatory for our scans: the pages already
    have a broken text layer from NAPS2, and without this flag Docling
    (like Paperless by default) may decide text already exists and skip
    re-recognizing the page.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        ocr_options=EasyOcrOptions(lang=[lang], force_full_page_ocr=True),
    )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


def patch_content(doc_id: int, markdown: str) -> None:
    """
    Writes the Markdown straight into the document's content field. Verified
    via the browser (OPTIONS /api/documents/{id}/): the field is NOT
    read-only, PATCH is allowed (Allow: GET, PUT, PATCH, DELETE, HEAD,
    OPTIONS). The PDF file itself is untouched — only what Paperless uses
    for search changes.
    """
    resp = requests.patch(
        f"{PAPERLESS_URL}/api/documents/{doc_id}/",
        headers=_headers(),
        json={"content": markdown},
        timeout=30,
    )
    resp.raise_for_status()
    print(f"  OK  content документа {doc_id} обновлён ({len(markdown)} символов)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc_id", type=int)
    ap.add_argument("--lang", default="uk", help="код языка для EasyOCR (по умолчанию uk)")
    ap.add_argument("--patch", action="store_true", help="записать результат в content документа")
    args = ap.parse_args()

    pdf_path = download_original(args.doc_id)
    print(f"  ..  гоняем Docling (lang={args.lang}); первый запуск дольше — грузит модели")
    markdown = run_docling(pdf_path, args.lang)

    print("\n" + "=" * 70)
    print(markdown)
    print("=" * 70 + "\n")

    out_path = str(RESULTS_DIR / f"doc_{args.doc_id}_docling.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"  OK  сохранено локально: {out_path}")

    if args.patch:
        patch_content(args.doc_id, markdown)
    else:
        print("  i   content НЕ записан — перезапусти с --patch, если результат устраивает")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        sys.exit(0)
    main()
