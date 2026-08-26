r"""
rapidocr_confidence_calibration.py
───────────────────────────────────────────────────────────────────────────
Calibrates RapidOCR confidence scores to determine when to escalate
processing to Gemma. Uses the same logic as in DocSorter.py and
rapidocr_compare_test.py for the latin + eslav engine pair.

The ESCALATION RULE (see ESCALATE_MEAN / ESCALATE_P10 below) is evaluated
NOT over the whole document at once, but PER PAGE SEPARATELY — "not the
average across the ward, but how many are in the morgue." An 8-page document
where 7 pages are perfect and 1 page is handwritten looks fine when averaged
over the whole document (the bad page gets drowned out), but gets caught
immediately with per-page scoring. If even one page of a document fails
mean < ESCALATE_MEAN OR p10 < ESCALATE_P10, the whole document is flagged
ESCALATE in the summary.

HOW THIS IS BUILT (important for running over many files):

  1. The file list lives in calibration_files.txt (one path/id per line, 
      # for comments). Append new lines and just rerun the script.

  2. Results accumulate in calibration_results.jsonl — one JSON line per
      document (path, category, per-page breakdown, all text lines and
      their score). Line-delimited format allows interruption and restart:
      the script finds already-processed documents (by path) in this file
      and skips them.

  3. A summary is written to calibration_summary.txt on every run: a list
      of documents flagged ESCALATE/ok, a per-category summary, and the
      worst/best lines. This is the only file you need to send/show after
      a run.

  4. SPEED: documents are processed in parallel (several processes at
      once — by default all cores minus two). This is the only significant
      speed lever: OCR is CPU-bound (models are already on disk, no downloads),
      and processing one document uses only one CPU core, leaving the rest idle.
      The Ryzen 5600 has 6 cores/12 threads, so a several-times-faster run
      is basically free.

USAGE:
    python rapidocr_confidence_calibration.py
    python rapidocr_confidence_calibration.py --scan-folder "D:\Documents\Unsorted scans"
    python rapidocr_confidence_calibration.py --workers 8

    (--workers defaults to number of cores minus 2, but never less than 1)
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Limit onnxruntime's internal multithreaded parallelism to 1 thread per
# worker process — otherwise each of the N parallel processes will try to
# grab all cores under the hood (onnxruntime/OpenMP), and instead of a
# speedup you get the opposite: processes fighting over the same cores.
# IMPORTANT: this must be set BEFORE importing onnxruntime/rapidocr, so it's
# done at module level, ahead of the relevant imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import fitz  # PyMuPDF
import numpy as np
import requests

try:
    from rapidocr import RapidOCR, LangRec, OCRVersion, ModelType
except ImportError:
    from rapidocr_onnxruntime import RapidOCR
    LangRec = OCRVersion = ModelType = None

SCRIPT_DIR    = Path(__file__).resolve().parent
RESULTS_DIR   = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FILES_LIST    = RESULTS_DIR / "calibration_files.txt"
RESULTS_JSONL = RESULTS_DIR / "calibration_results.jsonl"
SUMMARY_TXT   = RESULTS_DIR / "calibration_summary.txt"

# Secrets come from the environment only — see .env.example in the project root.
PAPERLESS_URL       = os.environ.get("PAPERLESS_URL", "https://paperless.example.ts.net")
PAPERLESS_API_TOKEN = os.environ["PAPERLESS_API_TOKEN"]
OCR_RENDER_DPI      = 300

# Escalation threshold into Gemma — evaluated PER PAGE (see the docstring above).
ESCALATE_MEAN = 0.95
ESCALATE_P10  = 0.95


def _headers() -> dict:
    return {"Authorization": f"Token {PAPERLESS_API_TOKEN}"}


def _log(msg: str) -> None:
    print(msg, flush=True)


# ── reading the file list ────────────────────────────────────────────────

def read_file_list(path: Path) -> list[str]:
    if not path.exists():
        _log(f"  !!  File not found: {path} — create it (see example in repository)")
        sys.exit(1)
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def download_paperless_doc(doc_id: int) -> str:
    path = str(RESULTS_DIR / f"doc_{doc_id}_original.pdf")
    if os.path.exists(path):
        return path
    _log(f"  ..  downloading doc_id={doc_id} from Paperless")
    resp = requests.get(
        f"{PAPERLESS_URL}/api/documents/{doc_id}/download/",
        headers=_headers(), params={"original": "true"}, timeout=60,
    )
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_folder_into_files_list(folder: str) -> int:
    """Recursively finds all *.pdf files in folder, deduplicates by sha256
    of their content, appends new paths to calibration_files.txt."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        _log(f"  !!  Folder not found: {folder}")
        sys.exit(1)

    existing_entries = set(read_file_list(FILES_LIST)) if FILES_LIST.exists() else set()
    existing_hashes = set()
    for entry in existing_entries:
        if entry.isdigit():
            continue
        p = Path(entry)
        if p.exists():
            try:
                existing_hashes.add(file_sha256(p))
            except OSError:
                pass

    # On Windows/NTFS matching is case-insensitive, so rglob("*.pdf") already
    # finds .PDF files too — both calls return the same list of paths.
    # Previously they were simply CONCATENATED (list + list), which meant
    # every file was counted twice (542 instead of 271 in the output — this
    # is what caused it) and hashed twice. Using set() collapses duplicates
    # at the path level — on Linux, where case matters, both name variants
    # will still be found correctly.
    all_pdfs = sorted(set(folder_path.rglob("*.pdf")) | set(folder_path.rglob("*.PDF")))
    seen_hashes = set(existing_hashes)
    new_paths = []
    for pdf_path in all_pdfs:
        try:
            h = file_sha256(pdf_path)
        except OSError as e:
            _log(f"  !!  unreadable {pdf_path}: {e}")
            continue
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        new_paths.append(str(pdf_path))

    if new_paths:
        with open(FILES_LIST, "a", encoding="utf-8") as f:
            f.write(f"\n# automatically added by folder scan: {folder}\n")
            for p in new_paths:
                f.write(p + "\n")

    _log(f"  ..  PDFs found total: {len(all_pdfs)}, unique new (by content): {len(new_paths)}")
    return len(new_paths)


def resolve_entry(entry: str) -> tuple[str, str]:
    """Returns (pdf_path, category). For numbers (Paperless ids) this already
    downloads the file — called in the main process BEFORE handing tasks out
    to workers, so the same document isn't downloaded in parallel from
    several processes at once."""
    if entry.isdigit():
        path = download_paperless_doc(int(entry))
        return path, "Paperless"
    path = entry
    category = Path(path).parent.name or "?"
    return path, category


# ── RapidOCR engines (production pair: latin + eslav) ───────────────────
# Global variables inside EACH worker process — initialized once via the
# pool's initializer at startup, then reused for every document that lands
# on that worker (we don't reload the models for every file).
_ENGINE_LATIN = None
_ENGINE_ESLAV = None


def build_ocr_engine(lang_rec, label: str):
    if LangRec is None:
        return RapidOCR()
    try:
        return RapidOCR(params={
            "Rec.lang_type": lang_rec,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
        })
    except Exception as e:
        _log(f"  !!  Failed to load model '{label}' ({e}) — falling back to default")
        return RapidOCR()


def _worker_init() -> None:
    global _ENGINE_LATIN, _ENGINE_ESLAV
    _ENGINE_LATIN = build_ocr_engine(LangRec.LATIN if LangRec else None, "latin")
    _ENGINE_ESLAV = build_ocr_engine(LangRec.ESLAV if LangRec else None, "eslav")


def ocr_lines(result):
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


def merge_best_per_line(latin_lines, eslav_lines):
    if latin_lines and eslav_lines and len(latin_lines) == len(eslav_lines):
        return [
            lt_ls if lt_ls[1] >= et_es[1] else et_es
            for lt_ls, et_es in zip(latin_lines, eslav_lines)
        ]
    return latin_lines + eslav_lines


def analyze_pdf_per_page(path: str, engine_latin, engine_eslav) -> list[list]:
    """Returns a list of pages; each page is a list of (text, score)."""
    pages_out = []
    doc = fitz.open(path)
    try:
        zoom = OCR_RENDER_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = img_rgb[:, :, ::-1]

            latin_lines = ocr_lines(engine_latin(img_bgr))
            eslav_lines = ocr_lines(engine_eslav(img_bgr))
            merged = merge_best_per_line(latin_lines, eslav_lines)
            pages_out.append(list(merged))
    finally:
        doc.close()
    return pages_out


def _process_one(task: tuple[str, str, str]) -> dict:
    """Runs inside a worker process. task = (entry, path, category)."""
    entry, path, category = task
    t0 = time.monotonic()
    try:
        pages = analyze_pdf_per_page(path, _ENGINE_LATIN, _ENGINE_ESLAV)
        return {
            "path": path, "category": category,
            "pages": pages, "error": None,
            "elapsed": round(time.monotonic() - t0, 1),
        }
    except Exception as e:
        return {
            "path": path, "category": category,
            "pages": [], "error": str(e),
            "elapsed": round(time.monotonic() - t0, 1),
        }


# ── accumulating results ────────────────────────────────────────────────

def load_already_done() -> set[str]:
    done = set()
    if RESULTS_JSONL.exists():
        for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["path"])
            except Exception:
                pass
    return done


def append_result(record: dict) -> None:
    with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_all_results() -> list[dict]:
    """Reads calibration_results.jsonl. Records with the old (pre-per-page)
    schema — {"scores": [...], "lines": [...]} without "pages" — are wrapped
    as a single "virtual" document page, so already-computed (expensive)
    results aren't lost. For such documents escalation is evaluated over the
    whole document at once (as before) — i.e. exactly as imprecise as it was
    before this change, not worse. To get an accurate per-page evaluation for
    them too, delete their line from calibration_results.jsonl; the path
    will remain in calibration_files.txt and the script will recompute it on
    the next run."""
    if not RESULTS_JSONL.exists():
        return []
    records = []
    for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "pages" not in r and "lines" in r:
            r["pages"] = [r["lines"]] if r["lines"] else []
        records.append(r)
    return records


# ── summary ──────────────────────────────────────────────────────────────

def page_stats(page_lines: list) -> tuple[float, float, float, int]:
    scores = np.array([s for _, s in page_lines], dtype=float)
    if len(scores) == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    return (float(scores.mean()), float(np.percentile(scores, 10)), float(scores.min()), len(scores))


def doc_needs_escalation(record: dict) -> list[int]:
    """Returns page numbers (1-based) that fail the threshold."""
    bad_pages = []
    for i, page_lines in enumerate(record.get("pages", []), 1):
        mean, p10, _min, n = page_stats(page_lines)
        if n == 0:
            continue
        if mean < ESCALATE_MEAN or p10 < ESCALATE_P10:
            bad_pages.append(i)
    return bad_pages


def write_summary(records: list[dict]) -> None:
    lines_out = []
    w = lines_out.append

    w("RapidOCR confidence calibration — summary")
    w(f"Total documents: {len(records)}")
    w(f"Escalation rule (per page): mean < {ESCALATE_MEAN} OR p10 < {ESCALATE_P10}")
    w("")

    escalate_list = []
    w(f"{'':9s} {'category':28s} {'file':45s} {'pg':>4s} {'mean':>6s} {'p10':>6s} {'min':>6s}")
    w("-" * 110)
    for r in records:
        all_lines = [ln for page in r.get("pages", []) for ln in page]
        arr = np.array([s for _, s in all_lines], dtype=float)
        name = Path(r["path"]).name
        if len(arr) == 0:
            w(f"{'ERR':9s} {r['category']:28s} {name:45s}  error: {r.get('error')}")
            continue
        bad_pages = doc_needs_escalation(r)
        tag = f"ESCALATE" if bad_pages else "ok"
        w(f"{tag:9s} {r['category']:28s} {name:45s} {len(r['pages']):4d} {arr.mean():6.3f} "
          f"{np.percentile(arr,10):6.3f} {arr.min():6.3f}")
        if bad_pages:
            escalate_list.append((r, bad_pages))

    w("")
    w(f"== DOCUMENTS TO ESCALATE TO GEMMA ({len(escalate_list)}) ==")
    w("(at least one page fails the threshold — only the failing pages are shown)")
    for r, bad_pages in escalate_list:
        name = Path(r["path"]).name
        w(f"\n  [{r['category']}] {name}")
        for pno in bad_pages:
            mean, p10, mn, n = page_stats(r["pages"][pno - 1])
            w(f"      pg.{pno}: n={n} mean={mean:.3f} p10={p10:.3f} min={mn:.3f}")

    w("")
    w("== SUMMARY BY CATEGORY (all lines, ignoring the per-page split) ==")
    by_cat = defaultdict(list)
    for r in records:
        for page in r.get("pages", []):
            by_cat[r["category"]].extend(s for _, s in page)
    for cat, scores in sorted(by_cat.items(), key=lambda kv: np.mean(kv[1]) if kv[1] else 0):
        if not scores:
            continue
        a = np.array(scores)
        w(f"{cat:32s} n={len(a):5d}  mean={a.mean():.3f}  p10={np.percentile(a,10):.3f}  min={a.min():.3f}")

    w("")
    w("== WORST 40 LINES ACROSS THE WHOLE SAMPLE ==")
    flat = []
    for r in records:
        for page in r.get("pages", []):
            for text, score in page:
                flat.append((score, r["category"], Path(r["path"]).name, text))
    for score, cat, name, text in sorted(flat, key=lambda x: x[0])[:40]:
        w(f"{score:.3f}  [{cat}] {name}: {text!r}")

    w("")
    w("== BEST 15 LINES (for contrast) ==")
    for score, cat, name, text in sorted(flat, key=lambda x: -x[0])[:15]:
        w(f"{score:.3f}  [{cat}] {name}: {text!r}")

    SUMMARY_TXT.write_text("\n".join(lines_out), encoding="utf-8")
    _log(f"\n  OK  summary written: {SUMMARY_TXT}  (documents to escalate: {len(escalate_list)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan-folder", default=None,
                     help="recursively find all PDFs in this folder and append them to calibration_files.txt")
    ap.add_argument("--workers", type=int, default=None,
                     help="how many processes to run in parallel (default: cores minus 2)")
    args = ap.parse_args()

    if args.scan_folder:
        scan_folder_into_files_list(args.scan_folder)

    entries = read_file_list(FILES_LIST)
    already_done = load_already_done()

    to_process = []
    for entry in entries:
        path, category = resolve_entry(entry)
        if path in already_done:
            continue
        to_process.append((entry, path, category))

    if not to_process:
        _log("  i   All files from the list are already processed — rebuilding the summary only.")
    else:
        n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
        _log(f"  ..  new files to process: {len(to_process)} (already done: {len(already_done)})")
        _log(f"  ..  parallel processes: {n_workers} (cores in system: {os.cpu_count()})")

        started = time.monotonic()
        done_count = 0
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
            futures = {pool.submit(_process_one, t): t for t in to_process}
            for fut in as_completed(futures):
                entry, path, category = futures[fut]
                done_count += 1
                try:
                    record = fut.result()
                except Exception as e:
                    record = {"path": path, "category": category, "pages": [], "error": str(e)}
                append_result(record)

                elapsed_total = time.monotonic() - started
                rate = done_count / elapsed_total if elapsed_total > 0 else 0
                remaining = (len(to_process) - done_count) / rate if rate > 0 else float("nan")
                status = "OK" if not record.get("error") else "ERR"
                _log(f"  {status}  [{done_count}/{len(to_process)}]  "
                      f"({record.get('elapsed', '?')}s)  ~{remaining/60:.1f} min remaining  "
                      f"{Path(path).name}")

    write_summary(load_all_results())


if __name__ == "__main__":
    main()
