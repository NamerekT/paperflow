"""Finds probable duplicate documents in the archive. Shows them - changes nothing.

Paperless-ngx has no archive-wide "show me all the duplicates" view anywhere in its
interface, only a "Duplicates" tab on one document's own page (see GitHub discussion
#345, "How to find duplicates?"). This is that missing batch job.

THREE EARLIER ATTEMPTS, for context if this stops finding what it should - do not
re-learn these the hard way a second time:

  1. checksum from the API - DocumentSerializer never declares that field in
     Meta.fields, for any API version. Never arrives.
  2. SHA-256 of the downloaded files - would not have helped even if checksum did
     arrive: Document.checksum is declared unique=True (a Postgres UNIQUE constraint),
     so Paperless physically refuses to store two byte-identical files. Every real
     duplicate in the archive is a RESCAN of the same document - different bytes, same
     meaning - not an identical file.
  3. perceptual hash (phash) of the whole page - a real run against the archive
     produced two failures at once: a false negative (id 55/56 - literally the same VHS
     letter, differing in one word, "Valentyn"/"Valentyna" - not caught), and a false
     positive (25 entirely different Jobcenter/Familienkasse letters - different sums,
     dates, recipients - merged into one group, because German official letters from
     one office share almost the same layout/letterhead, and a whole-page phash sees
     that shared structure first, not what the letter actually says). The clustering was
     also transitive (Union-Find): if A resembles B and B resembles C, A and C land in
     one group even if A and C do not resemble each other at all - hence a group of 25.

WHAT THIS (fourth) ATTEMPT DOES:

  - The primary signal is the document's own OCR text (the "content" field - unlike
    checksum, it IS in DocumentSerializer.Meta.fields and arrives over the ordinary REST
    API; text documents need no download at all). Normalised (lower-cased, whitespace
    collapsed) and cut into shingles - sliding 8-word windows; similarity is the Jaccard
    coefficient (|intersection| / |union|) of two documents' shingle sets. This catches
    exactly "same text, name/date/amount changed" - what phash missed - and does NOT
    catch "same office template, different content", because the content (dates, sums,
    names, case numbers) will not match almost anywhere.
  - Page phash remains a FALLBACK, only for documents with almost no OCR text
    (< TEXT_MIN_CHARS characters after normalising): passport scans, photos, documents
    with no text layer.
  - Transitive Union-Find clustering is GONE. Instead of "groups", this prints PAIRS
    with their actual similarity score, so it is visible exactly why a match fired, and
    unrelated documents can no longer be merged through a chain of intermediates.
"""

import argparse
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from paperflow.adapters.paperless.documents import PaperlessDocuments
from paperflow.adapters.paperless.tasks import PaperlessTasks
from paperflow.adapters.paperless.transport import PaperlessTransport
from paperflow.apps.cli import console
from paperflow.domain.models import DocumentSummary
from paperflow.i18n import t
from paperflow.i18n.keys import M
from paperflow.settings import Settings

try:
    import fitz
    import imagehash
    from PIL import Image

    _HAS_IMAGE_DEPS = True
except ImportError:
    _HAS_IMAGE_DEPS = False

#: Documents with less text than this (after normalising) are treated as "not text" -
#: almost certainly a scan with no OCR layer, or a purely graphical document (a photo,
#: a passport) - and go through the phash fallback instead.
TEXT_MIN_CHARS = 100
SHINGLE_WORDS = 8
DEFAULT_TEXT_THRESHOLD = 0.55  # Jaccard coefficient, 0-1
DEFAULT_IMAGE_THRESHOLD = 8  # Hamming distance between phashes, 0-64 (fallback path)
HASH_SIZE = 8  # imagehash.phash(hash_size=8) -> a 64-bit hash
IMAGE_PAGES = 3  # how many leading pages to hash on the fallback path

_WORD_RE = re.compile(r"\w+", re.UNICODE)


# ── text path (primary) ──────────────────────────────────────────────────────────


def _normalize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _shingles(words: list[str], n: int = SHINGLE_WORDS) -> set[tuple[str, ...]]:
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set[Any], b: set[Any]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


# ── image fallback (for documents with nearly no text) ──────────────────────────


def _pdf_page_hashes(path: Path, max_pages: int) -> list[Any]:
    hashes = []
    try:
        with fitz.open(path) as doc:
            for i in range(min(max_pages, doc.page_count)):
                pix = doc[i].get_pixmap()
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                hashes.append(imagehash.phash(image, hash_size=HASH_SIZE))
    except Exception:
        console.echo(f"  !!  {path.name}: could not render the PDF")
    return hashes


def _image_hash(path: Path) -> list[Any]:
    try:
        with Image.open(path) as image:
            return [imagehash.phash(image.convert("RGB"), hash_size=HASH_SIZE)]
    except Exception:
        console.echo(f"  !!  {path.name}: could not open as an image")
        return []


def _hash_document(
    documents: PaperlessDocuments, summary: DocumentSummary, tmp_dir: Path, max_pages: int
) -> tuple[DocumentSummary, list[Any]]:
    mime = summary.mime_type or ""
    suffix = ".pdf" if "pdf" in mime else (Path(summary.original_file_name or "").suffix or ".bin")
    dest = tmp_dir / f"{summary.id}{suffix}"
    try:
        documents.download_original(summary.id, dest)
        if "pdf" in mime:
            hashes = _pdf_page_hashes(dest, max_pages)
        elif mime.startswith("image/"):
            hashes = _image_hash(dest)
        else:
            hashes = []
        return summary, hashes
    except Exception as exc:
        console.echo(f"  !!  id={summary.id} ({summary.original_file_name or '?'}): {exc}")
        return summary, []
    finally:
        dest.unlink(missing_ok=True)


def _doc_link(base_url: str, doc_id: int) -> str:
    return t(M.DUP_DOC_LINK, url=f"{base_url}/documents/{doc_id}/details")


def _doc_line(d: DocumentSummary) -> str:
    created = (d.created or "?")[:10]
    title = d.title or "(untitled)"
    return t(
        M.DUP_DOC_LINE,
        doc_id=f"{d.id:<6}",
        created=created,
        title=f"{title!r:<60}",
        filename=d.original_file_name or "?",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers", type=int, default=6, help="parallelism for the fallback (image) download path"
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=DEFAULT_TEXT_THRESHOLD,
        help=f"min Jaccard coefficient by text, 0-1 (default {DEFAULT_TEXT_THRESHOLD})",
    )
    parser.add_argument(
        "--image-threshold",
        type=int,
        default=DEFAULT_IMAGE_THRESHOLD,
        help=f"max phash Hamming distance, text-less docs only (default {DEFAULT_IMAGE_THRESHOLD})",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=IMAGE_PAGES,
        help=f"leading pages to hash on the fallback path (default {IMAGE_PAGES})",
    )
    parser.add_argument("--locale", default=None, choices=["ru", "en"], help="output language")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.load("cli")
    console.configure_console()
    console.apply_locale(args.locale, settings)

    transport = PaperlessTransport(settings.paperless)
    transport.detect_api_version()
    documents = PaperlessDocuments(transport, PaperlessTasks(transport))

    console.echo(t(M.DUP_FETCHING))
    fields = ["id", "title", "original_file_name", "created", "mime_type", "content"]
    summaries = list(documents.iter_summaries(fields))
    console.echo(t(M.DUP_TOTAL_DOCS, count=len(summaries)))

    text_docs: list[tuple[DocumentSummary, set[Any]]] = []
    image_docs: list[DocumentSummary] = []
    for summary in summaries:
        words = _normalize_words(summary.content)
        if len(" ".join(words)) >= TEXT_MIN_CHARS:
            text_docs.append((summary, _shingles(words)))
        else:
            image_docs.append(summary)

    console.echo(t(M.DUP_TEXT_COUNT, count=len(text_docs)))
    console.echo(t(M.DUP_IMAGE_COUNT, count=len(image_docs)))

    text_pairs = []
    for i in range(len(text_docs)):
        doc_i, shingles_i = text_docs[i]
        for j in range(i + 1, len(text_docs)):
            doc_j, shingles_j = text_docs[j]
            score = _jaccard(shingles_i, shingles_j)
            if score >= args.text_threshold:
                text_pairs.append((score, doc_i, doc_j))
    text_pairs.sort(key=lambda x: -x[0])

    image_pairs = []
    if image_docs:
        if not _HAS_IMAGE_DEPS:
            console.echo(t(M.DUP_MISSING_IMAGE_DEPS, count=len(image_docs)))
        else:
            console.echo(t(M.DUP_HASHING, count=len(image_docs), workers=args.workers))
            doc_hashes: dict[int, tuple[DocumentSummary, list[Any]]] = {}
            with tempfile.TemporaryDirectory(prefix="paperflow_dupcheck_") as tmp:
                tmp_dir = Path(tmp)
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = [
                        pool.submit(_hash_document, documents, summary, tmp_dir, args.pages)
                        for summary in image_docs
                    ]
                    for future in as_completed(futures):
                        summary, hashes = future.result()
                        doc_hashes[summary.id] = (summary, hashes)
            usable = [(d, h) for d, h in doc_hashes.values() if h]
            for i in range(len(usable)):
                doc_i, hashes_i = usable[i]
                for j in range(i + 1, len(usable)):
                    doc_j, hashes_j = usable[j]
                    best = min(x - y for x in hashes_i for y in hashes_j)
                    if best <= args.image_threshold:
                        image_pairs.append((best, doc_i, doc_j))
            image_pairs.sort(key=lambda x: x[0])

    if not text_pairs and not image_pairs:
        console.echo(
            t(
                M.DUP_NONE_FOUND,
                text_threshold=args.text_threshold,
                image_threshold=args.image_threshold,
            )
        )
        return

    if text_pairs:
        console.echo(t(M.DUP_TEXT_MATCHES_HEADER, count=len(text_pairs)))
        for score, doc_i, doc_j in text_pairs:
            console.echo(t(M.DUP_TEXT_SIMILARITY, score=f"{score:.0%}"))
            console.echo(_doc_line(doc_i))
            console.echo(_doc_link(transport.base_url, doc_i.id))
            console.echo(_doc_line(doc_j))
            console.echo(_doc_link(transport.base_url, doc_j.id))
            console.echo("")

    if image_pairs:
        console.echo(t(M.DUP_IMAGE_MATCHES_HEADER, count=len(image_pairs)))
        for distance, doc_i, doc_j in image_pairs:
            console.echo(t(M.DUP_IMAGE_DISTANCE, distance=distance, threshold=args.image_threshold))
            console.echo(_doc_line(doc_i))
            console.echo(_doc_link(transport.base_url, doc_i.id))
            console.echo(_doc_line(doc_j))
            console.echo(_doc_link(transport.base_url, doc_j.id))
            console.echo("")


if __name__ == "__main__":
    main()
