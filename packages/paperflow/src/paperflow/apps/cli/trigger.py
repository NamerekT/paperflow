"""The nightly catch-up job: finds documents in Paperless and asks the webhook service
to process them. Runs no cascade itself - the only remaining implementation of that is
`paperflow.pipeline.document_processor`, reached here purely over HTTP.

Why this exists at all, given Paperless's own "Document Added" workflow already calls
the webhook on every new document: that trigger only fires for documents added after it
was configured, and not at all if the webhook was unreachable at the moment a document
was added (the service's container not up yet, mid-restart). This is the safety net -
once a day, it asks the service about every document, and the service itself decides
what is already done (the `Updated by OCR Cascade` flag) and what is not.

No local resumability: "already processed" is answered only by that live Paperless flag,
which `document_processor.already_processed` checks when the webhook request arrives.
This script sends ids and remembers nothing about what happened to them.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import requests

from paperflow.adapters.paperless.documents import PaperlessDocuments
from paperflow.adapters.paperless.tasks import PaperlessTasks
from paperflow.adapters.paperless.transport import PaperlessTransport
from paperflow.apps.cli import console
from paperflow.errors import TaskFailed, TaskTimeout
from paperflow.i18n import t
from paperflow.i18n.keys import M
from paperflow.settings import Settings, TriggerSettings

WEBHOOK_TIMEOUT = 10.0  # not retried - a skipped document is picked up by the next run

_UPLOADED_JSONL = Path(__file__).resolve().parent / "cascade_uploaded_from_folder.jsonl"


def trigger_webhook(trigger: TriggerSettings, doc_id: int, *, force: bool) -> dict[str, Any]:
    body: dict[str, Any] = {"document_id": doc_id}
    if force:
        body["force"] = True
    response = requests.post(
        trigger.webhook_url,
        headers={"X-Webhook-Token": trigger.webhook_secret, "Content-Type": "application/json"},
        json=body,
        timeout=WEBHOOK_TIMEOUT,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


# ── dedup by content hash, for --scan-folder (unchanged from the old script) ────────


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Content hash rather than path, since the same scan (a passport, say) often sits
    in more than one folder in the archive."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_uploaded_state() -> tuple[dict[str, int], dict[str, int]]:
    """(path -> doc_id, content_hash -> doc_id)."""
    by_path: dict[str, int] = {}
    by_hash: dict[str, int] = {}
    if not _UPLOADED_JSONL.exists():
        return by_path, by_hash

    for line in _UPLOADED_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        by_path[record["path"]] = record["doc_id"]
        file_hash = record.get("hash")
        if file_hash is None:
            candidate = Path(record["path"])
            if candidate.exists():
                try:
                    file_hash = file_sha256(candidate)
                except OSError:
                    file_hash = None
        if file_hash is not None:
            by_hash[file_hash] = record["doc_id"]
    return by_path, by_hash


def _append_uploaded(path: str, doc_id: int, file_hash: str) -> None:
    with open(_UPLOADED_JSONL, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"path": path, "doc_id": doc_id, "hash": file_hash}) + "\n")


def _upload_new_files(documents: PaperlessDocuments, folder: Path, limit: int | None) -> None:
    # The watcher's own folder: queue/ holds scans not yet uploaded, and "Uploaded to
    # Paperless/" holds ones already in the archive by that path - re-uploading either
    # from here would either catch a half-processed file or create a duplicate.
    watch_folder = os.environ.get("PAPERFLOW_WATCH_FOLDER")

    uploaded_by_path, uploaded_by_hash = _load_uploaded_state()
    all_pdfs = sorted(set(folder.rglob("*.pdf")) | set(folder.rglob("*.PDF")))
    if watch_folder:
        watch_path = Path(watch_folder)
        pdfs = [p for p in all_pdfs if not p.is_relative_to(watch_path)]
        skipped = len(all_pdfs) - len(pdfs)
        if skipped:
            console.echo(t(M.TRIGGER_SKIPPING_WATCH_FOLDER, count=skipped, folder=watch_folder))
    else:
        pdfs = all_pdfs

    new_uploads = 0
    for pdf_path in pdfs:
        key = str(pdf_path)
        if key in uploaded_by_path:
            continue

        try:
            file_hash = file_sha256(pdf_path)
        except OSError as exc:
            console.echo(t(M.TRIGGER_UPLOAD_FAILED, path=f"{pdf_path} ({exc})"))
            continue

        if file_hash in uploaded_by_hash:
            existing_id = uploaded_by_hash[file_hash]
            console.echo(t(M.TRIGGER_ALREADY_UPLOADED, path=str(pdf_path), doc_id=existing_id))
            uploaded_by_path[key] = existing_id
            _append_uploaded(key, existing_id, file_hash)
            continue

        if limit is not None and new_uploads >= limit:
            continue

        console.echo(t(M.TRIGGER_UPLOADING, path=str(pdf_path)))
        # No metadata: the universal "Document Added" workflow calls the webhook once
        # the document actually exists.
        try:
            new_id = documents.upload_and_wait(pdf_path)
        except (TaskTimeout, TaskFailed) as exc:
            console.echo(t(M.TRIGGER_UPLOAD_FAILED, path=f"{pdf_path} ({exc})"))
            continue
        if new_id is not None:
            _append_uploaded(key, new_id, file_hash)
            uploaded_by_path[key] = new_id
            uploaded_by_hash[file_hash] = new_id
            new_uploads += 1
        else:
            console.echo(t(M.TRIGGER_UPLOAD_FAILED, path=str(pdf_path)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="comma-separated ids - process only these")
    parser.add_argument(
        "--scan-folder", default=None, help="also upload new PDFs from this folder (no metadata)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="max NEW files from --scan-folder per run"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="force=true in the webhook body - reprocess even if already flagged done "
        "(only makes sense together with --only, almost always)",
    )
    parser.add_argument("--locale", default=None, choices=["ru", "en"], help="output language")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.load("cli")
    trigger_settings = TriggerSettings.load()
    console.configure_console()
    console.apply_locale(args.locale, settings)

    transport = PaperlessTransport(settings.paperless)
    transport.detect_api_version()
    tasks = PaperlessTasks(transport)
    documents = PaperlessDocuments(transport, tasks)

    if args.only:
        doc_ids = [int(x) for x in args.only.split(",")]
        console.echo(t(M.TRIGGER_ONLY_MODE, doc_ids=doc_ids))
    else:
        doc_ids = list(documents.iter_ids())
        console.echo(t(M.TRIGGER_DOC_COUNT, count=len(doc_ids)))

    if args.scan_folder:
        _upload_new_files(documents, Path(args.scan_folder), args.limit)

    console.echo(
        t(M.TRIGGER_SENDING_TO_WEBHOOK, count=len(doc_ids), url=trigger_settings.webhook_url)
    )
    ok, failed = 0, 0
    for doc_id in doc_ids:
        try:
            result = trigger_webhook(trigger_settings, doc_id, force=args.force)
            console.echo(t(M.TRIGGER_WEBHOOK_RESULT, doc_id=doc_id, result=result))
            ok += 1
        except Exception as exc:
            console.echo(t(M.TRIGGER_WEBHOOK_ERROR, doc_id=doc_id, error=str(exc)))
            failed += 1

    console.echo(t(M.TRIGGER_DONE, ok=ok, failed=failed))


if __name__ == "__main__":
    main()
