"""Entry point: `python -m paperflow.apps.watcher` (or `pythonw.exe`, under Task
Scheduler, with no console attached at all).

Upload, local backup, toast - nothing else. Real classification, OCR escalation and the
deadline search all happen later, asynchronously, once Paperless's own "Document Added"
workflow calls the service; this process has no idea that webhook exists and never calls
it.
"""

import logging
import os
import time
from pathlib import Path

from watchdog.observers import Observer

from paperflow import logging_setup, runtime
from paperflow.adapters.paperless.documents import PaperlessDocuments
from paperflow.adapters.paperless.tasks import PaperlessTasks
from paperflow.adapters.paperless.transport import PaperlessTransport
from paperflow.adapters.startup import startup_step_with_retry
from paperflow.adapters.windows.toast import send_toast
from paperflow.apps.watcher.observer import ScanHandler
from paperflow.apps.watcher.queue import ScanQueue
from paperflow.apps.watcher.worker import QueueWorker
from paperflow.errors import PaperlessError, PaperlessUnavailable, TaskFailed, TaskTimeout
from paperflow.i18n import set_locale, t
from paperflow.i18n.keys import M
from paperflow.settings import Settings, WatcherSettings

logger = logging.getLogger(__name__)


def safe_filename(name: str) -> str:
    """Strip characters Windows forbids in file names."""
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name


def archive_local_copy(src: Path, filename: str, settings: WatcherSettings) -> Path:
    """Move the uploaded file into the local backup folder.

    Once Paperless has accepted the upload, the local copy is no longer canonical -
    Paperless owns the document now. Kept as a dumb backup rather than deleted, in case
    something goes wrong on the Paperless side.
    """
    filename = safe_filename(filename)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    settings.archive_folder.mkdir(parents=True, exist_ok=True)
    dest = settings.archive_folder / filename
    if dest.exists():
        dest = settings.archive_folder / f"{filename[:-4]}_{int(time.time())}.pdf"

    src.replace(dest)
    return dest


def upload_pdf(
    path: Path,
    documents: PaperlessDocuments,
    settings: WatcherSettings,
) -> tuple[str, str] | None:
    """Foreground half: upload and back up locally. Fast - a network POST, not the
    consume wait - so the dispatcher can move on to the next queued file right after.

    Returns (task_id, name) for the background confirm step to pick up, or None if the
    file should stay queued for a retry.

    Errors reaching Paperless at all get one long pause on top of leaving the file
    queued - the alternative is a log flooded with a connection attempt every couple of
    seconds for as long as the archive is down.
    """
    name = path.name
    try:
        task_id = documents.upload(path)
        logger.info(f"uploaded to Paperless (task {task_id})")

        # Always the queue name, not Paperless's assigned title: fetching it would mean
        # waiting on the same consume step this function exists to not block on, and at
        # that point in the pipeline the title is still whatever Paperless defaulted to
        # pre-classification anyway - classification runs later, out of process.
        archived = archive_local_copy(path, name, settings)
        logger.info(f"local backup: {archived}")
        return task_id, name

    except PaperlessUnavailable as exc:
        logger.error(f"no connection to Paperless: {exc}")
        logger.warning("file stays in the queue.")
        time.sleep(60)  # avoid spamming the log while Paperless is down
        return None
    except PaperlessError as exc:
        logger.error(f"Paperless error: {exc}")
        send_toast(t(M.TOAST_FAILED_TITLE), t(M.TOAST_FAILED_BODY, name=name, reason=str(exc)))
        return None
    except Exception as exc:
        logger.error(f"unexpected error processing {name}", exc_info=True)
        send_toast(t(M.TOAST_FAILED_TITLE), t(M.TOAST_FAILED_BODY, name=name, reason=str(exc)))
        return None


def finish_pdf(
    task_id: str,
    name: str,
    documents: PaperlessDocuments,
    tasks: PaperlessTasks,
    settings: WatcherSettings,
) -> None:
    """Background half: wait for Paperless's own consume task, then toast.

    Runs on the confirm pool, decoupled from the dispatcher - the document is already
    safely uploaded and backed up by the time this runs, so nothing here is worth
    blocking the next upload for, and failure here has nothing left to retry.
    """
    doc_id: int | None = None
    try:
        doc_id = tasks.wait_for(task_id, timeout=settings.paperless_wait_timeout)
    except (TaskTimeout, TaskFailed) as exc:
        # Only the confirmation is missing or says the consume step failed - the
        # document itself was already uploaded. Proceed without a direct link.
        logger.warning(f"Paperless did not confirm processing in time: {exc}")

    open_target = documents.url_for(doc_id) if doc_id else None
    send_toast(
        t(M.TOAST_UPLOADED_TITLE),
        t(M.TOAST_UPLOADED_BODY, name=name),
        open_target=open_target,
    )


def main() -> None:
    settings = Settings.load("watcher")
    logging_setup.configure_logging(settings.log_level, logfile=settings.require_watcher().log_file)
    set_locale(settings.locale)
    runtime.install_excepthooks()

    watcher_settings = settings.require_watcher()
    os.makedirs(watcher_settings.watch_folder, exist_ok=True)
    os.makedirs(watcher_settings.queue_folder, exist_ok=True)
    os.makedirs(watcher_settings.archive_folder, exist_ok=True)

    transport = PaperlessTransport(settings.paperless)
    startup_step_with_retry(
        "connecting to Paperless (detect_api_version)", transport.detect_api_version
    )
    tasks = PaperlessTasks(transport)
    documents = PaperlessDocuments(transport, tasks)

    queue = ScanQueue(watcher_settings)
    worker = QueueWorker(
        queue,
        watcher_settings,
        upload=lambda path: upload_pdf(path, documents, watcher_settings),
        finish=lambda task_id, name: finish_pdf(task_id, name, documents, tasks, watcher_settings),
    )
    worker.start()

    handler = ScanHandler(queue, watcher_settings, worker.notify)
    observer = Observer()
    observer.schedule(handler, str(watcher_settings.watch_folder), recursive=False)
    observer.start()

    logger.info("=" * 60)
    logger.info("paperflow watcher")
    logger.info(f"watch:     {watcher_settings.watch_folder}")
    logger.info(f"queue:     {watcher_settings.queue_folder}")
    logger.info(f"archive:   {watcher_settings.archive_folder}")
    logger.info(f"paperless: {transport.base_url}")
    logger.info("=" * 60)

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("stopped via Ctrl+C")
        observer.stop()

    observer.join()
    logger.info("watcher stopped.")


if __name__ == "__main__":
    main()
