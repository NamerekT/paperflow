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


def process_pdf(
    path: Path,
    documents: PaperlessDocuments,
    tasks: PaperlessTasks,
    settings: WatcherSettings,
) -> bool:
    """Upload, back up locally, toast. True on success (the file has left the queue);
    false leaves it there for the worker to retry.

    Errors reaching Paperless at all get one long pause on top of leaving the file
    queued - the alternative is a log flooded with a connection attempt every couple of
    seconds for as long as the archive is down.
    """
    name = path.name
    try:
        task_id = documents.upload(path)
        logger.info(f"uploaded to Paperless (task {task_id})")

        doc_id: int | None = None
        try:
            doc_id = tasks.wait_for(task_id, timeout=settings.paperless_wait_timeout)
        except (TaskTimeout, TaskFailed) as exc:
            # The upload itself already succeeded - only the confirmation is missing or
            # says the consume step failed. Either way there is nothing left to retry
            # here: proceed as before, just without a direct link in the toast.
            logger.warning(f"Paperless did not confirm processing in time: {exc}")

        open_target = documents.url_for(doc_id) if doc_id else None

        # The backup file name is whatever Paperless actually assigned (typically the
        # original file name, since no metadata was sent) - not a result of
        # classification, which does not happen here at all.
        archive_name = name
        if doc_id:
            try:
                archive_name = documents.get(doc_id).get("title") or name
            except PaperlessError:
                logger.warning(
                    f"could not fetch document {doc_id}'s title for the backup name", exc_info=True
                )

        archived = archive_local_copy(path, archive_name, settings)
        logger.info(f"local backup: {archived}")

        send_toast(
            t(M.TOAST_UPLOADED_TITLE),
            t(M.TOAST_UPLOADED_BODY, name=archive_name),
            open_target=open_target,
        )
        return True

    except PaperlessUnavailable as exc:
        logger.error(f"no connection to Paperless: {exc}")
        logger.warning("file stays in the queue.")
        time.sleep(60)  # avoid spamming the log while Paperless is down
        return False
    except PaperlessError as exc:
        logger.error(f"Paperless error: {exc}")
        send_toast(t(M.TOAST_FAILED_TITLE), t(M.TOAST_FAILED_BODY, name=name, reason=str(exc)))
        return False
    except Exception as exc:
        logger.error(f"unexpected error processing {name}", exc_info=True)
        send_toast(t(M.TOAST_FAILED_TITLE), t(M.TOAST_FAILED_BODY, name=name, reason=str(exc)))
        return False


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
        queue, watcher_settings, lambda path: process_pdf(path, documents, tasks, watcher_settings)
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
