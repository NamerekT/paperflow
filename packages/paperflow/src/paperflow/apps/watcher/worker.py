"""Drains the queue strictly one file at a time - for the upload itself.

Wakes on `notify()` - a new scan was just queued - or every `idle_check_interval`
seconds regardless, which is what gives a file that failed once (Paperless briefly
unreachable, a transient error) a way back into processing without needing a second
filesystem event to trigger it.

Confirming with Paperless (waiting on its own consume task, then a toast) is handed off
to a separate bounded pool instead of running inline: it can take up to a minute per
file, and none of it needs the upload of the *next* file to wait around for it.

Draining does not begin until `paperless_ready` is set. Queueing is deliberately not
gated on that - scans reach the queue folder whether or not the archive is up - but
uploading into an archive that has never answered would only burn the retry path and
flood the log; the connection wait already logs its own throttled progress.
"""

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from paperflow.apps.watcher.queue import ScanQueue
from paperflow.settings import WatcherSettings

logger = logging.getLogger(__name__)


class QueueWorker(threading.Thread):
    def __init__(
        self,
        queue: ScanQueue,
        settings: WatcherSettings,
        *,
        upload: Callable[[Path], tuple[str, str] | None],
        finish: Callable[[str, str], None],
        paperless_ready: threading.Event | None = None,
    ) -> None:
        super().__init__(daemon=True, name="paperflow-watcher-worker")
        self._queue = queue
        self._settings = settings
        self._upload = upload
        self._finish = finish
        self._paperless_ready = paperless_ready
        self._wake = threading.Event()
        self._confirm_pool = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_confirmations,
            thread_name_prefix="paperflow-confirm",
        )

    def notify(self) -> None:
        self._wake.set()

    def run(self) -> None:
        logger.info("QueueWorker started")
        if self._paperless_ready is not None:
            self._paperless_ready.wait()
        self._wake.set()  # pick up anything left over from a previous run immediately
        while True:
            self._wake.wait(timeout=self._settings.idle_check_interval)
            self._wake.clear()
            self._drain()

    def _drain(self) -> None:
        files = self._queue.pending()
        if not files:
            return

        logger.info(f"queue: {len(files)} file(s) waiting")
        for path in files:
            if not path.exists():
                continue  # already processed while this pass was walking the list

            logger.info(f"starting: {path.name}")
            result = self._upload(path)
            if result is None:
                logger.warning(f"skipping {path.name}, will retry later")
                time.sleep(2)
                continue

            task_id, name = result
            self._confirm_pool.submit(self._finish, task_id, name)
