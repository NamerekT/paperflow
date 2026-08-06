"""Watches the scan folder; enqueues a PDF once it has stopped changing.

Handles both `on_created` and `on_modified` - NAPS2 always writes to the same `Scan.pdf`,
so a second scan looks like a modification of a file that already exists, not a new
file. Both events settle through the same delay and the same debounce, since either can
be the one that actually announces a finished scan.
"""

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler

from paperflow.apps.watcher.queue import ScanQueue
from paperflow.settings import WatcherSettings

logger = logging.getLogger(__name__)


class ScanHandler(FileSystemEventHandler):
    """Enqueues new PDFs after a settle delay; never processes them itself."""

    def __init__(
        self, queue: ScanQueue, settings: WatcherSettings, on_enqueued: Callable[[], None]
    ) -> None:
        self._queue = queue
        self._settings = settings
        self._on_enqueued = on_enqueued
        #: Paths currently waiting out the settle delay - the debounce that keeps a
        #: second on_modified for the same file (NAPS2 sends both created and modified)
        #: from starting a second settle thread.
        self._in_progress: set[str] = set()
        self._in_progress_lock = threading.Lock()

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle(event)

    def _handle(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = str(event.src_path)
        if not path.lower().endswith(".pdf"):
            return

        with self._in_progress_lock:
            if path in self._in_progress:
                return
            self._in_progress.add(path)

        threading.Thread(
            target=self._settle_and_enqueue,
            args=(path,),
            daemon=True,
            name=f"settle-{Path(path).name}",
        ).start()

    def _settle_and_enqueue(self, path: str) -> None:
        try:
            logger.info(
                f"detected: {Path(path).name} (waiting {self._settings.write_settle_delay}s "
                "for it to finish writing)"
            )
            time.sleep(self._settings.write_settle_delay)

            if not Path(path).exists():
                logger.warning(f"file disappeared before it could be queued: {path}")
                return

            if self._queue.enqueue(Path(path)) is not None:
                self._on_enqueued()
        finally:
            with self._in_progress_lock:
                self._in_progress.discard(path)
