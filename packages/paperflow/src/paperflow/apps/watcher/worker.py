"""Drains the queue strictly one file at a time.

Wakes on `notify()` - a new scan was just queued - or every `idle_check_interval`
seconds regardless, which is what gives a file that failed once (Paperless briefly
unreachable, a transient error) a way back into processing without needing a second
filesystem event to trigger it.
"""

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from paperflow.apps.watcher.queue import ScanQueue
from paperflow.settings import WatcherSettings

logger = logging.getLogger(__name__)


class QueueWorker(threading.Thread):
    def __init__(
        self, queue: ScanQueue, settings: WatcherSettings, process: Callable[[Path], bool]
    ) -> None:
        super().__init__(daemon=True, name="paperflow-watcher-worker")
        self._queue = queue
        self._settings = settings
        self._process = process
        self._wake = threading.Event()

    def notify(self) -> None:
        self._wake.set()

    def run(self) -> None:
        logger.info("QueueWorker started")
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
            if not self._process(path):
                logger.warning(f"skipping {path.name}, will retry later")
                time.sleep(2)
