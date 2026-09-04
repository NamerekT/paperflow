"""The on-disk queue: scans wait here between being detected and being uploaded.

A plain numbered folder, not a real queue library - `python DocSorter.py` has worked
this way since v1, and files surviving a restart in the queue folder is the recovery
mechanism, not a special case of one.
"""

import logging
import shutil
import threading
from pathlib import Path

from paperflow.settings import WatcherSettings

logger = logging.getLogger(__name__)


class ScanQueue:
    def __init__(self, settings: WatcherSettings) -> None:
        self._settings = settings
        #: Guards the read-max-then-write-next sequence in `enqueue` - two scans
        #: settling at the same moment must not both claim the same queue number.
        self._lock = threading.Lock()

    def enqueue(self, src_path: Path) -> Path | None:
        """Move `src_path` into the queue as `scan_NNNN.pdf`. None on failure."""
        with self._lock:
            self._settings.queue_folder.mkdir(parents=True, exist_ok=True)
            number = self._next_number()
            dest = self._settings.queue_folder / f"scan_{number:04d}.pdf"
            try:
                shutil.move(str(src_path), str(dest))
                logger.info(f"queued [{number:04d}]: {dest.name} <- {src_path.name}")
                return dest
            except FileNotFoundError:
                # The startup sweep and the observer can reach the same scan; whichever
                # loses the race finds nothing left to move, which is not an error.
                logger.debug(f"{src_path.name} was already queued by another pass")
                return None
            except OSError:
                logger.error(f"could not move {src_path.name} to the queue", exc_info=True)
                return None

    def _next_number(self) -> int:
        """The next sequential number. Call only while holding `_lock`."""
        highest = 0
        for existing in self._settings.queue_folder.glob("scan_*.pdf"):
            try:
                number = int(existing.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            highest = max(highest, number)
        return highest + 1

    def pending(self) -> list[Path]:
        folder = self._settings.queue_folder
        if not folder.exists():
            return []
        return sorted(folder.glob("scan_*.pdf"))
