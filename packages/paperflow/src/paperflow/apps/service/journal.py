"""The worker's queue, kept on disk so a restart of the container does not lose it.

A webhook is answered 202 as soon as its job is queued, and Paperless logs the call as a
success. A queue held only in memory would then drop every accepted job when the
container is recreated (a deploy) or killed (OOM, a crash in native code), with nothing
anywhere saying so. Each job is therefore one small JSON file in the `/data` volume,
written before the 202 and removed once the job has run; at startup the worker takes
whatever is left, oldest first.

One file per job rather than one append-only log: removing a finished job is an unlink,
not a rewrite, and a file broken by a crash mid-write affects only its own job.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

JobKind = Literal["cascade", "tag_review"]


@dataclass(frozen=True, slots=True)
class Job:
    kind: JobKind
    doc_id: int
    force: bool = False
    #: How many times the worker has taken this job up. A job still on disk with
    #: attempts > 0 was being processed when the process died.
    attempts: int = 0


class JobJournal:
    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._last_seq = 0

    def add(self, job: Job) -> Path:
        # The name is the order: a nanosecond clock, nudged forward so that two jobs
        # queued within the same tick still sort in arrival order.
        self._last_seq = max(time.time_ns(), self._last_seq + 1)
        path = self._dir / f"{self._last_seq:020d}.json"
        self._write(path, job)
        return path

    def read(self, path: Path) -> Job:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Job(
            kind=data["kind"],
            doc_id=int(data["doc_id"]),
            force=bool(data.get("force", False)),
            attempts=int(data.get("attempts", 0)),
        )

    def mark_started(self, path: Path, job: Job) -> Job:
        started = replace(job, attempts=job.attempts + 1)
        self._write(path, started)
        return started

    def remove(self, path: Path) -> None:
        path.unlink(missing_ok=True)

    def pending(self) -> list[tuple[Path, Job]]:
        """Jobs left from a previous run, oldest first. A file that cannot be read is
        renamed to `*.bad` and skipped, so one broken file does not block the rest."""
        jobs: list[tuple[Path, Job]] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                jobs.append((path, self.read(path)))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.error(f"unreadable queued job {path.name} set aside: {exc}")
                path.replace(path.with_name(path.name + ".bad"))
        return jobs

    def _write(self, path: Path, job: Job) -> None:
        # Write-then-rename, so a crash mid-write never leaves a half-written job file
        # under the name the worker reads.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(asdict(job)), encoding="utf-8")
        os.replace(tmp, path)
