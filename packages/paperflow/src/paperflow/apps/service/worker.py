"""The background queue: one thread, strictly one document at a time.

Not parallel, on purpose. Neither RapidOCR nor the calls to the LiteLLM proxy are built
for concurrent use from one process, so running two documents at once would not be a
speedup - it would be a race. `queue.Queue` plus a single daemon thread is the whole of
the concurrency story: no broker, nothing outside this process.

Tag reviews go through the same queue for the same reason. Both jobs replace a document's
whole tag list, and a review interleaved with a cascade run on the same document would
have one of them write over the other.

Every job is also written to `JobJournal` before it is queued, so the jobs left when the
container stops are picked up again at the next start - see that module for why.
"""

import logging
import queue
import threading
from pathlib import Path
from typing import Protocol

from paperflow.apps.service.journal import Job, JobJournal
from paperflow.pipeline import progress
from paperflow.pipeline.document_processor import DocumentProcessor
from paperflow.pipeline.tag_review import TagReviewer

logger = logging.getLogger(__name__)

#: A job the process has died on this many times is dropped at the next start instead
#: of being taken up again - otherwise a document that crashes native code would put
#: the container into a restart loop that never gets past it.
MAX_ATTEMPTS = 3


class WorkerSettings(Protocol):
    @property
    def progress_path(self) -> Path: ...

    @property
    def queue_dir(self) -> Path: ...


class QueueWorker:
    def __init__(
        self, processor: DocumentProcessor, tag_reviewer: TagReviewer, settings: WorkerSettings
    ) -> None:
        self._processor = processor
        self._tag_reviewer = tag_reviewer
        self._progress_path = settings.progress_path
        self._journal = JobJournal(settings.queue_dir)
        self._queue: queue.Queue[tuple[Path, Job]] = queue.Queue()
        self._thread: threading.Thread | None = None
        # Before any new job can be queued, so the leftovers keep their place ahead of it.
        self._recover()

    def enqueue(self, doc_id: int, *, force: bool) -> int:
        """Queue a document; return the queue length right after, for the log line."""
        return self._put(Job(kind="cascade", doc_id=doc_id, force=force))

    def enqueue_tag_review(self, doc_id: int) -> int:
        return self._put(Job(kind="tag_review", doc_id=doc_id))

    def pending(self) -> int:
        """Jobs not finished yet: queued plus the one being processed."""
        return self._queue.unfinished_tasks

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="paperflow-cascade-worker"
        )
        self._thread.start()

    def _put(self, job: Job) -> int:
        path = self._journal.add(job)
        self._queue.put((path, job))
        return self._queue.qsize()

    def _recover(self) -> None:
        recovered = 0
        for path, job in self._journal.pending():
            if job.attempts >= MAX_ATTEMPTS:
                logger.error(
                    f"[{job.doc_id}] {job.kind} dropped: the process died on it "
                    f"{job.attempts} times"
                )
                progress.append(
                    self._progress_path,
                    {
                        "doc_id": job.doc_id,
                        "error": f"{job.kind} dropped after {job.attempts} interrupted attempts",
                    },
                )
                self._journal.remove(path)
                continue
            self._queue.put((path, job))
            recovered += 1
        if recovered:
            logger.info(f"resuming {recovered} job(s) queued before the last stop")

    def _run(self) -> None:
        logger.info("worker started, waiting for documents")
        while True:
            path, job = self._queue.get()
            try:
                try:
                    job = self._journal.mark_started(path, job)
                except OSError:
                    logger.exception(f"[{job.doc_id}] could not update the queued job file")
                if job.kind == "cascade":
                    self._process(job.doc_id, job.force)
                else:
                    self._review_tags(job.doc_id)
            finally:
                try:
                    self._journal.remove(path)
                except OSError:
                    logger.exception(f"[{job.doc_id}] could not remove the queued job file")
                self._queue.task_done()

    def _process(self, doc_id: int, force: bool) -> None:
        try:
            self._processor.process(doc_id, force=force)
        except Exception as exc:
            logger.exception(f"[{doc_id}] processing failed")
            progress.append(self._progress_path, {"doc_id": doc_id, "error": str(exc)})

    def _review_tags(self, doc_id: int) -> None:
        try:
            self._tag_reviewer.review(doc_id)
        except Exception:
            logger.exception(f"[{doc_id}] tag review failed")
