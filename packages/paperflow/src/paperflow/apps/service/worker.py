"""The background queue: one thread, strictly one document at a time.

Not parallel, on purpose. Neither RapidOCR nor the calls to the LiteLLM proxy are built
for concurrent use from one process, so running two documents at once would not be a
speedup - it would be a race. `queue.Queue` plus a single daemon thread is the whole of
the concurrency story: no broker, nothing outside this process.

Tag reviews go through the same queue for the same reason. Both jobs replace a document's
whole tag list, and a review interleaved with a cascade run on the same document would
have one of them write over the other.
"""

import logging
import queue
import threading
from collections.abc import Callable

from paperflow.pipeline import progress
from paperflow.pipeline.document_processor import DocumentProcessor
from paperflow.pipeline.tag_review import TagReviewer
from paperflow.settings import ServiceSettings

logger = logging.getLogger(__name__)


class QueueWorker:
    def __init__(
        self, processor: DocumentProcessor, tag_reviewer: TagReviewer, settings: ServiceSettings
    ) -> None:
        self._processor = processor
        self._tag_reviewer = tag_reviewer
        self._progress_path = settings.progress_path
        self._queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._thread: threading.Thread | None = None

    def enqueue(self, doc_id: int, *, force: bool) -> int:
        """Queue a document; return the queue length right after, for the log line."""
        self._queue.put(lambda: self._process(doc_id, force))
        return self._queue.qsize()

    def enqueue_tag_review(self, doc_id: int) -> int:
        self._queue.put(lambda: self._review_tags(doc_id))
        return self._queue.qsize()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="paperflow-cascade-worker"
        )
        self._thread.start()

    def _run(self) -> None:
        logger.info("worker started, waiting for documents")
        while True:
            job = self._queue.get()
            try:
                job()
            finally:
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
