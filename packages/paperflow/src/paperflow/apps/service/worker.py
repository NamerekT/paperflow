"""The background queue: one thread, strictly one document at a time.

Not parallel, on purpose. Neither RapidOCR nor the calls to the LiteLLM proxy are built
for concurrent use from one process, so running two documents at once would not be a
speedup - it would be a race. `queue.Queue` plus a single daemon thread is the whole of
the concurrency story: no broker, nothing outside this process.
"""

import logging
import queue
import threading

from paperflow.pipeline import progress
from paperflow.pipeline.document_processor import DocumentProcessor
from paperflow.settings import ServiceSettings

logger = logging.getLogger(__name__)


class QueueWorker:
    def __init__(self, processor: DocumentProcessor, settings: ServiceSettings) -> None:
        self._processor = processor
        self._progress_path = settings.progress_path
        self._queue: queue.Queue[tuple[int, bool]] = queue.Queue()
        self._thread: threading.Thread | None = None

    def enqueue(self, doc_id: int, *, force: bool) -> int:
        """Queue a document; return the queue length right after, for the log line."""
        self._queue.put((doc_id, force))
        return self._queue.qsize()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="paperflow-cascade-worker"
        )
        self._thread.start()

    def _run(self) -> None:
        logger.info("worker started, waiting for documents")
        while True:
            doc_id, force = self._queue.get()
            try:
                self._processor.process(doc_id, force=force)
            except Exception as exc:
                logger.exception(f"[{doc_id}] processing failed")
                progress.append(self._progress_path, {"doc_id": doc_id, "error": str(exc)})
            finally:
                self._queue.task_done()
