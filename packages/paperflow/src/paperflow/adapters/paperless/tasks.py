import logging
import time
from typing import Any

from paperflow.adapters.paperless.transport import PaperlessTransport
from paperflow.errors import PaperlessError, TaskFailed, TaskTimeout

logger = logging.getLogger(__name__)


class PaperlessTasks:
    def __init__(self, transport: PaperlessTransport, *, poll_interval: float = 3.0) -> None:
        self._transport = transport
        self._poll_interval = poll_interval

    def wait_for(self, task_id: str, *, timeout: float) -> int | None:
        """Block until the task finishes; return the document id it produced.

        Two shapes of the same endpoint have to be handled, and both cost real
        debugging time before they were understood:

        Statuses come back lower-case - "success", "failure", "pending",
        "started" - not in the upper-case Celery style. Comparing against
        "SUCCESS" simply never matched, so the loop ran to its timeout on
        documents that had been consumed successfully seconds earlier. The
        field listing the resulting ids is `related_document_ids`.

        And the endpoint answers either a bare list, on older versions, or a
        paginated object with a `results` key. Indexing `[0]` straight into a
        paginated response raised KeyError on every poll, with the same
        symptom: a loop that spins to its timeout while the work is already
        done.

        Raises rather than returning None on timeout or failure. The previous
        version returned None for both, along with "consumed but produced no
        document", so the caller could not tell a slow archive from a rejected
        file.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                payload = self._transport.get(
                    "/api/tasks/",
                    params={"task_id": task_id},
                    timeout=self._transport.settings.timeout_short,
                ).json()
            except PaperlessError as exc:
                # Keep polling - the archive may simply be restarting - but say
                # so at warning. This used to be swallowed entirely, so a
                # persistent failure looked identical to a slow document.
                logger.warning(f"polling task {task_id}: {exc}")
                time.sleep(self._poll_interval)
                continue

            results = self._extract_results(payload)
            if results:
                status = (results[0].get("status") or "").lower()
                if status == "success":
                    related = results[0].get("related_document_ids") or []
                    return int(related[0]) if related else None
                if status == "failure":
                    raise TaskFailed(
                        f"Paperless could not process the file (task {task_id})",
                        detail=str(results[0].get("result_data"))[:500],
                    )

            time.sleep(self._poll_interval)

        raise TaskTimeout(f"task {task_id} did not finish within {timeout:.0f}s")

    @staticmethod
    def _extract_results(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            return list(payload.get("results") or [])
        if isinstance(payload, list):
            return payload
        return []
