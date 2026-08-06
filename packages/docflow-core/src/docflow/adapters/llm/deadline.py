"""POST with a hard wall-clock ceiling.

The case this exists for: a POST to the model gateway sometimes simply hangs.
Not an error, not a read timeout - nothing at all, past ten minutes with a 180
second timeout set. Meanwhile the provider's own dashboard showed a completed
request and spent tokens, so the generation happened; the HTTP response just
never arrived. The gateway appears to hold the connection alive at the proxy
level, so the client never sees a long enough silence for its own read timeout
to fire. A `timeout=` argument is not protection against this.

So the request is run on a worker thread and the *wait* is bounded from
outside, regardless of what the socket is doing. Python cannot kill the
thread; it stays parked until the provider eventually drops the connection.
That is acceptable - what matters is that a hung request stops meaning a hung
process.
"""

import concurrent.futures
from typing import Any

import requests


class DeadlinePoster:
    def __init__(self, *, max_workers: int = 4, session: requests.Session | None = None) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="llm-http"
        )
        self._session = session or requests.Session()

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        timeout: float,
        deadline_margin: float = 10.0,
    ) -> requests.Response:
        """POST with a ceiling of `timeout + deadline_margin` seconds.

        Raises `concurrent.futures.TimeoutError` when the ceiling is hit.
        Callers must handle that *alongside* `requests.ReadTimeout`, since
        either may fire first and both are normal.
        """
        future = self._pool.submit(
            self._session.post, url, headers=headers, json=json, timeout=timeout
        )
        return future.result(timeout=timeout + deadline_margin)

    def close(self) -> None:
        """Release the session but leave the pool alone.

        Deliberately not `shutdown(wait=True)`: that would block on exactly the
        hung workers this class exists to survive. The threads are daemons and
        go when the process does.
        """
        self._session.close()
