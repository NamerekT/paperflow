"""Waiting out Paperless not being up yet when a dependent process starts.

Paperless runs in Docker, which is not guaranteed to be ready before whatever depends on
it: the watcher starts on Windows logon, on its own schedule; the service's own stack
comes up under `docker compose up`, which starts containers without waiting for any of
them to actually answer. So Paperless being unreachable at startup is an expected
condition, not an outage - and this waits for it WITHOUT a deadline, never giving up and
never letting the process fall over on its own. An unhandled exception at startup used to
mean a silent death: a process launched via Task Scheduler under pythonw.exe leaves no
console and no traceback anywhere, only a return code in the scheduler's own history.

Logging every attempt at the retry interval would be spam if Paperless only comes up an
hour later, so a warning is written on the first failure and then at most once per
`log_every` seconds while the wait continues. The retries themselves stay frequent, so
the step still reacts quickly once Paperless answers - only the log is throttled, not the
check.

Not for one-off scripts, such as the nightly catch-up job: there, failing once with a
clear error and letting the next scheduled run try again is the more sensible choice than
blocking that run indefinitely.
"""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def startup_step_with_retry(
    step_name: str,
    fn: Callable[[], T],
    *,
    interval: float = 5.0,
    log_every: float = 1200.0,
) -> T:
    attempt = 0
    last_logged: float | None = None
    while True:
        attempt += 1
        try:
            result = fn()
            if attempt > 1:
                logger.info(f"{step_name}: available (after {attempt} attempts)")
            return result
        except Exception as exc:
            now = time.monotonic()
            if last_logged is None or now - last_logged >= log_every:
                logger.warning(
                    f"{step_name}: not available ({exc}) - waiting "
                    f"(Paperless is probably not up yet); attempt {attempt}"
                )
                last_logged = now
            time.sleep(interval)
