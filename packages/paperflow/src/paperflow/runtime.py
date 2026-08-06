"""Process-wide setup that has to happen before anything else runs.

Called from an application's `main()`, never on import. Every function here
mutates global interpreter state, which is exactly why it must be an explicit
step a program takes rather than a side effect of loading a module.
"""

import logging
import os
import sys
import threading
from types import TracebackType

log = logging.getLogger(__name__)


def install_excepthooks() -> None:
    """Route uncaught exceptions - in any thread - into the log.

    The watcher runs from Task Scheduler under pythonw.exe: no console, and
    stdout and stderr attached to nothing. An unhandled exception therefore
    used to mean a completely silent death, with nothing in the log file and
    only a return code in the scheduler's own history. That is not a
    hypothetical - on one occasion the scheduler recorded a launch while the
    application log had no entry for that time at all.

    `sys.excepthook` covers the main thread. `threading.excepthook` covers the
    background workers, and it matters more: a thread's exception also goes to
    stderr by default, so the thread dies quietly and leaves the process
    formally alive but doing nothing, which is far harder to notice than a
    crash. The webhook service needs this for the same reason - its queue
    worker survives today only because it wraps everything in a bare except.
    """

    def _log_main_thread(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        log.critical(
            "Unhandled exception in the main thread - the process is exiting",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    def _log_thread(args: threading.ExceptHookArgs) -> None:
        name = args.thread.name if args.thread else "<unknown>"
        message = f"Unhandled exception in thread {name!r} - the thread is exiting"
        # exc_value is typed as optional because a hook can in principle be
        # invoked without one; logging refuses that shape, so fall back to a
        # bare critical rather than losing the record entirely.
        if args.exc_value is None:
            log.critical(f"{message} (no exception object was supplied)")
        else:
            log.critical(message, exc_info=args.exc_value)

    sys.excepthook = _log_main_thread
    threading.excepthook = _log_thread


def limit_math_threads() -> None:
    """Keep the numeric libraries single-threaded.

    OCR runs one page at a time by design, so the BLAS backends spawning a
    thread per core buys nothing and costs contention with everything else in
    the process. Must be called before numpy or onnxruntime are imported -
    these variables are read once, at import.

    This used to run as a side effect of importing the OCR engine module,
    which made it silently ineffective for any importer that had already
    pulled in numpy.
    """
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
