"""One place that configures logging, called once per process.

Library modules never call this. They do `logger = logging.getLogger(__name__)`
and nothing else, which is what lets an operator raise the level on one
subsystem - `paperflow.adapters.paperless.documents`, say - without drowning
in the rest.

Log records are always English. They are read by whoever is debugging, they
end up in `docker logs`, and they have to survive a Windows console; the
user-facing strings that do get translated live in `paperflow.i18n` and never
pass through here.
"""

import logging
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO", *, logfile: Path | None = None) -> None:
    """Set up root logging for this process.

    `logfile` is for the watcher, which runs under pythonw.exe with no console
    attached and would otherwise leave no trace at all.
    """
    # A Windows console defaults to a code page that cannot encode much beyond
    # Latin-1, and logging raises UnicodeEncodeError rather than dropping the
    # character - so one unlucky filename in a log message used to take down
    # the handler. Document titles and file names still arrive in any script,
    # even though the messages around them are English.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))

    logging.basicConfig(
        handlers=handlers,
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FORMAT,
        datefmt=_DATE_FORMAT,
        force=True,
    )


class DocumentLogAdapter(logging.LoggerAdapter[logging.Logger]):
    """Prefixes every record with the document being processed.

    The worker used to write `f"[{doc_id}] ..."` by hand at sixteen call
    sites, which held only as long as everybody remembered to do it.
    """

    def __init__(self, logger: logging.Logger, doc_id: int) -> None:
        super().__init__(logger, {"doc_id": doc_id})
        self.doc_id = doc_id

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        return f"[{self.doc_id}] {msg}", kwargs
