"""Error roots for the Paperless binding."""

from docflow.errors import DocflowError


class PaperflowError(Exception):
    """Base class for every error this package raises on purpose."""


class PaperlessError(PaperflowError):
    """The archive could not do what was asked."""


class PaperlessUnavailable(PaperlessError):
    """The archive could not be reached."""


class TaxonomyUnavailable(PaperlessError):
    """The list of existing tags, correspondents or types could not be read.

    Its own type because of what the caller must not do with it. This used to
    be swallowed and reported as an empty list, which is indistinguishable
    from "nothing has been filed yet" - and the classifier responds to an
    empty list by inventing a fresh near-duplicate of every entity it needs.
    A network blip could therefore silently pollute the archive.
    """


class TaskTimeout(PaperlessError):
    """A background task did not finish in time."""


class TaskFailed(PaperlessError):
    """A background task finished unsuccessfully."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail


__all__ = [
    "DocflowError",
    "PaperflowError",
    "PaperlessError",
    "PaperlessUnavailable",
    "TaskFailed",
    "TaskTimeout",
    "TaxonomyUnavailable",
]
