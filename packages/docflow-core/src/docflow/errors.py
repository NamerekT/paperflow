"""Error roots for the engine.

Everything the engine raises deliberately derives from `DocflowError`, so an
application embedding it can distinguish "the engine gave up on this document"
from a genuine bug in its own code with a single `except`.
"""


class DocflowError(Exception):
    """Base class for every error the engine raises on purpose."""


class ConfigError(DocflowError):
    """Configuration is missing or contradictory.

    Raised at startup, never mid-document: the point is to fail before any
    expensive work has been done, not after.
    """
