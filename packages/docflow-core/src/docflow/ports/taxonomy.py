"""What the archive already contains.

The classifier needs to know which tags, correspondents and document types
exist, so that it reuses "Jobcenter Wesel" instead of inventing
"Jobcenter-Wesel" beside it. That is the whole of the engine's interest in the
archive: a list of names in, a set of names out. Resolving those names to rows
in somebody's database is the adapter's problem.

Keeping it this narrow is what makes the engine portable. A binding for a
different document manager implements one method.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TaxonomySnapshot:
    """The names currently in use in the archive."""

    tags: tuple[str, ...] = ()
    correspondents: tuple[str, ...] = ()
    document_types: tuple[str, ...] = ()


class TaxonomySource(Protocol):
    def snapshot(self) -> TaxonomySnapshot:
        """Return the names that already exist.

        Implementations must raise rather than return an empty snapshot when
        the archive cannot be reached. An empty list is indistinguishable from
        "nothing has been filed yet", and the classifier reacts to it by
        creating a fresh near-duplicate of every entity it needs - which is
        both wrong and tedious to undo.
        """
        ...


@dataclass(frozen=True, slots=True)
class ClassificationFallback:
    """What to record when the model cannot say what a document is.

    Supplied by the application rather than defined here: these are real entity
    names in somebody's archive, and the engine has no business choosing them.
    In the Paperless binding they are German, because the archive is.
    """

    correspondent: str
    document_type: str
    tag: str
    family_member: str


@dataclass(frozen=True, slots=True)
class ClassifierPolicy:
    """The classifier's non-prompt rules."""

    #: Names the model must choose exactly one of.
    family_members: tuple[str, ...]

    fallback: ClassificationFallback

    #: Applied to every document regardless of what the model said. Added in
    #: code rather than asked for in the prompt, because "always" is not
    #: something a model can be relied upon to do.
    always_tags: tuple[str, ...] = field(default_factory=tuple)
