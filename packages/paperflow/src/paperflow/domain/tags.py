"""Which tags describe documents, and which only steer the tooling around them."""

from paperflow.domain import constants


def merge_target(name: str) -> str | None:
    """The tag `name` asks to be merged into, or None for an ordinary tag."""
    if not name.startswith(constants.MERGE_PREFIX):
        return None
    return name[len(constants.MERGE_PREFIX) :].strip() or None


def is_technical_tag(name: str) -> bool:
    """True for a tag the classifier must neither see nor choose."""
    return (
        name == constants.TAG_REVIEW_PARENT
        or name.startswith(constants.MERGE_PREFIX)
        or constants.TECHNICAL_TAG_MARKER.casefold() in name.casefold()
    )
