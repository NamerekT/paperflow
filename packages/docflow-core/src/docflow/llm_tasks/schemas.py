"""Reply shapes, validated.

Pydantic is used here and in the webhook body, and nowhere else. Both are
places where untrusted input crosses into the system: a model's output is not
under our control no matter how firmly the prompt asks for a shape.

Each model below replaces a hand-written check - a `len(value) == 10 and
value[4] == "-"` for a date, an `in (0, 90, 180, 270)` for an angle - that had
to be kept in step with the prompt by hand.
"""

# Imported as a module rather than `from datetime import date`: two of the
# models below have a field called `date`, and a bare `date` annotation would
# resolve to that field instead of the type when pydantic evaluates it.
import datetime

from pydantic import BaseModel, ConfigDict, Field


class OrientationDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rotate: int = Field(default=0)

    def normalised(self) -> int:
        """The angle, or 0 if the model returned something off-axis.

        Anything outside the four quarter turns is meaningless to a PDF page
        rotation, and guessing at a nearest value would turn a page sideways
        with more confidence than the answer deserves.
        """
        return self.rotate if self.rotate in (0, 90, 180, 270) else 0


class FristDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    #: None is the common case, not a failure: most documents carry no
    #: deadline at all.
    frist: datetime.date | None = None


class ClassificationDraft(BaseModel):
    """Every field optional: a reply missing one is still used for the fields it does have."""

    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    correspondent: str | None = None
    document_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    family_member: str | None = None
    date: datetime.date | None = None
