from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from paperflow.domain import constants

PaperlessFieldType = Literal["select", "longtext", "float", "boolean", "date", "string"]


@dataclass(frozen=True, slots=True)
class CustomFieldSpec:
    """A custom field this application requires Paperless to have.

    The schema is declared rather than created imperatively so that startup
    can both create what is missing and verify that what already exists has
    the type we are about to write into it. A field of the wrong type is a
    configuration error worth failing on immediately; discovering it at the
    first write means discovering it after the document has been OCR'd.
    """

    name: str
    data_type: PaperlessFieldType
    #: Labels for a select field, in the order they should appear.
    select_options: tuple[str, ...] | None = None
    #: True when the field is expected to be created by hand in the Paperless
    #: UI. Startup then verifies it but never creates it.
    externally_managed: bool = False


@dataclass(frozen=True, slots=True)
class CustomField:
    """A custom field as Paperless actually returned it."""

    id: int
    name: str
    data_type: str
    extra_data: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class CustomFieldValue:
    """One value to write, addressed by field id."""

    field: int
    value: Any


@dataclass(frozen=True, slots=True)
class CustomFieldSet:
    """The fields this application uses, resolved to ids once at startup.

    Callers reach fields through the named properties below rather than by
    subscripting with a string. The literals themselves still exist - they are
    persisted names and cannot change - but they exist in exactly one place,
    `paperflow.domain.constants`, instead of being spelled out at every use
    site across two modules. A misspelling is now a type error rather than a
    KeyError raised mid-document.
    """

    by_name: Mapping[str, CustomField]

    def __getitem__(self, name: str) -> CustomField:
        try:
            return self.by_name[name]
        except KeyError:
            known = ", ".join(sorted(self.by_name)) or "<none>"
            raise KeyError(
                f"custom field {name!r} was not resolved at startup; known fields: {known}"
            ) from None

    @property
    def content_by(self) -> CustomField:
        return self[constants.FIELD_CONTENT_BY]

    @property
    def mixed_created_by(self) -> CustomField:
        return self[constants.FIELD_MIXED_CREATED_BY]

    @property
    def avg_confidence(self) -> CustomField:
        return self[constants.FIELD_AVG_CONFIDENCE]

    @property
    def avg_accuracy(self) -> CustomField:
        return self[constants.FIELD_AVG_ACCURACY]

    @property
    def cascade_flag(self) -> CustomField:
        return self[constants.FIELD_CASCADE_FLAG]

    @property
    def frist(self) -> CustomField:
        return self[constants.FIELD_FRIST]


@dataclass(frozen=True, slots=True)
class ResolvedClassification:
    """A classification with every name turned into a Paperless id.

    The engine produces names (`docflow.domain.models.DocumentClassification`);
    this is what it becomes once the archive has been consulted and missing
    entities created. Ids may be None when Paperless refused to create the
    entity - the document is still worth saving with the fields that resolved.
    """

    title: str
    correspondent_id: int | None
    document_type_id: int | None
    tag_ids: tuple[int, ...]
    date: date


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    """The subset of a document's metadata the duplicate finder reads."""

    id: int
    title: str
    original_file_name: str | None
    created: str | None
    mime_type: str | None
    content: str
