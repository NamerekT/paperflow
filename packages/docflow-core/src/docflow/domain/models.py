"""The record types that cross layer boundaries inside the engine.

All frozen: a `CascadeResult` is a finished statement about a document, and
nothing downstream has any business editing one. Every type here replaces a
bare dict that used to be indexed by string literal in two or three files at
once, where a typo surfaced as a KeyError in production, mid-document, after
the expensive OCR had already been paid for.
"""

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class ContentSource(StrEnum):
    """Where a document's text ultimately came from.

    These values are persisted: they are the select-option labels of the
    Paperless custom field "Content by", already stored on documents in the
    archive. Changing a value here silently splits reporting between the old
    label and the new one.
    """

    RAPIDOCR = "RapidOCR"
    VISION = "Vision-LLM (Gemma 31B)"
    MIXED = "Mixed"


@dataclass(frozen=True, slots=True)
class OcrLine:
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class PageStats:
    """Confidence summary for a single page.

    `p10` - the 10th percentile of line scores - is the useful half of this
    pair. The mean hides a handful of badly read lines among many good ones,
    which is exactly the failure the escalation threshold exists to catch.
    """

    mean: float
    p10: float
    line_count: int

    @property
    def is_empty(self) -> bool:
        return self.line_count == 0


@dataclass(frozen=True, slots=True)
class PageReport:
    """Per-page audit trail entry; replaces a positional tuple that used to be unpacked by index."""

    page: int  # 1-based, as printed to the user
    source: str  # "RapidOCR" | "Native PDF" | "Gemma 31B (LiteLLM: ...)"
    mean: float
    p10: float


@dataclass(frozen=True, slots=True)
class CascadeResult:
    content: str
    content_by: ContentSource
    #: Per-page attribution, set only when `content_by` is MIXED - some pages
    #: were read locally and some were escalated.
    mixed_by: str | None
    #: None when there was nothing to average: a born-digital document is not
    #: OCR'd at all, so it has no confidence to report.
    avg_confidence_pct: float | None
    avg_accuracy_pct: float | None
    page_reports: tuple[PageReport, ...]


@dataclass(frozen=True, slots=True)
class DocumentClassification:
    """What the document is, expressed in names rather than identifiers.

    This is the type that keeps the engine portable. Resolving a
    correspondent name to a database row is a property of the document
    manager, not of the document, so the engine stops here and the adapter
    takes over. A different archive means a different resolver, not a
    different classifier.
    """

    title: str
    correspondent: str | None
    document_type: str | None
    tags: tuple[str, ...]
    #: Which person the document concerns. Distinct from `correspondent`,
    #: which is the organisation that sent it.
    family_member: str | None
    date: date


@dataclass(frozen=True, slots=True)
class DocumentAnalysis:
    """Complete output of `docflow.pipeline.document_workflow`."""

    cascade: CascadeResult
    classification: DocumentClassification
    #: The deadline found in the text, if any. Most documents have none.
    frist: date | None
