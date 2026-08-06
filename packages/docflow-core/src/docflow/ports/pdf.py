"""Opening, rendering and rotating PDF pages, independent of which library does it.

Pulling this out of the pipeline is what lets `docflow.pipeline.native_text` and
`docflow.pipeline.cascade` be exercised with a handful of in-memory fakes rather
than a real file on disk and PyMuPDF installed. There is exactly one
implementation today - `docflow.adapters.pdf.fitz_document` - but the pipeline
never imports it directly.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


class PageHandle(Protocol):
    """One page of an open PDF document, numbered 1-based as it is shown to a user."""

    number: int

    @property
    def rotation(self) -> int:
        """The page's current /Rotate value: absolute degrees clockwise."""
        ...

    def set_rotation(self, degrees: int) -> None:
        """Set the page's absolute /Rotate.

        Affects every later render of this page - a correction made here is
        visible to both the OCR engine's own render and a subsequent escalation
        to the vision model, without either having to be told about it
        separately.
        """
        ...

    def text(self) -> str:
        """The page's embedded text layer, stripped of surrounding whitespace.

        "" when the page has no text layer at all - the common case for a scan.
        """
        ...

    def image_coverage(self) -> float:
        """Fraction of the page area covered by embedded images (0..1, occasionally more).

        A scanned page is, physically, one image covering the whole sheet
        (~1.0) - even when a text layer has been laid invisibly on top of it
        (a "searchable PDF"). A genuinely digital page (a Word/SAP/LaTeX
        export) has no images at all, or only small ones - a logo, a
        signature - so coverage is low. See
        `docflow.pipeline.native_text` for why this matters more than simply
        checking whether a text layer exists.
        """
        ...

    def render_jpeg_b64(self, *, dpi: int, quality: int) -> str:
        """Render the page to a base64-encoded JPEG, for a vision model call."""
        ...

    def render_bgr(self, *, dpi: int) -> "np.ndarray":
        """Render the page to a BGR array, for a local OCR engine."""
        ...


class PdfDocument(Protocol):
    """An open PDF document."""

    @property
    def page_count(self) -> int: ...

    def pages(self) -> Iterator[PageHandle]:
        """Iterate the document's pages, 1-based.

        Safe to call more than once on the same document: the pipeline looks
        at every page once to decide whether the document is born-digital,
        then again to actually process it.
        """
        ...

    def close(self) -> None: ...


class PdfOpener(Protocol):
    """Opens a PDF file from disk."""

    def open(self, path: str | Path) -> PdfDocument: ...
