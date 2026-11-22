"""The one `docflow.ports.pdf` implementation: PyMuPDF.

Absorbs the three private renderers that used to live beside the pipeline
itself - `_render_page_jpeg_b64` and `_page_to_bgr` from `ocr_cascade.py` and
`ocr_engine.py` respectively, and `_page_image_coverage` from `ocr_engine.py`.
Both call sites rendered the same page at the same DPI for two different
purposes (a JPEG for the vision model, a BGR array for local OCR); keeping
both renders behind one page handle means that duplication cannot happen
again.
"""

import base64
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pymupdf


class FitzPageHandle:
    """A `docflow.ports.pdf.PageHandle` backed by a `pymupdf.Page`."""

    def __init__(self, page: pymupdf.Page, number: int) -> None:
        self._page = page
        self.number = number

    def text(self) -> str:
        return str(self._page.get_text()).strip()

    def image_coverage(self) -> float:
        page_area = float(abs(self._page.rect))
        if not page_area:
            return 0.0
        total = 0.0
        for img in self._page.get_images(full=True):
            for rect in self._page.get_image_rects(img[0]):
                total += float(abs(rect))
        return total / page_area

    def _pixmap(self, dpi: int) -> pymupdf.Pixmap:
        zoom = dpi / 72
        return self._page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB)

    def render_jpeg_b64(self, *, dpi: int, quality: int) -> str:
        pix = self._pixmap(dpi)
        return base64.b64encode(pix.tobytes("jpeg", jpg_quality=quality)).decode("ascii")

    def render_bgr(self, *, dpi: int) -> np.ndarray:
        pix = self._pixmap(dpi)
        img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return img_rgb[:, :, ::-1]  # RapidOCR/OpenCV expect BGR, not RGB


class FitzDocument:
    """A `docflow.ports.pdf.PdfDocument` backed by a `pymupdf.Document`."""

    def __init__(self, document: pymupdf.Document) -> None:
        self._document = document

    @property
    def page_count(self) -> int:
        return int(self._document.page_count)

    def pages(self) -> Iterator[FitzPageHandle]:
        for number, page in enumerate(self._document, start=1):
            yield FitzPageHandle(page, number)

    def close(self) -> None:
        """Close the document and empty MuPDF's process-wide store.

        The store caches decoded page images and fonts and outlives the document; a
        reopened file gets new keys, so without this a long-running service grew by
        about the decoded size of every document until the store's 256 MB cap (PPF-9).
        Documents are processed one at a time, so nothing useful is lost.
        """
        try:
            self._document.close()
        finally:
            pymupdf.TOOLS.store_shrink(100)


class FitzOpener:
    """A `docflow.ports.pdf.PdfOpener` backed by PyMuPDF."""

    def open(self, path: str | Path) -> FitzDocument:
        return FitzDocument(pymupdf.open(str(path)))
