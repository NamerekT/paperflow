"""The one `docflow.ports.pdf` implementation: PyMuPDF (fitz).

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

import fitz  # PyMuPDF
import numpy as np


class FitzPageHandle:
    """A `docflow.ports.pdf.PageHandle` backed by a `fitz.Page`."""

    def __init__(self, page: fitz.Page, number: int) -> None:
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

    def _pixmap(self, dpi: int) -> fitz.Pixmap:
        zoom = dpi / 72
        return self._page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)

    def render_jpeg_b64(self, *, dpi: int, quality: int) -> str:
        pix = self._pixmap(dpi)
        return base64.b64encode(pix.tobytes("jpeg", jpg_quality=quality)).decode("ascii")

    def render_bgr(self, *, dpi: int) -> np.ndarray:
        pix = self._pixmap(dpi)
        img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return img_rgb[:, :, ::-1]  # RapidOCR/OpenCV expect BGR, not RGB


class FitzDocument:
    """A `docflow.ports.pdf.PdfDocument` backed by a `fitz.Document`."""

    def __init__(self, document: fitz.Document) -> None:
        self._document = document

    @property
    def page_count(self) -> int:
        return int(self._document.page_count)

    def pages(self) -> Iterator[FitzPageHandle]:
        for number, page in enumerate(self._document, start=1):
            yield FitzPageHandle(page, number)

    def close(self) -> None:
        self._document.close()


class FitzOpener:
    """A `docflow.ports.pdf.PdfOpener` backed by PyMuPDF."""

    def open(self, path: str | Path) -> FitzDocument:
        return FitzDocument(fitz.open(str(path)))
