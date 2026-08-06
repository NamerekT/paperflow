"""`FitzDocument`/`FitzPageHandle` against real files - the one place PyMuPDF itself is
exercised, rather than a fake `PageHandle`.

`EK Valentyna Tymchenko (Printed to PDF).pdf` is the actual document behind the
motivating story in `docflow.pipeline.native_text`: a scan that was printed to PDF and
gained a text layer that is not genuine digital text but somebody else's OCR output. Its
image coverage is why the check exists.
"""

from pathlib import Path

import pytest

from docflow.adapters.pdf.fitz_document import FitzOpener
from docflow.domain.policies import NativeTextConfig
from docflow.pipeline.native_text import is_native_text_document

DATA = Path(__file__).resolve().parent.parent / "experiments" / "data"
CONFIG = NativeTextConfig()

# The fixtures here are real, personally identifying documents from the archive this
# project processes - gitignored (see experiments/data/ in .gitignore), not present in a
# fresh checkout. These tests skip rather than fail when they are absent; where they are
# present, they are the only place PyMuPDF itself, rather than a fake PageHandle, is
# exercised.
pytestmark = pytest.mark.skipif(
    not DATA.is_dir(), reason="experiments/data/ fixtures are gitignored and local-only"
)


def test_a_printed_to_pdf_scan_is_not_trusted_as_native_text():
    doc = FitzOpener().open(DATA / "EK Valentyna Tymchenko (Printed to PDF).pdf")
    try:
        assert doc.page_count == 4
        # Page 1 carries a text layer (someone else's OCR), but the page is still, in
        # substance, a full-page scanned image - the image-coverage check rejects it.
        pages = list(doc.pages())
        assert len(pages[0].text()) > 0
        assert pages[0].image_coverage() > 0.9
        assert is_native_text_document(doc, CONFIG) is False
    finally:
        doc.close()


def test_the_same_scan_without_the_ocr_layer_is_also_not_native():
    doc = FitzOpener().open(DATA / "EK Valentyna Tymchenko.pdf")
    try:
        assert is_native_text_document(doc, CONFIG) is False
    finally:
        doc.close()


def test_a_genuine_digital_contract_is_trusted_as_native_text():
    doc = FitzOpener().open(DATA / "Vertragsunterlagen.pdf")
    try:
        assert doc.page_count == 40
        assert is_native_text_document(doc, CONFIG) is True
    finally:
        doc.close()


def test_page_rotation_round_trips_through_the_handle():
    doc = FitzOpener().open(DATA / "Vertragsunterlagen.pdf")
    try:
        page = next(doc.pages())
        assert page.rotation == 0
        page.set_rotation(90)
        assert page.rotation == 90
    finally:
        doc.close()


def test_render_jpeg_b64_produces_a_non_empty_base64_string():
    doc = FitzOpener().open(DATA / "Vertragsunterlagen.pdf")
    try:
        page = next(doc.pages())
        jpeg_b64 = page.render_jpeg_b64(dpi=100, quality=70)
        assert isinstance(jpeg_b64, str)
        assert len(jpeg_b64) > 100
    finally:
        doc.close()


def test_render_bgr_produces_a_height_by_width_by_3_array():
    doc = FitzOpener().open(DATA / "Vertragsunterlagen.pdf")
    try:
        page = next(doc.pages())
        img = page.render_bgr(dpi=100)
        assert img.ndim == 3
        assert img.shape[2] == 3
    finally:
        doc.close()
