"""Deciding whether a PDF needs OCR at all, or already carries a trustworthy text layer.

Two levels, each pinned separately: `extract_native_text` on one page, and
`is_native_text_document`'s backstop that a text layer only counts if it is present
across nearly the whole document - the check added after a "printed to PDF" scan carried
someone else's OCR text on one page only.
"""

from docflow.domain.policies import NativeTextConfig
from docflow.pipeline.native_text import extract_native_text, is_native_text_document

CONFIG = NativeTextConfig(min_chars=40, max_image_coverage=0.5, min_page_ratio=0.8)


class FakePage:
    def __init__(self, number=1, text="", image_coverage=0.0):
        self.number = number
        self._text = text
        self._image_coverage = image_coverage

    def text(self):
        return self._text

    def image_coverage(self):
        return self._image_coverage


class FakeDocument:
    def __init__(self, pages):
        self._pages = pages

    @property
    def page_count(self):
        return len(self._pages)

    def pages(self):
        return iter(self._pages)


LONG_TEXT = "Bedarfsberechnung " * 5  # well over 40 characters


def test_short_text_is_not_trusted():
    assert extract_native_text(FakePage(text="short"), CONFIG) is None


def test_long_text_with_low_image_coverage_is_trusted():
    assert extract_native_text(FakePage(text=LONG_TEXT, image_coverage=0.1), CONFIG) == LONG_TEXT


def test_long_text_over_a_full_page_image_is_a_searchable_pdf_not_native_text():
    # The "printed to PDF" case: a scan with someone else's OCR layer laid over it.
    # The image covers the whole page, so the text layer is not trusted however long it is.
    assert extract_native_text(FakePage(text=LONG_TEXT, image_coverage=1.0), CONFIG) is None


def test_empty_document_is_not_native():
    assert is_native_text_document(FakeDocument([]), CONFIG) is False


def test_document_with_text_on_every_page_is_native():
    pages = [FakePage(number=i, text=LONG_TEXT) for i in range(1, 4)]
    assert is_native_text_document(FakeDocument(pages), CONFIG) is True


def test_document_with_text_on_only_one_of_several_pages_is_not_native():
    # A scan with a stray, accidental text layer on a single page - the failure mode
    # this backstop exists to catch, distinct from the per-page image-coverage check.
    pages = [FakePage(number=1, text=LONG_TEXT)] + [
        FakePage(number=i, text="") for i in range(2, 6)
    ]
    assert is_native_text_document(FakeDocument(pages), CONFIG) is False


def test_ratio_exactly_at_the_threshold_counts_as_native():
    pages = [FakePage(number=i, text=LONG_TEXT) for i in range(1, 5)] + [
        FakePage(number=5, text="")
    ]
    assert is_native_text_document(FakeDocument(pages), CONFIG) is True  # 4/5 == 0.8
