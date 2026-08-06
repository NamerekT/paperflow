"""Deciding whether a PDF page needs OCR at all, or already carries good text.

This is a decision about which branch the pipeline takes, not a property of
the OCR engine - which is why it lives here rather than inside
`docflow.adapters.ocr.rapidocr_dual`, where it used to sit as two methods on
`DualRapidOcrEngine` that had nothing to do with recognising anything.
"""

from docflow.domain.policies import NativeTextConfig
from docflow.ports.pdf import PageHandle, PdfDocument


def extract_native_text(page: PageHandle, config: NativeTextConfig) -> str | None:
    """A page with an embedded text layer that isn't mostly image is a born-digital
    export and needs no OCR - its text is already accurate, free and instant.

    A real case that broke a version of this check which looked only at whether text
    was present: a scan someone later "printed to PDF" gained a text layer, but it was
    SOMEONE ELSE'S embedded OCR output, not genuine digital text - "Kateryha" instead
    of "Kateryna", "welteren" instead of "weiteren". The image underneath still
    covered 100% of the page, which is what gives a case like this away despite the
    presence of text - hence the image-coverage check below runs even when text exists.

    Returns the text if it exceeds `config.min_chars` AND images cover no more than
    `config.max_image_coverage` of the page, otherwise None (the page is a scan)."""
    text = page.text()
    if len(text) < config.min_chars:
        return None
    if page.image_coverage() > config.max_image_coverage:
        return None
    return text


def is_native_text_document(doc: PdfDocument, config: NativeTextConfig) -> bool:
    """Document-level backstop on top of `extract_native_text`'s per-page image check: if
    some other tool ever embeds an invisible OCR layer without a full-page image under it
    (e.g. a vectorised scan), that check alone wouldn't catch it. A born-digital export
    carries text on nearly every page; a scan's text pages (if any) are sparse and
    accidental - so the document is trusted only if >= `config.min_page_ratio` of its
    pages carry a text layer. Otherwise the whole document goes through the OCR cascade,
    even the individual pages where text was found."""
    if doc.page_count == 0:
        return False
    with_text = sum(1 for page in doc.pages() if extract_native_text(page, config) is not None)
    return (with_text / doc.page_count) >= config.min_page_ratio
