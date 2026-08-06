"""Deciding whether a PDF page needs OCR at all, or already carries good text.

This is a decision about which branch the pipeline takes, not a property of
the OCR engine - which is why it lives here rather than inside
`docflow.adapters.ocr.rapidocr_dual`, where it used to sit as two methods on
`DualRapidOcrEngine` that had nothing to do with recognising anything.
"""

from docflow.domain.policies import NativeTextConfig
from docflow.ports.pdf import PageHandle, PdfDocument


def extract_native_text(page: PageHandle, config: NativeTextConfig) -> str | None:
    """If a PDF page already carries an embedded text layer AND the page is not, in
    substance, a picture (see `image_coverage` below), it is not a paper scan but a
    programmatically generated/exported PDF (for example, an official letter downloaded
    as a PDF rather than scanned on a printer) - it needs no OCR at all: the text is
    already 100% accurate, free and instant. Running such a page through RapidOCR, let
    alone the vision model, would spend time (and, for the vision model, money/tokens)
    on text that is GUARANTEED to be worse than what is already in the file.

    A real case that caught the previous version of this check (when it looked only at
    whether text was present): a scan that someone later "printed to PDF" - the page
    gained a text layer, but it turned out to be SOMEONE ELSE'S, embedded OCR output,
    not genuine digital text: "Valenign" instead of "Valentyna", "welteren" instead of
    "weiteren" - classic OCR mistakes, including a mangled name. The image underneath
    the text covered 100% of the page - which is what gave it away as a scan, despite
    the presence of text.

    Returns the text if it is longer than `config.min_chars` characters AND images cover
    no more than `config.max_image_coverage` of the page area, otherwise None - meaning
    the page really is a scan and needs to be recognised as usual."""
    text = page.text()
    if len(text) < config.min_chars:
        return None
    if page.image_coverage() > config.max_image_coverage:
        return None
    return text


def is_native_text_document(doc: PdfDocument, config: NativeTextConfig) -> bool:
    """A second, backstop layer on top of `extract_native_text`'s image check - decides AT
    THE LEVEL OF THE WHOLE DOCUMENT whether the PDF's text layer can be trusted, not just
    at the level of one page.

    The image check in `extract_native_text` already catches the main case on its own (a
    searchable PDF - text laid over an image covering the whole sheet), but it is not
    relied on ALONE: if some other tool ever embeds an invisible OCR layer without a
    full-page image underneath it (for example, a vectorised scan), this check backstops
    it by requiring the text to be present consistently on almost ALL pages of the
    document, which is typical specifically of born-digital documents (a Word/SAP/LaTeX
    export), not of a scan, whose pages with text (if there are any at all) are usually
    isolated/accidental.

    The document is considered born-digital only if the fraction of pages carrying a text
    layer is >= `config.min_page_ratio`. Otherwise the whole document is treated as a scan
    (even the individual pages where text was found after all) - and goes through the
    ordinary OCR cascade, as it should for a scan."""
    if doc.page_count == 0:
        return False
    with_text = sum(1 for page in doc.pages() if extract_native_text(page, config) is not None)
    return (with_text / doc.page_count) >= config.min_page_ratio
