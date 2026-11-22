"""Closing a document must release what MuPDF cached for it.

MuPDF keeps decoded page images in a process-wide store that outlives the document;
a reopened file gets new keys, so in a long-running service the store grew by roughly
the decoded size of every document until it hit its cap (PPF-9).
"""

import fitz
import pytest

from docflow.adapters.pdf.fitz_document import FitzDocument


def test_close_empties_the_mupdf_store(monkeypatch: pytest.MonkeyPatch) -> None:
    shrinks: list[int] = []
    monkeypatch.setattr(fitz.TOOLS, "store_shrink", lambda percent: shrinks.append(percent))
    source = fitz.open()
    source.new_page()

    FitzDocument(source).close()

    assert shrinks == [100]
    assert source.is_closed


def test_close_empties_the_store_even_if_closing_the_document_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shrinks: list[int] = []
    monkeypatch.setattr(fitz.TOOLS, "store_shrink", lambda percent: shrinks.append(percent))
    source = fitz.open()
    source.close()

    with pytest.raises(ValueError):
        FitzDocument(source).close()

    assert shrinks == [100]
