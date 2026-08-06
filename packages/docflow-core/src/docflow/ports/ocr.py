"""Recognising text on a rendered page, independent of which engine does it.

Kept to one method deliberately: the cascade needs a page read locally and
scored, and nothing else. `experiments/` shows that a docling, Gemini or
Ollama backend has each already been tried by hand; behind this port, adding a
second engine is about forty lines plus one line in composition, not a change
to the pipeline.
"""

from typing import Protocol

from docflow.domain.models import OcrLine
from docflow.ports.pdf import PageHandle


class PageOcr(Protocol):
    #: Attribution label for a page kept from this engine - goes into
    #: `PageReport.source` and, for an all-local document, into `ContentSource`.
    name: str

    def read_lines(self, page: PageHandle, *, dpi: int) -> list[OcrLine]: ...
