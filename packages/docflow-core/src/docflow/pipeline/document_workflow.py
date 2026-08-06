"""The engine's single entry point: open a PDF, run it through the cascade, work out
what it is, and look for a deadline.

Everything upstream - `native_text`, `cascade`, the four `llm_tasks` - is a building
block; this is where they compose into the one thing an application actually wants, a
complete `DocumentAnalysis`, from nothing but a file on disk. Nothing here writes
anywhere or talks to anything but the model gateway - see `docflow.ports` for why that
boundary is the one that keeps the engine reusable against a document manager other than
Paperless.
"""

from pathlib import Path

from docflow.domain.models import DocumentAnalysis
from docflow.llm_tasks.classify import ClassifyTask
from docflow.llm_tasks.frist import FristTask
from docflow.pipeline.cascade import OcrCascade
from docflow.ports.pdf import PdfOpener
from docflow.ports.taxonomy import TaxonomySource


class DocumentWorkflow:
    def __init__(
        self,
        opener: PdfOpener,
        cascade: OcrCascade,
        classify: ClassifyTask,
        frist: FristTask,
        taxonomy: TaxonomySource,
    ) -> None:
        self._opener = opener
        self._cascade = cascade
        self._classify = classify
        self._frist = frist
        self._taxonomy = taxonomy

    def run(self, path: str | Path) -> DocumentAnalysis:
        doc = self._opener.open(path)
        try:
            cascade_result = self._cascade.run(doc)
        finally:
            doc.close()

        # The taxonomy is read fresh for every document rather than cached on this
        # instance: the archive gains tags and correspondents as it is worked through,
        # and a stale snapshot is exactly what causes the classifier to invent a
        # near-duplicate of something that was created five minutes ago.
        snapshot = self._taxonomy.snapshot()
        classification = self._classify.classify(cascade_result.content, snapshot)
        frist = self._frist.extract(cascade_result.content, classification.date)

        return DocumentAnalysis(cascade=cascade_result, classification=classification, frist=frist)
