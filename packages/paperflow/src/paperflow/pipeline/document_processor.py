"""Processing one document end to end: download, run the engine, write the result back.

Mirrors the old batch cycle - download the original, run the cascade, classify, PATCH -
and is now the only place that cycle exists. The live webhook and the nightly catch-up
job both call `DocumentProcessor.process` and nothing else, so the two paths cannot drift
the way a live upload and a batch script once did.

Recoverable failures are swallowed here, each with its own narrow `try`, exactly where
the old code drew the line: a Frist that could not be written, a reminder that could not
be created, none of them should undo an otherwise successful write. What is NOT caught
here - the engine raising, the archive being unreachable for the main write - is
deliberate: that belongs to the caller, which decides what an unrecoverable failure means
for the queue.
"""

import logging
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from docflow.domain.models import DocumentAnalysis, DocumentClassification
from docflow.pipeline.document_workflow import DocumentWorkflow
from paperflow.adapters.google.tasks import GoogleTasks
from paperflow.adapters.paperless.custom_fields import PaperlessCustomFields
from paperflow.adapters.paperless.documents import PaperlessDocuments
from paperflow.adapters.paperless.taxonomy import PaperlessTaxonomy
from paperflow.domain import constants
from paperflow.domain.models import CustomFieldSet, CustomFieldValue, ResolvedClassification
from paperflow.errors import PaperlessError
from paperflow.logging_setup import DocumentLogAdapter
from paperflow.pipeline import progress
from paperflow.pipeline.custom_field_mapping import cascade_field_values

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(
        self,
        workflow: DocumentWorkflow,
        documents: PaperlessDocuments,
        custom_fields: PaperlessCustomFields,
        taxonomy: PaperlessTaxonomy,
        fields: CustomFieldSet,
        google_tasks: GoogleTasks | None,
        *,
        apply_changes: bool,
        progress_path: Path,
    ) -> None:
        self._workflow = workflow
        self._documents = documents
        self._custom_fields = custom_fields
        self._taxonomy = taxonomy
        self._fields = fields
        self._google_tasks = google_tasks
        self._apply_changes = apply_changes
        self._progress_path = progress_path

    def already_processed(self, doc_id: int) -> bool:
        document = self._documents.get(doc_id)
        return bool(self._custom_fields.value_on(document, self._fields.cascade_flag.id))

    def process(self, doc_id: int, *, force: bool) -> None:
        log = DocumentLogAdapter(logger, doc_id)

        if not force and self.already_processed(doc_id):
            log.info(
                f"already flagged {constants.FIELD_CASCADE_FLAG!r} - skipping "
                "(force=true in the webhook body to redo it)"
            )
            progress.append(
                self._progress_path,
                {"doc_id": doc_id, "applied": False, "skipped": True, "error": None},
            )
            return

        with tempfile.TemporaryDirectory(prefix=f"paperflow_{doc_id}_") as tmp_dir:
            pdf_path = Path(tmp_dir) / f"{doc_id}.pdf"
            self._documents.download_original(doc_id, pdf_path)
            log.info("original downloaded, running the engine")

            analysis = self._workflow.run(pdf_path)
            log.info(
                f"engine finished: content_by={analysis.cascade.content_by.value} "
                f"confidence={analysis.cascade.avg_confidence_pct}% "
                f"accuracy={analysis.cascade.avg_accuracy_pct}%"
            )

            resolved = self._resolve(analysis.classification)
            log.info(
                f"classification resolved: title={resolved.title!r} tags={len(resolved.tag_ids)} "
                f"correspondent_id={resolved.correspondent_id} "
                f"document_type_id={resolved.document_type_id}"
            )

            if self._apply_changes:
                self._apply(doc_id, analysis, resolved, log)
            else:
                log.info("apply_changes is off - not writing to Paperless (dry run)")

        progress.append(
            self._progress_path,
            {
                "doc_id": doc_id,
                "content_by": analysis.cascade.content_by.value,
                "avg_confidence_pct": analysis.cascade.avg_confidence_pct,
                "avg_accuracy_pct": analysis.cascade.avg_accuracy_pct,
                "title": resolved.title,
                "applied": self._apply_changes,
                "error": None,
            },
        )
        log.info("done")

    def _resolve(self, classification: DocumentClassification) -> ResolvedClassification:
        """Turn the engine's names into archive ids.

        This is the whole of what makes the engine reusable against a different
        document manager: it produces names, and only here - never upstream - do they
        become ids in a particular Paperless instance's database.
        """
        tag_ids = tuple(
            tag_id
            for tag_id in (
                self._taxonomy.get_or_create_id("tags", name) for name in classification.tags
            )
            if tag_id is not None
        )
        return ResolvedClassification(
            title=classification.title,
            correspondent_id=self._taxonomy.get_or_create_id(
                "correspondents", classification.correspondent or ""
            ),
            document_type_id=self._taxonomy.get_or_create_id(
                "document_types", classification.document_type or ""
            ),
            tag_ids=tag_ids,
            date=classification.date,
        )

    def _apply(
        self,
        doc_id: int,
        analysis: DocumentAnalysis,
        resolved: ResolvedClassification,
        log: DocumentLogAdapter,
    ) -> None:
        values = cascade_field_values(
            self._fields, self._custom_fields, analysis.cascade, mark_done=True
        )
        payload = self._patch_payload(resolved, content=analysis.cascade.content)
        self._custom_fields.patch(doc_id, values, extra_payload=payload)
        log.info("written to Paperless")

        if analysis.frist is not None:
            self._write_frist(doc_id, analysis.frist, resolved.title, log)

    def _write_frist(self, doc_id: int, frist: date, title: str, log: DocumentLogAdapter) -> None:
        # A separate PATCH, deliberately: a failure here must not look like the main
        # write - which has already succeeded - failed too.
        try:
            self._custom_fields.patch(
                doc_id, [CustomFieldValue(field=self._fields.frist.id, value=frist.isoformat())]
            )
            log.info(f"Frist: {frist}")
        except PaperlessError as exc:
            log.error(f"could not write the Frist field: {exc}")
            return

        if self._google_tasks is not None:
            try:
                self._google_tasks.create_task(title, frist, notes=self._documents.url_for(doc_id))
            except Exception as exc:
                log.error(f"Google Tasks: could not create a reminder: {exc}")

    @staticmethod
    def _patch_payload(resolved: ResolvedClassification, *, content: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "content": content,
            "title": resolved.title,
            "created": resolved.date.isoformat(),
            "tags": list(resolved.tag_ids),
        }
        if resolved.correspondent_id:
            payload["correspondent"] = resolved.correspondent_id
        if resolved.document_type_id:
            payload["document_type"] = resolved.document_type_id
        return payload
