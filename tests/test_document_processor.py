"""`DocumentProcessor.process` - the download/engine/resolve/write cycle.

Exercised entirely against fakes standing in for the Paperless adapters, the engine and
Google Tasks: the decisions being pinned here (skip-if-already-done, resolving names to
ids, which failures are allowed to swallow themselves) are orchestration, not I/O, and
belong to this module alone.
"""

from datetime import date
from pathlib import Path

from docflow.domain.models import (
    CascadeResult,
    ContentSource,
    DocumentAnalysis,
    DocumentClassification,
    PageReport,
)
from paperflow.domain import constants
from paperflow.domain.models import CustomField, CustomFieldSet
from paperflow.errors import PaperlessError
from paperflow.pipeline.document_processor import DocumentProcessor


def make_fields() -> CustomFieldSet:
    def cf(name: str, field_id: int, data_type: str = "string") -> CustomField:
        return CustomField(id=field_id, name=name, data_type=data_type, extra_data=None)

    return CustomFieldSet(
        by_name={
            constants.FIELD_CONTENT_BY: cf(constants.FIELD_CONTENT_BY, 1, "select"),
            constants.FIELD_MIXED_CREATED_BY: cf(constants.FIELD_MIXED_CREATED_BY, 2, "longtext"),
            constants.FIELD_AVG_CONFIDENCE: cf(constants.FIELD_AVG_CONFIDENCE, 3, "float"),
            constants.FIELD_AVG_ACCURACY: cf(constants.FIELD_AVG_ACCURACY, 4, "float"),
            constants.FIELD_CASCADE_FLAG: cf(constants.FIELD_CASCADE_FLAG, 5, "boolean"),
            constants.FIELD_FRIST: cf(constants.FIELD_FRIST, 6, "date"),
        }
    )


def make_analysis(*, frist=None, mixed_by=None) -> DocumentAnalysis:
    cascade = CascadeResult(
        content="hallo welt",
        content_by=ContentSource.RAPIDOCR,
        mixed_by=mixed_by,
        avg_confidence_pct=99.0,
        avg_accuracy_pct=98.0,
        page_reports=(PageReport(page=1, source="RapidOCR", mean=0.99, p10=0.98),),
    )
    classification = DocumentClassification(
        title="Bescheid Januar 2026",
        correspondent="Jobcenter Musterstadt",
        document_type="Bescheid",
        tags=("Bürgergeld", "Personal", "Sample"),
        family_member="Sample",
        date=date(2026, 1, 15),
    )
    return DocumentAnalysis(cascade=cascade, classification=classification, frist=frist)


class FakeDocuments:
    def __init__(self, meta_by_id=None):
        self.meta_by_id = meta_by_id or {}
        self.downloaded: list[int] = []

    def get(self, doc_id):
        return self.meta_by_id.get(doc_id, {})

    def download_original(self, doc_id, dest: Path):
        self.downloaded.append(doc_id)
        dest.write_bytes(b"%PDF-1.7 fake")
        return dest

    def url_for(self, doc_id):
        return f"https://paperless.test/documents/{doc_id}/details"


class FakeCustomFields:
    def __init__(self):
        self.patches: list[tuple[int, list, dict | None]] = []
        self.fail_on_call: int | None = None

    @staticmethod
    def value_on(document, field_id):
        for entry in document.get("custom_fields") or []:
            if entry.get("field") == field_id:
                return entry.get("value")
        return None

    @staticmethod
    def select_option_value(field, label):
        return label

    def patch(self, doc_id, values, extra_payload=None):
        if self.fail_on_call is not None and len(self.patches) == self.fail_on_call:
            self.patches.append((doc_id, list(values), extra_payload))
            raise PaperlessError("simulated failure")
        self.patches.append((doc_id, list(values), extra_payload))
        return {}


class FakeTaxonomy:
    def __init__(self):
        self._ids: dict[tuple[str, str], int] = {}
        self._next = 100

    def get_or_create_id(self, resource, name):
        if not name:
            return None
        key = (resource, name)
        if key not in self._ids:
            self._ids[key] = self._next
            self._next += 1
        return self._ids[key]


class FakeWorkflow:
    def __init__(self, analysis: DocumentAnalysis):
        self.analysis = analysis
        self.calls: list[Path] = []

    def run(self, path):
        self.calls.append(Path(path))
        return self.analysis


class FakeGoogleTasks:
    def __init__(self):
        self.created: list[tuple[str, date, str | None]] = []

    def create_task(self, title, due, *, notes=None):
        self.created.append((title, due, notes))
        return "task-1"


def make_processor(
    analysis=None, *, apply_changes=True, google_tasks=None, tmp_path=None
) -> tuple[DocumentProcessor, dict]:
    workflow = FakeWorkflow(analysis or make_analysis())
    documents = FakeDocuments()
    custom_fields = FakeCustomFields()
    taxonomy = FakeTaxonomy()
    processor = DocumentProcessor(
        workflow,
        documents,
        custom_fields,
        taxonomy,
        make_fields(),
        google_tasks,
        apply_changes=apply_changes,
        progress_path=(tmp_path or Path(".")) / "progress.jsonl",
    )
    return processor, {
        "workflow": workflow,
        "documents": documents,
        "custom_fields": custom_fields,
        "taxonomy": taxonomy,
    }


def test_already_flagged_document_is_skipped_without_force(tmp_path):
    processor, parts = make_processor(tmp_path=tmp_path)
    parts["documents"].meta_by_id[1] = {"custom_fields": [{"field": 5, "value": True}]}

    processor.process(1, force=False)

    assert parts["workflow"].calls == []
    assert parts["documents"].downloaded == []


def test_force_reprocesses_an_already_flagged_document(tmp_path):
    processor, parts = make_processor(tmp_path=tmp_path)
    parts["documents"].meta_by_id[1] = {"custom_fields": [{"field": 5, "value": True}]}

    processor.process(1, force=True)

    assert parts["documents"].downloaded == [1]


def test_happy_path_writes_content_classification_and_cascade_fields(tmp_path):
    processor, parts = make_processor(tmp_path=tmp_path)

    processor.process(1, force=False)

    assert parts["documents"].downloaded == [1]
    doc_id, values, payload = parts["custom_fields"].patches[0]
    assert doc_id == 1
    assert payload["content"] == "hallo welt"
    assert payload["title"] == "Bescheid Januar 2026"
    assert payload["created"] == "2026-01-15"
    # correspondent/document_type/tags all resolved through the taxonomy fake
    assert payload["correspondent"] == parts["taxonomy"].get_or_create_id(
        "correspondents", "Jobcenter Musterstadt"
    )
    assert len(payload["tags"]) == 3
    field_ids = {value.field for value in values}
    assert field_ids == {1, 3, 4, 5}  # content_by, avg_confidence, avg_accuracy, cascade_flag


def test_dry_run_does_not_write_anything(tmp_path):
    processor, parts = make_processor(apply_changes=False, tmp_path=tmp_path)

    processor.process(1, force=False)

    assert parts["documents"].downloaded == [1]
    assert parts["custom_fields"].patches == []


def test_frist_is_written_as_a_separate_patch_and_creates_a_reminder(tmp_path):
    analysis = make_analysis(frist=date(2026, 2, 1))
    google_tasks = FakeGoogleTasks()
    processor, parts = make_processor(analysis, google_tasks=google_tasks, tmp_path=tmp_path)

    processor.process(1, force=False)

    assert len(parts["custom_fields"].patches) == 2
    _frist_doc_id, frist_values, frist_payload = parts["custom_fields"].patches[1]
    assert frist_values[0].field == 6
    assert frist_values[0].value == "2026-02-01"
    assert frist_payload is None
    assert google_tasks.created == [
        ("Bescheid Januar 2026", date(2026, 2, 1), "https://paperless.test/documents/1/details")
    ]


def test_no_frist_means_no_second_patch_and_no_reminder(tmp_path):
    google_tasks = FakeGoogleTasks()
    processor, parts = make_processor(google_tasks=google_tasks, tmp_path=tmp_path)

    processor.process(1, force=False)

    assert len(parts["custom_fields"].patches) == 1
    assert google_tasks.created == []


def test_a_failed_frist_write_does_not_create_a_reminder(tmp_path):
    analysis = make_analysis(frist=date(2026, 2, 1))
    google_tasks = FakeGoogleTasks()
    processor, parts = make_processor(analysis, google_tasks=google_tasks, tmp_path=tmp_path)
    parts["custom_fields"].fail_on_call = 1  # the second patch() call - the Frist write

    processor.process(1, force=False)  # must not raise

    assert google_tasks.created == []


def test_a_google_tasks_failure_does_not_fail_the_document(tmp_path):
    class BrokenGoogleTasks:
        def create_task(self, title, due, *, notes=None):
            raise RuntimeError("no refresh token")

    analysis = make_analysis(frist=date(2026, 2, 1))
    processor, parts = make_processor(analysis, google_tasks=BrokenGoogleTasks(), tmp_path=tmp_path)

    processor.process(1, force=False)  # must not raise

    assert len(parts["custom_fields"].patches) == 2  # the Frist field was still written


def test_a_mixed_result_writes_the_mixed_created_by_field(tmp_path):
    analysis = make_analysis(mixed_by="p.1: RapidOCR\np.2: Gemma 31B (LiteLLM: test)")
    processor, parts = make_processor(analysis, tmp_path=tmp_path)

    processor.process(1, force=False)

    _, values, _ = parts["custom_fields"].patches[0]
    field_ids = {value.field for value in values}
    assert 2 in field_ids  # Mixed Created by


def test_progress_is_appended_after_a_successful_run(tmp_path):
    processor, _parts = make_processor(tmp_path=tmp_path)

    processor.process(1, force=False)

    log_path = tmp_path / "progress.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"applied": true' in lines[0]
