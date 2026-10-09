"""Tag review: keeping the classifier's inventions visible, and merging them away.

Three pieces are pinned here. Technical tags never reach the classifier's prompt. A tag
the classifier invents is created under the review parent. And `TagReviewer` - the
tag-review webhook's handler - replaces a tag renamed to "→ Target" with Target and takes
the review parent off a document that no longer has anything under review.
"""

from paperflow.adapters.paperless.taxonomy import PaperlessTaxonomy
from paperflow.apps.service.worker import QueueWorker
from paperflow.domain import constants
from paperflow.domain.models import TagRef
from paperflow.domain.tags import is_technical_tag, merge_target
from paperflow.pipeline.tag_review import TagReviewer

REVIEW = 1


def test_merge_target_is_the_name_after_the_arrow():
    assert merge_target("→ Miete") == "Miete"
    assert merge_target("→Miete") == "Miete"
    assert merge_target("Miete") is None
    assert merge_target("→ ") is None


def test_technical_tags_are_recognised():
    assert is_technical_tag(constants.TAG_REVIEW_PARENT)
    assert is_technical_tag("→ Miete")
    assert is_technical_tag("Is due soon (Technical tag, not to classify)")
    assert not is_technical_tag("Miete")


# ── PaperlessTaxonomy, against a fake transport ──────────────────────────────


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class FakeTransport:
    def __init__(self, tags):
        self.tags = list(tags)
        self.posted: list[dict] = []

    def paginate(self, path, *, params=None, versioned=True):
        yield from (self.tags if path == "/api/tags/" else [])

    def get(self, path, *, params=None, versioned=False):
        wanted = params["name__iexact"].casefold()
        return FakeResponse(
            {"results": [tag for tag in self.tags if tag["name"].casefold() == wanted]}
        )

    def post(self, path, *, json, versioned=False):
        self.posted.append(json)
        created = {"id": 900 + len(self.posted), "parent": None, **json}
        self.tags.append(created)
        return FakeResponse(created)


def test_the_snapshot_hides_technical_tags_from_the_classifier():
    transport = FakeTransport(
        [
            {"id": REVIEW, "name": constants.TAG_REVIEW_PARENT, "parent": None},
            {"id": 2, "name": "Miete", "parent": None},
            {"id": 3, "name": "→ Miete", "parent": REVIEW},
            {"id": 4, "name": "Is due soon (Technical tag, not to classify)", "parent": None},
            {"id": 5, "name": "Nebenkosten", "parent": REVIEW},
        ]
    )

    assert PaperlessTaxonomy(transport).snapshot().tags == ("Miete", "Nebenkosten")


def test_an_existing_tag_is_returned_with_its_parent_and_nothing_is_created():
    transport = FakeTransport([{"id": 5, "name": "Nebenkosten", "parent": REVIEW}])

    tag = PaperlessTaxonomy(transport).get_or_create_tag("nebenkosten", new_parent_id=REVIEW)

    assert tag == TagRef(id=5, parent_id=REVIEW)
    assert transport.posted == []


def test_a_missing_tag_is_created_under_the_given_parent():
    transport = FakeTransport([])

    tag = PaperlessTaxonomy(transport).get_or_create_tag("Nebenkosten", new_parent_id=REVIEW)

    assert tag == TagRef(id=901, parent_id=REVIEW)
    assert transport.posted[0]["name"] == "Nebenkosten"
    assert transport.posted[0]["parent"] == REVIEW


# ── TagReviewer, against fake adapters ───────────────────────────────────────


class FakeDocuments:
    def __init__(self, tags_by_doc: dict[int, list[int]]):
        self.tags_by_doc = tags_by_doc
        self.patches: list[tuple[int, dict]] = []

    def get(self, doc_id):
        return {"id": doc_id, "tags": list(self.tags_by_doc[doc_id])}

    def patch(self, doc_id, payload):
        self.patches.append((doc_id, payload))
        self.tags_by_doc[doc_id] = list(payload["tags"])
        return {}


class FakeTaxonomy:
    def __init__(self, documents: FakeDocuments, names: dict[int, str], parents: dict[int, int]):
        self._documents = documents
        self.names = names
        self.parents = parents
        self.deleted: list[int] = []

    def _record(self, tag_id):
        return {
            "id": tag_id,
            "name": self.names[tag_id],
            "parent": self.parents.get(tag_id),
            "document_count": sum(tag_id in tags for tags in self._documents.tags_by_doc.values()),
        }

    def tags_by_id(self, tag_ids):
        return {tag_id: self._record(tag_id) for tag_id in tag_ids if tag_id in self.names}

    def find_tag(self, name):
        for tag_id, tag_name in self.names.items():
            if tag_name.casefold() == name.casefold():
                return self._record(tag_id)
        return None

    def delete_tag(self, tag_id):
        self.deleted.append(tag_id)
        del self.names[tag_id]


MIETE, ARROW_MIETE, NEBENKOSTEN, PERSONAL = 2, 3, 5, 6


def make_reviewer(tags_by_doc, *, parents=None, apply_changes=True):
    documents = FakeDocuments(tags_by_doc)
    taxonomy = FakeTaxonomy(
        documents,
        names={
            REVIEW: constants.TAG_REVIEW_PARENT,
            MIETE: "Miete",
            ARROW_MIETE: "→ Miete",
            NEBENKOSTEN: "Nebenkosten",
            PERSONAL: "Personal",
        },
        parents=parents or {},
    )
    reviewer = TagReviewer(documents, taxonomy, REVIEW, apply_changes=apply_changes)
    return reviewer, documents, taxonomy


def test_a_merge_tag_is_replaced_by_its_target_and_deleted_once_unused():
    reviewer, documents, taxonomy = make_reviewer({10: [ARROW_MIETE, PERSONAL]})

    reviewer.review(10)

    assert documents.patches == [(10, {"tags": [MIETE, PERSONAL]})]
    assert taxonomy.deleted == [ARROW_MIETE]


def test_a_merge_tag_other_documents_still_carry_is_not_deleted():
    reviewer, documents, taxonomy = make_reviewer({10: [ARROW_MIETE], 11: [ARROW_MIETE]})

    reviewer.review(10)

    assert documents.tags_by_doc[10] == [MIETE]
    assert taxonomy.deleted == []


def test_a_target_already_on_the_document_is_not_duplicated():
    reviewer, documents, _ = make_reviewer({10: [MIETE, ARROW_MIETE]})

    reviewer.review(10)

    assert documents.patches == [(10, {"tags": [MIETE]})]


def test_a_merge_tag_without_an_existing_target_is_left_alone():
    reviewer, documents, taxonomy = make_reviewer({10: [ARROW_MIETE, PERSONAL]})
    del taxonomy.names[MIETE]

    reviewer.review(10)

    assert documents.patches == []
    assert taxonomy.deleted == []


def test_the_review_parent_is_removed_once_nothing_under_it_is_left():
    reviewer, documents, _ = make_reviewer({10: [NEBENKOSTEN, REVIEW, PERSONAL]})

    reviewer.review(10)

    assert documents.patches == [(10, {"tags": [NEBENKOSTEN, PERSONAL]})]


def test_the_review_parent_stays_while_a_tag_under_it_remains():
    reviewer, documents, _ = make_reviewer(
        {10: [NEBENKOSTEN, REVIEW, PERSONAL]}, parents={NEBENKOSTEN: REVIEW}
    )

    reviewer.review(10)

    assert documents.patches == []


def test_merging_the_last_tag_under_review_also_removes_the_review_parent():
    reviewer, documents, taxonomy = make_reviewer(
        {10: [ARROW_MIETE, REVIEW, PERSONAL]}, parents={ARROW_MIETE: REVIEW}
    )

    reviewer.review(10)

    assert documents.patches == [(10, {"tags": [MIETE, PERSONAL]})]
    assert taxonomy.deleted == [ARROW_MIETE]


def test_the_worker_runs_reviews_and_cascade_jobs_in_arrival_order(tmp_path):
    calls: list[tuple] = []

    class RecordingProcessor:
        def process(self, doc_id, *, force):
            calls.append(("cascade", doc_id, force))

    class FailingThenRecordingReviewer:
        def review(self, doc_id):
            calls.append(("review", doc_id))
            if doc_id == 1:
                raise RuntimeError("simulated failure")

    class StubSettings:
        progress_path = tmp_path / "progress.jsonl"

    worker = QueueWorker(RecordingProcessor(), FailingThenRecordingReviewer(), StubSettings())
    worker.enqueue_tag_review(1)
    worker.enqueue(2, force=True)
    worker.enqueue_tag_review(3)
    worker.start()
    worker._queue.join()

    assert calls == [("review", 1), ("cascade", 2, True), ("review", 3)]


def test_a_dry_run_changes_nothing():
    reviewer, documents, taxonomy = make_reviewer({10: [ARROW_MIETE, REVIEW]}, apply_changes=False)

    reviewer.review(10)

    assert documents.patches == []
    assert taxonomy.deleted == []
