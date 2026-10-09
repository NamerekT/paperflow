"""`paperflow-rare-tags` - which tags make it into the report."""

from paperflow.apps.cli.rare_tags import select_rare
from paperflow.domain import constants


def tag(name: str, document_count: int) -> dict:
    return {"id": abs(hash(name)) % 1000, "name": name, "document_count": document_count}


def test_only_tags_at_or_below_the_limit_are_reported_rarest_first():
    tags = [tag("Miete", 12), tag("Zeugnis", 1), tag("Nebenkosten", 2), tag("Altvertrag", 0)]

    rare = select_rare(tags, max_documents=2)

    assert [row["name"] for row in rare] == ["Altvertrag", "Zeugnis", "Nebenkosten"]


def test_tags_with_the_same_count_are_sorted_by_name():
    tags = [tag("zeugnis", 1), tag("Ausweis", 1)]

    assert [row["name"] for row in select_rare(tags, max_documents=1)] == ["Ausweis", "zeugnis"]


def test_technical_tags_are_never_reported():
    tags = [
        tag(constants.TAG_REVIEW_PARENT, 0),
        tag("→ Miete", 1),
        tag("Is due soon (Technical tag, not to classify)", 0),
        tag("Zeugnis", 1),
    ]

    assert [row["name"] for row in select_rare(tags, max_documents=1)] == ["Zeugnis"]
