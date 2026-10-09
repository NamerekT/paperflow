"""`ClassifyTask` - which of the model's tags survive.

The prompt already asks the model not to tag a document with its own sender or its own
kind. It does so anyway, often enough to fill the archive with tags that say nothing the
correspondent and the document type do not already say - so the rule is enforced in code,
and pinned here.
"""

from docflow.domain.policies import RetryPolicy
from docflow.llm_tasks.classify import ClassifyTask
from docflow.ports.taxonomy import ClassificationFallback, ClassifierPolicy, TaxonomySnapshot
from docflow.settings import LlmTaskConfig


class FakeLlmClient:
    def __init__(self, reply: dict):
        self.reply = reply

    def complete_json(self, request, *, timeout, retry):
        return self.reply


def classify_tags(reply: dict) -> tuple[str, ...]:
    policy = ClassifierPolicy(
        family_members=("Sample", "Familie"),
        fallback=ClassificationFallback(
            correspondent="Unbekannt",
            document_type="Sonstiges",
            tag="Unsortiert",
            family_member="Familie",
        ),
        always_tags=("Personal",),
    )
    config = LlmTaskConfig(
        model="test", temperature=0.0, timeout_seconds=1.0, retry=RetryPolicy(attempts=1)
    )
    task = ClassifyTask(FakeLlmClient({"family_member": "Sample", **reply}), config, policy)
    return task.classify("text", TaxonomySnapshot()).tags


def test_topic_tags_are_kept_in_the_models_order():
    tags = classify_tags(
        {
            "correspondent": "Krankenkasse Musterstadt",
            "document_type": "Bescheid",
            "tags": ["Krankenversicherung", "2026"],
        }
    )
    assert tags == ("Krankenversicherung", "2026", "Personal", "Sample")


def test_a_tag_repeating_the_document_type_is_dropped():
    tags = classify_tags(
        {
            "correspondent": "Krankenkasse Musterstadt",
            "document_type": "Bescheid",
            "tags": ["bescheid", "Krankenversicherung"],
        }
    )
    assert tags == ("Krankenversicherung", "Personal", "Sample")


def test_a_tag_repeating_the_correspondent_is_dropped():
    tags = classify_tags(
        {
            "correspondent": "Krankenkasse Musterstadt",
            "document_type": "Bescheid",
            "tags": ["krankenkasse musterstadt", "Krankenversicherung"],
        }
    )
    assert tags == ("Krankenversicherung", "Personal", "Sample")


def test_a_tag_that_is_the_leading_words_of_the_correspondent_is_dropped():
    tags = classify_tags(
        {
            "correspondent": "Krankenkasse Musterstadt/Beispielhausen",
            "document_type": "Bescheid",
            "tags": ["Krankenkasse", "Krankenversicherung"],
        }
    )
    assert tags == ("Krankenversicherung", "Personal", "Sample")


def test_a_tag_from_the_middle_of_the_correspondent_is_kept():
    tags = classify_tags(
        {
            "correspondent": "Jobcenter (Kreis Musterstadt)",
            "document_type": "Bescheid",
            "tags": ["Kreis Musterstadt"],
        }
    )
    assert tags == ("Kreis Musterstadt", "Personal", "Sample")


def test_a_tag_that_only_shares_leading_letters_with_the_correspondent_is_kept():
    tags = classify_tags(
        {
            "correspondent": "Krankenkasse Musterstadt",
            "document_type": "Bescheid",
            "tags": ["Kranken"],
        }
    )
    assert tags == ("Kranken", "Personal", "Sample")


def test_the_fallback_tag_is_used_when_every_tag_was_redundant():
    tags = classify_tags(
        {
            "correspondent": "Krankenkasse Musterstadt",
            "document_type": "Bescheid",
            "tags": ["Krankenkasse", "Bescheid"],
        }
    )
    assert tags == ("Unsortiert", "Personal", "Sample")


def test_blank_tags_are_dropped():
    tags = classify_tags(
        {
            "correspondent": "Krankenkasse Musterstadt",
            "document_type": "Bescheid",
            "tags": ["  ", " Krankenversicherung "],
        }
    )
    assert tags == ("Krankenversicherung", "Personal", "Sample")
