"""Deciding what a document is.

Produces a title, a sender, a kind and a set of topic tags - all as names.
Turning those names into archive identifiers is the binding's job; see
`docflow.ports.taxonomy` for why the split falls here.

The prompt lives in this module and only here. It used to be duplicated in two
scripts, and every refinement to it - do not tag the correspondent's own name,
prefer an existing document type - had to be made twice, which is how the two
copies came to disagree.
"""

import logging
from datetime import date, datetime

from docflow.domain.models import DocumentClassification
from docflow.domain.text import truncate_for_llm
from docflow.llm_tasks.base import LlmTask
from docflow.llm_tasks.schemas import ClassificationDraft
from docflow.ports.llm import LlmClient, LlmError, LlmRequestError
from docflow.ports.taxonomy import ClassifierPolicy, TaxonomySnapshot
from docflow.settings import LlmTaskConfig

logger = logging.getLogger(__name__)


def build_system_prompt(
    snapshot: TaxonomySnapshot, family_members: tuple[str, ...], today: date
) -> str:
    return f"""You are a document classifier for a Ukrainian migrant family in Germany.
Documents are archived in a document management system instead of folders -
your job is to assign metadata, not to pick a folder.

EXISTING TAGS (reuse a good match instead of creating a near-duplicate):
{", ".join(snapshot.tags) or "(none yet)"}

EXISTING CORRESPONDENTS (senders):
{", ".join(snapshot.correspondents) or "(none yet)"}

EXISTING DOCUMENT TYPES:
{", ".join(snapshot.document_types) or "(none yet)"}

FAMILY MEMBERS (pick exactly one - see family_member below):
{", ".join(family_members)}

RESPONSE FORMAT (JSON only, no Markdown, no explanations):
{{
  "title": "Short descriptive title, e.g. Bedarfsberechnung Januar 2026",
  "correspondent": "Sender name - reuse an existing one if it matches",
  "document_type": "Document type - reuse an existing one if it matches",
  "tags": ["1 to 4 tags - reuse existing tags where sensible"],
  "family_member": "Exactly one name from FAMILY MEMBERS above",
  "date": "YYYY-MM-DD, the date on the document, or {today:%Y-%m-%d} if unknown"
}}

RULES:
- correspondent: the organization/person who SENT the document (e.g.
  "Jobcenter Wesel", "Agentur für Arbeit (Inkasso-Service)", "Familienkasse").
  Be precise - German authorities that sound similar are often legally
  different institutions (Jobcenter != Agentur für Arbeit != Familienkasse !=
  Sozialamt). Never guess a "parent" institution name if the letterhead
  names a specific one.
- document_type: the KIND of document (e.g. "Bescheid", "Rückzahlung",
  "Rechnung", "Vertrag", "Ausweis", "Zeugnis"). ALWAYS prefer an EXISTING
  document_type that specifically matches over inventing a new one AND over
  falling back to a more generic existing type. Example: if the letter is
  specifically demanding repayment of an overpayment and "Rückzahlung"
  already exists in EXISTING DOCUMENT TYPES, use "Rückzahlung" - do not use
  a more generic type like "Bescheid" or "Schreiben" just because it's also
  technically true; pick the MOST SPECIFIC existing type that fits.
- tags: describe the SUBJECT/TOPIC of the document (e.g. "Bürgergeld",
  "Miete", "Schule", "2026") - NOT who sent it (that's `correspondent`) and
  NOT the kind of document (that's `document_type`). Do NOT add a tag that
  just repeats or aliases the correspondent's organization name (e.g. don't
  tag "Jobcenter" when correspondent is "Agentur für Arbeit
  (Inkasso-Service)" - besides being redundant, those are different
  institutions, don't conflate them).
- family_member: who the document is addressed to, or who it's about -
  e.g. whose Bürgergeld application, whose diploma, whose passport. This is
  about the FAMILY MEMBER, not the sender. If the document is a shared/
  household matter (rent, car registration) or you genuinely can't tell,
  use the shared-household name from FAMILY MEMBERS.
- Prefer reusing an existing correspondent/document_type/tag over inventing
  a near-duplicate (e.g. if "Jobcenter Wesel" exists, don't create
  "Jobcenter-Wesel" or "jobcenter wesel").
- title: no file extension, no forbidden characters, max ~80 characters."""


class ClassifyTask(LlmTask):
    """Classifies document text into names."""

    def __init__(
        self,
        client: LlmClient,
        config: LlmTaskConfig,
        policy: ClassifierPolicy,
        *,
        truncation_chars: int = 10_000,
    ) -> None:
        super().__init__(client, config)
        self._policy = policy
        self._truncation_chars = truncation_chars

    def classify(self, content_text: str, snapshot: TaxonomySnapshot) -> DocumentClassification:
        """Classify `content_text`, falling back rather than failing.

        A document that cannot be classified is still worth filing: it lands
        under the fallback names, where it is visible and can be corrected by
        hand. Losing it entirely because a model was rate-limited would be
        worse.

        The exception is `LlmRequestError` - a 4xx that is not a rate limit
        means this code sent something the gateway would not accept, which is a
        defect here. Substituting a fallback for it would file a stream of
        unclassified documents while hiding the reason.
        """
        today = datetime.now().date()
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(snapshot, self._policy.family_members, today),
            },
            {
                "role": "user",
                "content": (
                    "DOCUMENT TEXT:\n"
                    f"{truncate_for_llm(content_text, self._truncation_chars)}"
                ),
            },
        ]

        try:
            draft = ClassificationDraft.model_validate(self._ask_json(messages))
        except LlmRequestError:
            raise
        except LlmError as exc:
            logger.warning(f"classification failed ({exc}) - filing under the fallback names")
            draft = ClassificationDraft()

        return self._resolve(draft, today)

    def _resolve(self, draft: ClassificationDraft, today: date) -> DocumentClassification:
        fallback = self._policy.fallback
        family_member = self._match_family_member(draft.family_member or fallback.family_member)

        # "Personal" and the family-member tag are added here rather than asked
        # for in the prompt. A model cannot be relied on for a deterministic
        # guarantee - the same reason it sometimes confused two institutions or
        # dropped an existing document type. dict.fromkeys keeps the model's
        # own ordering while removing duplicates.
        tags = tuple(
            dict.fromkeys(
                [
                    *(draft.tags or [fallback.tag]),
                    *self._policy.always_tags,
                    family_member,
                ]
            )
        )

        return DocumentClassification(
            title=draft.title or f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{fallback.correspondent}",
            correspondent=draft.correspondent or fallback.correspondent,
            document_type=draft.document_type or fallback.document_type,
            tags=tags,
            family_member=family_member,
            date=draft.date or today,
        )

    def _match_family_member(self, raw_value: str) -> str:
        """Map the model's answer onto a canonical name from the list.

        The model does not always echo the list verbatim: it may add or drop a
        surname, change the case, or reorder the words - particularly since the
        prompt also shows it existing tags containing full names. So: exact
        match ignoring case first, then match on the first name, and only then
        fall back to the shared-household name, with a log line - the point is
        that a near-miss is visible rather than quietly discarding a correct
        answer.
        """
        normalized = raw_value.strip()

        for member in self._policy.family_members:
            if member.casefold() == normalized.casefold():
                return member

        words = normalized.split()
        first_word = words[0].casefold() if words else ""
        if first_word:
            for member in self._policy.family_members:
                if member.split()[0].casefold() == first_word:
                    if member != normalized:
                        logger.info(
                            f"model said {raw_value!r}, matched to {member!r} by first name"
                        )
                    return member

        logger.warning(
            f"model chose an unknown family member {raw_value!r} - "
            f"using {self._policy.fallback.family_member!r}"
        )
        return self._policy.fallback.family_member
