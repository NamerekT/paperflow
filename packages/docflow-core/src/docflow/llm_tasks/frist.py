"""Finding the deadline in an official letter.

A Frist is a legal deadline - to object to a Bescheid, to pay, to supply
documents - and missing one has consequences that are not recoverable by
apologising. This runs as its own step after the document has been filed,
reading the text back from the archive rather than the original file.

Deliberately separate from classification: a different model on a different
quota, a different prompt, one job. Sharing a call with the classifier would
mean a day of heavy classification could exhaust the budget for the one thing
here that has a legal clock attached to it.
"""

import logging
from datetime import date

from docflow.domain.text import truncate_for_llm
from docflow.llm_tasks.base import LlmTask
from docflow.llm_tasks.schemas import FristDraft
from docflow.ports.llm import LlmClient, LlmError
from docflow.settings import LlmTaskConfig

logger = logging.getLogger(__name__)

# The quoted phrases are search cues, not prose: they are what actually appears
# on the page. The German ones stay German, and the Ukrainian and Russian ones
# stay as they are too - the archive is trilingual, and a letter from a Ukrainian
# institution states its deadline in Ukrainian.
PROMPT_SYSTEM = """You help a household in Germany not to miss deadlines \
(Fristen) in official letters (Jobcenter, Familienkasse, Agentur für Arbeit, a \
bank, a landlord and so on).

You are given the text of a letter and the date OF THE LETTER ITSELF (when it \
was written or sent - do NOT confuse that date with the deadline you are \
looking for).

Find the deadline that is stated explicitly in the text - the date (or period) \
by which the recipient has to do something: reply, file an objection \
(Widerspruch), pay, supply documents and so on. Look for wordings such as \
"Frist", "Widerspruchsfrist", "bis zum", "spätestens", "innerhalb von N \
Wochen/Tagen nach Erhalt/Zugang dieses Schreibens", "срок до", \
"протягом N днів".

RULES:
- If the deadline is given as a RELATIVE period (for example "in 4 Wochen nach \
Zugang dieses Schreibens"), work out the absolute date FROM THE DATE OF THE \
LETTER - not from today, since you do not know when this is being read.
- Do NOT confuse the deadline with the date of the letter itself, with a date \
of birth, with "gültig bis" on an identity document, or with the date of an \
event that has already passed. Look specifically for a deadline on a FUTURE \
action by the recipient.
- If no deadline is stated EXPLICITLY in the text, do not invent or guess one \
- return null.
- If there are several deadlines, return the NEAREST one (the first that \
actually has to be met).

Answer STRICTLY as JSON, with no markdown and no explanation:
{"frist": "YYYY-MM-DD"} or {"frist": null}"""


class FristTask(LlmTask):
    def __init__(
        self,
        client: LlmClient,
        config: LlmTaskConfig,
        *,
        truncation_chars: int = 10_000,
    ) -> None:
        super().__init__(client, config)
        self._truncation_chars = truncation_chars

    def extract(self, content_text: str, document_date: date | str) -> date | None:
        """Return the deadline, or None if there is not one.

        None covers two different things on purpose, because the caller does
        the same thing with both: no deadline was stated - the common case, and
        not a failure - or the model could not be reached. Only the second is
        logged as a problem.
        """
        if not content_text or not content_text.strip():
            return None

        messages = [
            {"role": "system", "content": PROMPT_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"LETTER DATE: {document_date}\n\n"
                    f"LETTER TEXT:\n{truncate_for_llm(content_text, self._truncation_chars)}"
                ),
            },
        ]

        try:
            payload = self._ask_json(messages)
        except LlmError:
            logger.warning("deadline extraction failed, continuing without one", exc_info=True)
            return None

        draft = FristDraft.model_validate(payload)
        if draft.frist is None:
            logger.debug("no deadline stated in this document")
        return draft.frist
