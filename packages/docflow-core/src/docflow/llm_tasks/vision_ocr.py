"""Transcribing a page the local OCR could not read confidently.

The expensive call, and the only one whose failure is allowed to abort a
document: if this page cannot be read, there is no text to file.
"""

import logging

from docflow.llm_tasks.base import LlmTask

logger = logging.getLogger(__name__)

# Three real incidents of the same family, found by streaming the generation
# token by token:
#
#   1. A page with a barcode along the bottom. The model latched onto it and
#      tried to read the stripe pattern letter by letter, emitting "||||||..."
#      until it hit the token ceiling.
#   2. A Schweigepflichtentbindung with an empty signature rule - "befreie
#      hiermit Frau / Herrn ____________" - where it tried to draw the rule out
#      in underscores across the full width of the page.
#   3. A diploma supplement with a long results table, mostly blank rows below
#      the last filled-in one. Told to mark each empty cell as "[empty field]"
#      rather than draw a rule, it did exactly that - once per cell, once per
#      row, for every one of the ~100 blank rows on the page - and never
#      reached MAX_TOKENS-free ground before the ceiling hit. The single-
#      character repetition rule below did not fire because no character ever
#      repeated more than a handful of times in a row; it was the whole row
#      that repeated.
#
# The common cause is that any long, visually repeating element with no textual
# content of its own - a barcode, a QR code, a blank rule for a signature, a
# dotted or solid table border, a table with many blank rows - has no natural
# length. However many characters (or rows) you emit, it never looks finished,
# and with nothing forbidding it the model will keep going to the token
# ceiling. max_tokens and frequency_penalty in the task settings bound the
# damage; the actual fix is the paragraph below, which names the whole
# category and says not to transcribe it character by character or row by
# row. That removes the cause rather than truncating the symptom, and it
# cannot cost accuracy on real text, because a model that never spends tokens
# on these elements never runs out of them either.
PROMPT_VISION = (
    "Transcribe ALL the text in this document image - printed and handwritten "
    "(the document may be in Ukrainian or German). Output plain text, "
    "preserving the order of the blocks on the page and the structure of "
    "tables (a tab or ' | ' is fine). "
    "Do not invent anything and do not skip anything - if something is "
    "illegible, mark it as [illegible], but do not leave gaps silently and do "
    "not substitute plausible but invented words for the real text. "
    "Barcodes, QR codes, watermarks, logos and any other purely graphical "
    "elements are NOT text: do not try to read them character by character and "
    "do not reproduce a pattern of lines or dots as a sequence of characters. "
    "Just mark them with one short note such as [barcode] or [QR code] and "
    "move straight on. The same goes for EMPTY fields to be filled in or "
    "signed (a long rule, line or row of dots with no text on it, for example "
    "'befreie hiermit Frau / Herrn ___________') - do not draw the rule out "
    "across its full visible length, just mark it with a short note such as "
    "[empty field] or [signature space] and move on. If a TABLE has a run of "
    "consecutive fully empty rows (every cell blank), do not emit one "
    "[empty field] row per blank row - transcribe at most 2 of them as a "
    "sample, then collapse the rest into one note such as '[38 more empty "
    "rows]' with your best count, and move on to the next non-empty content "
    "on the page. General rule: NEVER output the same character (including "
    "'_', '-', '.', '|') more than 3-5 times in a row, and NEVER output the "
    "same row or line of a table more than twice in a row, even if the table "
    "looks longer in the image - as soon as you notice you are about to "
    "repeat the same character, row or a short fragment of text, stop and "
    "move on to the next element on the page. Do not add comments or "
    "explanations of your own - only the transcribed text."
)


class VisionOcrTask(LlmTask):
    def transcribe(self, image_b64: str) -> tuple[str, str]:
        """Return the page text and a label saying who produced it.

        Raises on failure rather than degrading. Every other task here has a
        sensible answer to "the model did not reply"; this one does not, and a
        page silently replaced by an empty string would be filed as though it
        had been read.
        """
        response = self._ask([self.image_message(PROMPT_VISION, image_b64)])
        provider = response.provider or self._config.model
        return response.text, f"Gemma 31B (LiteLLM: {provider})"
