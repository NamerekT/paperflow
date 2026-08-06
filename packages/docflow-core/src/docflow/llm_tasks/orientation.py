"""Working out which way up a scanned page is.

The scanner does not rotate wrongly-fed sheets. Its software can only deskew a
couple of degrees; detecting a 90, 180 or 270 degree rotation has been an open
request against it since 2023 and is still not implemented. Receipts are the
usual victim.

Asked as its own narrow call rather than folded into the transcription prompt:
giving a model two different jobs in one request makes it likelier to do both
of them worse. And asked *before* OCR rather than as a reaction to a low
confidence score, because trying all four rotations and comparing scores is
unreliable on exactly the pages that need it - a receipt has too little text
for the scores to separate. A model that simply looks at the image is far more
dependable than an inference drawn from confidence numbers.

Running first is not merely an extra cost, it is a saving: a page that was
upside down and has been corrected will usually be read correctly by the local
OCR, so the expensive transcription is never reached. One cheap narrow call
avoids one costly one.
"""

import logging

from docflow.llm_tasks.base import LlmTask
from docflow.llm_tasks.schemas import OrientationDraft
from docflow.ports.llm import LlmError

logger = logging.getLogger(__name__)

PROMPT_ORIENTATION = (
    "In this image of a document page the text may be turned the wrong way "
    "(the sheet was fed in wrongly when scanning - sideways or upside down). "
    "Work out by how many degrees CLOCKWISE the image has to be rotated for "
    "the text to read normally - left to right, top to bottom, like an "
    "ordinary page. If the text already reads normally, return 0. Answer "
    "STRICTLY as JSON, with no markdown and no explanation: "
    '{"rotate": 0} or {"rotate": 90} or {"rotate": 180} or {"rotate": 270}.'
)


class OrientationTask(LlmTask):
    """Detects how far a page must be turned clockwise."""

    def detect(self, image_b64: str) -> int:
        """Return 0, 90, 180 or 270.

        Returns 0 on any model failure. The value is directly compatible with
        a PDF page rotation - same convention, clockwise, as /Rotate itself.

        Degrading to 0 is deliberate and long-standing: this check must not be
        able to fail a document. At worst it misses a turned page, which is
        exactly where things stood before the check existed. What has changed
        is that the failure is now logged at warning with its cause, instead of
        vanishing into a debug-level callback nobody was reading, and that only
        model errors are swallowed - anything else is a defect here and
        propagates.
        """
        try:
            payload = self._ask_json([self.image_message(PROMPT_ORIENTATION, image_b64)])
        except LlmError:
            logger.warning("orientation detection failed, assuming 0 degrees", exc_info=True)
            return 0

        draft = OrientationDraft.model_validate(payload)
        angle = draft.normalised()
        if angle != draft.rotate:
            logger.warning(f"orientation model returned {draft.rotate!r}, treating as 0")
        return angle
