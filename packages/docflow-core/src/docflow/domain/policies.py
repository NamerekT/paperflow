"""Tunable policies: pure data that decides how the engine behaves.

These live in `domain` rather than in `settings` because the ports refer to
them - a `RetryPolicy` is part of the contract of `LlmClient.complete`, not a
detail of how the process happens to be configured. `settings` builds them
from the environment; a test constructs one inline.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts: int

    #: Linear, not exponential: the upstream is a self-hosted proxy fronting
    #: several providers, and a failure is far more often "this provider is
    #: briefly unhappy" than "we are overloading it".
    backoff_seconds: float = 5.0

    #: Used for HTTP 429 when the response carries no Retry-After header.
    #: Deliberately longer than `backoff_seconds` - a rate limit is a
    #: statement about a quota window, not a transient blip.
    rate_limit_seconds: float = 10.0

    #: Off by default, deliberately: a timed-out request already waited its
    #: full timeout (90-180s for vision calls), so sleeping again on top of it
    #: was measured to double a slow-provider episode for no gain.
    sleep_after_timeout: bool = False

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {self.attempts}")


@dataclass(frozen=True, slots=True)
class NativeTextConfig:
    """Thresholds for deciding a PDF is born-digital and needs no OCR.

    See `docflow.pipeline.native_text` for why each one exists; the defaults
    are the values the pipeline has been running on.
    """

    #: Below this many characters a text layer is noise - a stray header, a
    #: scanner watermark - not a real transcription.
    min_chars: int = 40

    #: A page more than half covered by embedded images is a scan, even if it
    #: carries a text layer. That combination is precisely what a "searchable
    #: PDF" looks like, and its text layer is somebody else's OCR output.
    max_image_coverage: float = 0.5

    #: Fraction of pages that must carry a usable text layer before the whole
    #: document is trusted as born-digital. Genuine exports have text on
    #: nearly every page; a scan with a text layer has it on a scattered few.
    min_page_ratio: float = 0.8


@dataclass(frozen=True, slots=True)
class EscalationPolicy:
    """Thresholds below which a locally-recognised page is escalated to the vision model.

    The 0.95 default is not arbitrary: at 0.80, a real Diplomsupplement grade
    table (mixed Cyrillic/Latin/digits) scored mean 0.938 / p10 0.834 -
    comfortably above threshold - while the text was badly wrong ("82/B/ope/Good"
    for "82/B/Good"): confident and incorrect at once, which local OCR cannot
    self-report. Raised to 0.85 then 0.90 still let poor pages through; below
    0.95 the local engine was not reliably right on anything.
    """

    mean: float = 0.95
    p10: float = 0.95

    #: Lines the engine scored below this are left out of the page text and of `mean`
    #: and `p10`. 0.5 is RapidOCR's own default `text_score`, which used to drop them
    #: inside the engine, where nothing downstream could see that a line was lost.
    line_score: float = 0.5

    #: A dropped line with at least this many letters or digits was text the engine
    #: failed to read. Below that it is what a fold, a stain or a rule line leaves.
    lost_line_chars: int = 5

    #: Escalate a page once this many dropped lines carried text, whatever its mean and
    #: p10 are: those only describe the lines that were kept. A page losing one line of a
    #: paragraph used to file at 99.8% confidence (PPF-16).
    lost_lines: int = 1

    def __post_init__(self) -> None:
        if self.lost_lines < 1:
            raise ValueError(f"lost_lines must be at least 1, got {self.lost_lines}")

    def is_lost_text(self, text: str) -> bool:
        return sum(ch.isalnum() for ch in text) >= self.lost_line_chars

    def should_escalate(self, mean: float, p10: float, lost_lines: int = 0) -> bool:
        if mean != mean or p10 != p10:  # NaN: no lines were recognised at all
            return True
        return mean < self.mean or p10 < self.p10 or lost_lines >= self.lost_lines
