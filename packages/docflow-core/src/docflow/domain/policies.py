"""Tunable policies: pure data that decides how the engine behaves.

These live in `domain` rather than in `settings` because the ports refer to
them - a `RetryPolicy` is part of the contract of `LlmClient.complete`, not a
detail of how the process happens to be configured. `settings` builds them
from the environment; a test constructs one inline.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How hard to try before giving up on a model call."""

    attempts: int

    #: Linear, not exponential: the upstream is a self-hosted proxy fronting
    #: several providers, and a failure is far more often "this provider is
    #: briefly unhappy" than "we are overloading it".
    backoff_seconds: float = 5.0

    #: Used for HTTP 429 when the response carries no Retry-After header.
    #: Deliberately longer than `backoff_seconds` - a rate limit is a
    #: statement about a quota window, not a transient blip.
    rate_limit_seconds: float = 10.0

    #: Whether to sleep after a timeout.
    #:
    #: Off by default, and that is not an oversight. A request that timed out
    #: has already consumed its full timeout - 90 to 180 seconds for the
    #: vision calls - so the wait has happened. Sleeping again on top of it
    #: was measured to make a slow-provider episode take twice as long for no
    #: gain. The four hand-written retry loops this policy replaces all
    #: `continue`d straight through on timeout, and that behaviour is
    #: preserved deliberately.
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
    """When a locally recognised page is not good enough to keep.

    History of the threshold, because the number looks arbitrary and is not:

    It started at 0.80. A real document (a Diplomsupplement grade table, mixed
    Cyrillic, Latin and digits in short cells) scored mean 0.938 and p10 0.834
    - comfortably above that threshold - while the text itself was badly
    wrong: "82/B/ope/Good" for "82/B/Good", "poepayeann a Pyihon" for
    "Programming Python". Confident and incorrect at the same time, which is
    the failure mode local OCR cannot self-report.

    Raised to 0.85, then to 0.90 after a full run showed that at 0.85 nearly
    everything escalated anyway, and the handful of pages that did not were
    themselves poor. Below 0.95 the local engine was not reliably right on
    anything, so there was no value in a lower bar.
    """

    mean: float = 0.95
    p10: float = 0.95

    def should_escalate(self, mean: float, p10: float) -> bool:
        """True when a page must go to the vision model.

        A page with no recognised lines gives NaN for both statistics; every
        comparison against NaN is False, so the explicit check below is what
        makes an empty page escalate rather than silently pass.
        """
        if mean != mean or p10 != p10:  # NaN: no lines were recognised at all
            return True
        return mean < self.mean or p10 < self.p10
