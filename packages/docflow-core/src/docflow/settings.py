"""Engine configuration, read from the environment.

Plain frozen dataclasses with an explicit `from_env`, not pydantic-settings:
this module has to be importable with nothing installed beyond the standard
library, because the applications that only need the shared `EnvReader` must
not be made to pull in a validation framework to get it.

Secrets have no defaults. A missing one is a startup error naming the
variable, not a 401 discovered halfway through a document.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass, field

from docflow.domain.policies import EscalationPolicy, NativeTextConfig, RetryPolicy
from docflow.errors import ConfigError


class EnvReader:
    """Reads environment variables, collecting failures instead of raising.

    The point is the `raise_if_incomplete` at the end. Configuring this
    system takes half a dozen variables; discovering them one restart at a
    time is miserable, so every miss is recorded and reported together.
    """

    def __init__(self, *, hint: str = "") -> None:
        self._missing: list[str] = []
        self._invalid: list[str] = []
        self._hint = hint

    def optional(self, name: str, default: str) -> str:
        return os.environ.get(name, default)

    def required(self, name: str) -> str:
        """A value with no default. Returns "" and records the miss."""
        value = os.environ.get(name, "").strip()
        if not value:
            self._missing.append(name)
        return value

    def flag(self, name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() not in ("false", "0", "no", "off", "")

    def integer(self, name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return default
        try:
            return int(raw)
        except ValueError:
            self._invalid.append(f"{name}={raw!r} (expected an integer)")
            return default

    def number(self, name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            return default
        try:
            return float(raw)
        except ValueError:
            self._invalid.append(f"{name}={raw!r} (expected a number)")
            return default

    def raise_if_incomplete(self) -> None:
        if not self._missing and not self._invalid:
            return
        parts: list[str] = []
        if self._missing:
            parts.append("missing required variables: " + ", ".join(sorted(self._missing)))
        if self._invalid:
            parts.append("invalid values: " + "; ".join(sorted(self._invalid)))
        message = "Configuration is incomplete - " + "; ".join(parts)
        if self._hint:
            message += f". {self._hint}"
        raise ConfigError(message)


@dataclass(frozen=True, slots=True)
class LlmTaskConfig:
    """Per-task model settings.

    One of these per job rather than one global model, because the jobs have
    genuinely different economics: page transcription is the expensive one and
    runs on a large vision model, while deadline extraction and classification
    are single cheap calls per document. Giving them separate models also gives
    them separate quota, so a day of heavy OCR cannot starve classification.
    """

    model: str
    temperature: float
    timeout_seconds: float
    retry: RetryPolicy
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    #: Provider-specific fields passed through untouched to the proxy.
    extra_body: Sequence[tuple[str, object]] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class LlmSettings:
    """How to reach the model gateway, and what to ask of it per task.

    A single self-hosted LiteLLM proxy fronts every provider. That indirection
    is load-bearing: when a provider rate-limits or hangs, the proxy's own
    fallback chain moves to the next one, so this code does not have to know
    which vendor actually answered.
    """

    base_url: str
    api_key: str
    vision: LlmTaskConfig
    frist: LlmTaskConfig
    classifier: LlmTaskConfig

    #: How much document text to send. Applied identically to the classifier
    #: and the deadline extractor - see `docflow.domain.text.truncate_for_llm`.
    text_truncation_chars: int = 10_000

    #: Slack added to the wall-clock deadline enforced around each request, on
    #: top of the task's own timeout. The gateway has been observed to hold a
    #: connection open indefinitely without ever timing out on its own, so the
    #: deadline is enforced from this side.
    deadline_margin_seconds: float = 10.0

    @classmethod
    def from_env(cls, env: EnvReader) -> "LlmSettings":
        base_url = env.optional("VISION_PROXY_URL", "").strip() or env.required("LLM_BASE_URL")
        api_key = env.optional("VISION_PROXY_KEY", "").strip() or env.required("LLM_API_KEY")

        # dev bypasses the proxy's fallback chain and goes straight to the
        # cheap model. Debugging runs push the same document through the
        # classifier over and over; without this they exhaust the daily quota
        # of the good models on work that will be thrown away.
        classifier_model = (
            "google-gemini-3.5-flash-lite"
            if env.optional("CLASSIFIER_ENV", "prod").strip().lower() == "dev"
            else env.optional("CLASSIFIER_LLM_MODEL", "google-gemini-3.6-flash")
        )

        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            vision=LlmTaskConfig(
                model=env.optional("VISION_PROXY_MODEL", "openrouter-gemma-4-31b-it"),
                temperature=0.0,
                timeout_seconds=env.number("LLM_VISION_TIMEOUT", 90.0),
                retry=RetryPolicy(attempts=4),
                # Both of these are belt-and-braces against a model looping on
                # a barcode or a signature rule. The real fix is in the prompt,
                # which names those elements and forbids transcribing them
                # character by character; these only bound the damage if a
                # variant shows up that the prompt does not cover.
                max_tokens=8192,
                frequency_penalty=0.3,
            ),
            frist=LlmTaskConfig(
                model=env.optional("FRIST_LLM_MODEL", "google-gemini-3.5-flash-lite"),
                temperature=0.0,
                timeout_seconds=env.number("LLM_FRIST_TIMEOUT", 90.0),
                retry=RetryPolicy(attempts=4),
            ),
            classifier=LlmTaskConfig(
                model=classifier_model,
                temperature=0.5,
                timeout_seconds=env.number("LLM_CLASSIFIER_TIMEOUT", 120.0),
                retry=RetryPolicy(attempts=4),
            ),
            text_truncation_chars=env.integer("LLM_TEXT_TRUNCATION_CHARS", 10_000),
        )


@dataclass(frozen=True, slots=True)
class CascadeSettings:
    #: One value for the whole engine. It used to be 300 at the call site and
    #: 200 in the OCR engine's own default, which meant the number that
    #: actually applied depended on which constructor argument was passed.
    render_dpi: int = 300
    jpeg_quality: int = 85
    escalation: EscalationPolicy = field(default_factory=EscalationPolicy)
    native_text: NativeTextConfig = field(default_factory=NativeTextConfig)

    @classmethod
    def from_env(cls, env: EnvReader) -> "CascadeSettings":
        return cls(
            render_dpi=env.integer("PAPERFLOW_RENDER_DPI", 300),
            jpeg_quality=env.integer("PAPERFLOW_JPEG_QUALITY", 85),
            escalation=EscalationPolicy(
                mean=env.number("PAPERFLOW_ESCALATE_MEAN", 0.95),
                p10=env.number("PAPERFLOW_ESCALATE_P10", 0.95),
            ),
        )
