"""The model gateway port and its error taxonomy.

This replaces four hand-written copies of the same retry loop - one each for
page transcription, classification and deadline extraction. They
had drifted: different retry counts, different sets of caught exceptions, and
three separate copies of the same line that strips Markdown code fences off a
JSON reply. The differences were not decisions; they were the result of the
fourth copy being written from memory of the third.

What is a decision, and is preserved, is what each caller does when the model
finally cannot answer. That belongs to the caller, so it stays there: this
module's job is to classify the failure precisely enough that the caller can
act on it, which is why there are eight error types rather than one.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from docflow.domain.policies import RetryPolicy
from docflow.errors import DocflowError


class LlmError(DocflowError):
    pass


class LlmTimeout(LlmError):
    """No answer within the deadline.

    Covers both the HTTP client's own read timeout and the wall-clock deadline
    enforced above it. Both are needed: the gateway has been observed to hold a
    connection open indefinitely, sending just enough to keep the socket from
    ever going quiet long enough for the client to give up on its own.
    """


class LlmUnavailable(LlmError):
    pass


class LlmRateLimited(LlmError):
    """HTTP 429."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LlmServerError(LlmError):
    """HTTP 5xx - the gateway or a provider behind it is unwell."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class LlmRequestError(LlmError):
    """HTTP 4xx other than 429.

    Not retryable, and deliberately distinguished from the rest: a malformed
    request is a bug in this code. Retrying it wastes the quota, and papering
    over it with a fallback value hides the defect behind a document that looks
    merely badly classified.
    """

    def __init__(self, message: str, status: int, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class LlmEmptyResponse(LlmError):
    """A well-formed reply with no content in it."""


class LlmInvalidJson(LlmError):
    """The reply was not JSON, after fences were stripped."""

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class LlmExhausted(LlmError):
    """Every attempt failed.

    Always raised with `from` the last underlying error, so the log line says
    what actually went wrong. The message this replaces read "could not get a
    response in 4 attempts", which told the reader nothing they could act on.
    """

    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class LlmRequest:
    """One chat completion request.

    `messages` is left as loose mappings on purpose: the vision calls send
    multimodal content parts, and modelling every provider's message shape
    would buy nothing here - the payload is passed through to the gateway
    unexamined.
    """

    model: str
    messages: Sequence[Mapping[str, Any]]
    temperature: float = 0.0
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(self.messages),
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.frequency_penalty is not None:
            payload["frequency_penalty"] = self.frequency_penalty
        payload.update(self.extra_body)
        return payload


@dataclass(frozen=True, slots=True)
class LlmResponse:
    text: str
    #: Which provider actually answered, as reported by the gateway. The proxy
    #: falls back between providers on its own, so this is the only way to know
    #: whether the primary or the standby produced the result - and it ends up
    #: recorded against the document.
    provider: str | None
    model: str


class LlmClient(Protocol):
    def complete(
        self, request: LlmRequest, *, timeout: float, retry: RetryPolicy
    ) -> LlmResponse:
        """Send `request`, retrying per `retry`.

        Raises the specific `LlmError` subclass for a non-retryable failure, or
        `LlmExhausted` when the retries ran out.
        """
        ...

    def complete_json(
        self, request: LlmRequest, *, timeout: float, retry: RetryPolicy
    ) -> Any:
        """`complete`, then strip code fences and parse the reply as JSON.

        Raises `LlmInvalidJson` if what came back is not JSON.
        """
        ...
