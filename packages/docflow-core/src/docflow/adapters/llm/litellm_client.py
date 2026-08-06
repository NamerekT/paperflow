"""The one implementation of `LlmClient`.

Talks to a self-hosted LiteLLM proxy that fronts every provider. That
indirection carries real weight: provider fallback - OpenRouter failing over
to Google AI Studio, and the chain of Gemini models below the classifier's
first choice - is configured in the proxy's router, not here. This code never
learns which vendor answered except by reading it back off the response.

What remains this side of the proxy is what a self-hosted service cannot do
for itself: retry on rate limits and server errors, and defend against the
proxy hanging exactly the way a direct provider connection used to.
"""

import concurrent.futures
import logging
import time
from typing import Any

import requests

from docflow.adapters.llm.deadline import DeadlinePoster
from docflow.adapters.llm.json_mode import parse_json_payload
from docflow.domain.policies import RetryPolicy
from docflow.ports.llm import (
    LlmEmptyResponse,
    LlmError,
    LlmExhausted,
    LlmRateLimited,
    LlmRequest,
    LlmRequestError,
    LlmResponse,
    LlmServerError,
    LlmTimeout,
    LlmUnavailable,
)

logger = logging.getLogger(__name__)


class LiteLlmClient:
    """`LlmClient` over an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        poster: DeadlinePoster | None = None,
        deadline_margin: float = 10.0,
        sleep: Any = time.sleep,
    ) -> None:
        # The four call sites this replaces each built this URL themselves from
        # the same two settings.
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._poster = poster or DeadlinePoster()
        self._deadline_margin = deadline_margin
        # Injected so the retry policy can be exercised without real waiting.
        self._sleep = sleep

    def complete(
        self, request: LlmRequest, *, timeout: float, retry: RetryPolicy
    ) -> LlmResponse:
        payload = request.to_payload()
        last_error: LlmError | None = None

        for attempt in range(1, retry.attempts + 1):
            try:
                response = self._post(payload, timeout=timeout)
            except LlmRequestError:
                # A bad request will be just as bad next time, and it means a
                # defect here rather than a bad day upstream.
                raise
            except LlmError as exc:
                last_error = exc
                if attempt >= retry.attempts:
                    break
                self._wait_before_retry(exc, attempt, retry, request.model)
                continue

            return self._read_response(response, request)

        assert last_error is not None
        raise LlmExhausted(
            f"{request.model}: no response after {retry.attempts} attempts ({last_error})",
            attempts=retry.attempts,
        ) from last_error

    def complete_json(
        self, request: LlmRequest, *, timeout: float, retry: RetryPolicy
    ) -> Any:
        return parse_json_payload(self.complete(request, timeout=timeout, retry=retry).text)

    # ── one attempt ───────────────────────────────────────────────────────

    def _post(self, payload: dict[str, Any], *, timeout: float) -> requests.Response:
        """Issue one request, translating transport failures into LlmErrors."""
        try:
            response = self._poster.post(
                self._endpoint,
                headers=self._headers,
                json=payload,
                timeout=timeout,
                deadline_margin=self._deadline_margin,
            )
        except requests.exceptions.ReadTimeout as exc:
            raise LlmTimeout(f"no reply within {timeout:.0f}s") from exc
        except concurrent.futures.TimeoutError as exc:
            # The client's own timeout did not fire: the connection is being
            # held open without data arriving. This is the failure DeadlinePoster
            # exists for, and it is worth naming separately in the log.
            raise LlmTimeout(
                f"connection held open with no reply for {timeout:.0f}s "
                f"(the client's own timeout never fired)"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LlmUnavailable(f"cannot reach the gateway: {exc}") from exc

        if response.status_code == 429:
            retry_after = self._retry_after(response)
            raise LlmRateLimited(
                f"rate limited: {response.text[:300]}", retry_after=retry_after
            )
        if response.status_code >= 500:
            raise LlmServerError(
                f"HTTP {response.status_code}: {response.text[:300]}", status=response.status_code
            )
        if response.status_code >= 400:
            raise LlmRequestError(
                f"HTTP {response.status_code}: {response.text[:500]}",
                status=response.status_code,
                body=response.text[:500],
            )
        return response

    @staticmethod
    def _retry_after(response: requests.Response) -> float | None:
        raw = response.headers.get("Retry-After")
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            # Retry-After may be an HTTP date. Not worth parsing: the caller
            # falls back to its own backoff, which is the right order of
            # magnitude anyway.
            return None

    def _read_response(self, response: requests.Response, request: LlmRequest) -> LlmResponse:
        data = response.json()
        choices = data.get("choices")
        if not choices:
            raise LlmEmptyResponse(f"reply carried no choices: {str(data)[:300]}")

        text = (choices[0].get("message") or {}).get("content")
        if not text or not text.strip():
            raise LlmEmptyResponse("reply carried an empty message")

        return LlmResponse(
            text=text.strip(),
            provider=data.get("provider"),
            model=data.get("model") or request.model,
        )

    # ── backoff ───────────────────────────────────────────────────────────

    def _wait_before_retry(
        self, error: LlmError, attempt: int, retry: RetryPolicy, model: str
    ) -> None:
        """Log the failure and sleep for however long this kind warrants."""
        if isinstance(error, LlmRateLimited):
            delay = error.retry_after or retry.rate_limit_seconds * attempt
        elif isinstance(error, LlmTimeout):
            # No sleep by default. The request already consumed its full
            # timeout, so the wait has happened; sleeping again on top of it
            # only doubles the time a slow provider costs.
            delay = retry.backoff_seconds * attempt if retry.sleep_after_timeout else 0.0
        else:
            delay = retry.backoff_seconds * attempt

        # Warning, not info: a retry is not routine traffic. At INFO these
        # lines are the only sign that the gateway is degrading.
        logger.warning(
            f"{model}: attempt {attempt}/{retry.attempts} failed ({error}); "
            f"retrying in {delay:.0f}s"
        )
        if delay > 0:
            self._sleep(delay)
