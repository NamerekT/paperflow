"""Retry behaviour and the error taxonomy of the model gateway client.

This is the piece the migration is most likely to get subtly wrong: four
hand-written retry loops, each with slightly different rules, collapsed into
one. What each caller does when the model finally gives up is a decision that
was made deliberately in each of the four, and these tests pin those decisions
down.
"""

import concurrent.futures

import pytest
import requests

from docflow.adapters.llm.litellm_client import LiteLlmClient
from docflow.domain.policies import RetryPolicy
from docflow.ports.llm import (
    LlmEmptyResponse,
    LlmExhausted,
    LlmInvalidJson,
    LlmRateLimited,
    LlmRequest,
    LlmRequestError,
    LlmTimeout,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._payload


def reply(text="hello", provider="Google AI Studio"):
    return FakeResponse(
        payload={
            "choices": [{"message": {"content": text}}],
            "provider": provider,
            "model": "test-model",
        }
    )


class FakePoster:
    """Returns or raises whatever the test queued, in order."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def post(self, url, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else reply()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class RecordingSleep:
    def __init__(self):
        self.slept = []

    def __call__(self, seconds):
        self.slept.append(seconds)


def make_client(poster, sleep=None):
    return LiteLlmClient(
        base_url="http://gateway.test/",
        api_key="key",
        poster=poster,
        sleep=sleep or RecordingSleep(),
    )


def request():
    return LlmRequest(model="test-model", messages=[{"role": "user", "content": "hi"}])


def test_endpoint_is_built_once_from_the_base_url():
    poster = FakePoster()
    captured = {}

    def capture(url, **kwargs):
        captured["url"] = url
        return reply()

    poster.post = capture
    make_client(poster).complete(request(), timeout=1, retry=RetryPolicy(attempts=1))
    assert captured["url"] == "http://gateway.test/chat/completions"


def test_successful_reply_carries_the_provider_that_answered():
    # The proxy falls back between providers on its own; this is the only way
    # to find out which one produced the text.
    response = make_client(FakePoster(reply("text", provider="Google AI Studio"))).complete(
        request(), timeout=1, retry=RetryPolicy(attempts=1)
    )
    assert response.text == "text"
    assert response.provider == "Google AI Studio"


def test_a_timeout_is_retried_and_then_succeeds():
    poster = FakePoster(requests.exceptions.ReadTimeout(), reply("second time lucky"))
    response = make_client(poster).complete(request(), timeout=1, retry=RetryPolicy(attempts=4))
    assert response.text == "second time lucky"
    assert poster.calls == 2


def test_a_silent_hang_is_a_timeout_too():
    # The wall-clock deadline firing where the HTTP client's own timeout never
    # did. Both have to be caught; catching only one is what the deadline
    # wrapper exists to cover.
    poster = FakePoster(concurrent.futures.TimeoutError(), reply())
    make_client(poster).complete(request(), timeout=1, retry=RetryPolicy(attempts=2))
    assert poster.calls == 2


def test_no_sleep_after_a_timeout_by_default():
    # The request already consumed its full timeout, so the wait has happened.
    # Sleeping again on top of it doubles what a slow provider costs.
    sleep = RecordingSleep()
    poster = FakePoster(requests.exceptions.ReadTimeout(), reply())
    make_client(poster, sleep).complete(request(), timeout=1, retry=RetryPolicy(attempts=2))
    assert sleep.slept == []


def test_sleep_after_timeout_can_be_switched_on():
    sleep = RecordingSleep()
    poster = FakePoster(requests.exceptions.ReadTimeout(), reply())
    make_client(poster, sleep).complete(
        request(), timeout=1, retry=RetryPolicy(attempts=2, sleep_after_timeout=True)
    )
    assert sleep.slept == [5.0]


def test_rate_limit_honours_retry_after():
    sleep = RecordingSleep()
    poster = FakePoster(FakeResponse(429, headers={"Retry-After": "42"}, text="slow down"), reply())
    make_client(poster, sleep).complete(request(), timeout=1, retry=RetryPolicy(attempts=2))
    assert sleep.slept == [42.0]


def test_rate_limit_without_a_header_backs_off_per_attempt():
    sleep = RecordingSleep()
    poster = FakePoster(FakeResponse(429), FakeResponse(429), reply())
    make_client(poster, sleep).complete(request(), timeout=1, retry=RetryPolicy(attempts=3))
    assert sleep.slept == [10.0, 20.0]


def test_server_errors_are_retried():
    poster = FakePoster(FakeResponse(503, text="upstream down"), reply())
    make_client(poster).complete(request(), timeout=1, retry=RetryPolicy(attempts=2))
    assert poster.calls == 2


def test_a_bad_request_is_not_retried():
    # A 4xx that is not a rate limit means this code sent something wrong.
    # Retrying burns quota; falling back would hide the defect.
    poster = FakePoster(FakeResponse(400, text="malformed"))
    with pytest.raises(LlmRequestError) as caught:
        make_client(poster).complete(request(), timeout=1, retry=RetryPolicy(attempts=4))
    assert caught.value.status == 400
    assert poster.calls == 1


def test_exhaustion_names_the_last_cause():
    # The message this replaces said only "no response in 4 attempts", which
    # told the reader nothing they could act on.
    poster = FakePoster(*[FakeResponse(503, text="down")] * 3)
    with pytest.raises(LlmExhausted) as caught:
        make_client(poster).complete(request(), timeout=1, retry=RetryPolicy(attempts=3))
    assert caught.value.attempts == 3
    assert isinstance(caught.value.__cause__, Exception)
    assert "503" in str(caught.value.__cause__)


def test_an_empty_message_is_not_silently_accepted():
    poster = FakePoster(FakeResponse(payload={"choices": [{"message": {"content": "   "}}]}))
    with pytest.raises(LlmEmptyResponse):
        make_client(poster).complete(request(), timeout=1, retry=RetryPolicy(attempts=1))


def test_a_reply_without_choices_is_an_empty_response():
    poster = FakePoster(FakeResponse(payload={"error": "something"}))
    with pytest.raises(LlmEmptyResponse):
        make_client(poster).complete(request(), timeout=1, retry=RetryPolicy(attempts=1))


@pytest.mark.parametrize(
    "wrapped",
    [
        '{"rotate": 90}',
        '```json\n{"rotate": 90}\n```',
        '```\n{"rotate": 90}\n```',
        '  ```json {"rotate": 90} ```  ',
    ],
)
def test_json_replies_survive_a_code_fence(wrapped):
    # Models asked for "strictly JSON, no markdown" comply most of the time.
    poster = FakePoster(reply(wrapped))
    parsed = make_client(poster).complete_json(request(), timeout=1, retry=RetryPolicy(attempts=1))
    assert parsed == {"rotate": 90}


def test_a_non_json_reply_is_reported_with_what_came_back():
    poster = FakePoster(reply("I'm afraid I can't do that"))
    with pytest.raises(LlmInvalidJson) as caught:
        make_client(poster).complete_json(request(), timeout=1, retry=RetryPolicy(attempts=1))
    assert "afraid" in caught.value.raw


def test_retry_policy_rejects_a_zero_attempt_budget():
    with pytest.raises(ValueError):
        RetryPolicy(attempts=0)


def test_rate_limited_error_keeps_the_retry_after_value():
    error = LlmRateLimited("slow down", retry_after=7.5)
    assert error.retry_after == 7.5


def test_timeout_is_distinct_from_a_transport_failure():
    poster = FakePoster(requests.exceptions.ReadTimeout())
    with pytest.raises(LlmExhausted) as caught:
        make_client(poster).complete(request(), timeout=1, retry=RetryPolicy(attempts=1))
    assert isinstance(caught.value.__cause__, LlmTimeout)
