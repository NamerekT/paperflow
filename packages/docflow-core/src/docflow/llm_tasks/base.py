from typing import Any

from docflow.ports.llm import LlmClient, LlmRequest
from docflow.settings import LlmTaskConfig


class LlmTask:
    def __init__(self, client: LlmClient, config: LlmTaskConfig) -> None:
        self._client = client
        self._config = config

    def _request(self, messages: list[dict[str, Any]]) -> LlmRequest:
        return LlmRequest(
            model=self._config.model,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            frequency_penalty=self._config.frequency_penalty,
            extra_body=dict(self._config.extra_body),
        )

    def _ask(self, messages: list[dict[str, Any]]) -> Any:
        return self._client.complete(
            self._request(messages),
            timeout=self._config.timeout_seconds,
            retry=self._config.retry,
        )

    def _ask_json(self, messages: list[dict[str, Any]]) -> Any:
        return self._client.complete_json(
            self._request(messages),
            timeout=self._config.timeout_seconds,
            retry=self._config.retry,
        )

    @staticmethod
    def image_message(prompt: str, image_b64: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }
