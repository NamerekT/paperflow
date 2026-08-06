"""Getting JSON back out of a chat completion.

Models asked for "strictly JSON, no Markdown" comply most of the time and wrap
the answer in a fenced code block the rest of it. The one line that undoes
that used to exist verbatim in three files.
"""

import json
from typing import Any

from docflow.ports.llm import LlmInvalidJson

_FENCES = ("```json", "```JSON", "```")


def strip_json_fences(text: str) -> str:
    stripped = text.strip()
    for fence in _FENCES:
        if stripped.startswith(fence):
            stripped = stripped[len(fence) :]
            break
    if stripped.endswith("```"):
        stripped = stripped[: -len("```")]
    return stripped.strip()


def parse_json_payload(text: str) -> Any:
    raw = strip_json_fences(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # The raw text is truncated into the message on purpose: a runaway
        # generation can be thousands of characters, and the first few hundred
        # are enough to see what the model actually did.
        raise LlmInvalidJson(f"model reply is not valid JSON ({exc}): {raw[:300]!r}", raw) from exc
