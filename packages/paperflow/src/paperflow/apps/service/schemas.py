"""The webhook request body, and the double-JSON-encoding Paperless sends it with.

Real case, found 3 August 2026: with a Webhook action configured with a Body template
(not Params) and "As JSON" enabled, Paperless takes the already-rendered Jinja string
(a valid piece of JSON text like '{"document_id": 307}') and passes it as the value of
httpx's own `json=...` parameter - httpx then serialises that string a SECOND time (see
the paperless-ngx source, src/documents/workflows/webhooks.py: `if as_json:
post_args["json"] = data`, where `data` is already a finished string, not a dict). The
request body ends up as a JSON string CONTAINING JSON, not the object itself - a plain
`json.loads` returns a `str`, not a `dict`. Rather than depending on the exact
combination of "As JSON"/"Params" checkboxes in a given Paperless version, the double
wrapping is unwrapped here instead: if the first `json.loads` produces a string rather
than a dict, it is parsed a second time.
"""

import json
from typing import Any

from pydantic import BaseModel


class WebhookBody(BaseModel):
    document_id: int
    #: Never sent by Paperless itself - its webhook template carries only
    #: document_id. Used by the manual/scripted callers: the nightly catch-up job's
    #: --force flag, and any hand-run request during testing.
    force: bool = False


def parse_webhook_body(raw_body: bytes) -> WebhookBody:
    parsed: Any = json.loads(raw_body)
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return WebhookBody.model_validate(parsed)
