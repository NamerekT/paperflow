"""Parsing the webhook body, including Paperless's double-JSON-encoding quirk.

See `paperflow.apps.service.schemas` for the real case behind the double-encoding
branch: with certain Webhook action settings, Paperless serialises an already-serialised
JSON string a second time, so the request body is a JSON string containing JSON.
"""

import json

import pytest
from pydantic import ValidationError

from paperflow.apps.service.schemas import parse_webhook_body


def test_a_plain_object_is_parsed_directly():
    body = parse_webhook_body(json.dumps({"document_id": 307}).encode())
    assert body.document_id == 307
    assert body.force is False


def test_the_double_encoded_body_paperless_actually_sends_is_unwrapped():
    inner = json.dumps({"document_id": 307})
    wrapped = json.dumps(inner)  # a JSON string containing JSON
    body = parse_webhook_body(wrapped.encode())
    assert body.document_id == 307


def test_force_defaults_to_false():
    body = parse_webhook_body(json.dumps({"document_id": 1, "force": True}).encode())
    assert body.force is True


def test_a_json_array_is_rejected():
    with pytest.raises(ValueError):
        parse_webhook_body(b"[]")


def test_a_body_without_document_id_is_rejected():
    with pytest.raises(ValidationError):
        parse_webhook_body(json.dumps({"force": True}).encode())
