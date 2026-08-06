"""The locale catalogues stay in sync with the key enum, and every value actually
formats.

Two failure modes this guards against: a key added to `M` but forgotten in one of the
JSON files (a silent fallback to `en`, or to the bare key, at runtime instead of a
translated string), and a template whose `{placeholder}` names do not match what the
caller actually passes (a `KeyError` raised from inside `t()` the first time that
message is ever shown for real).
"""

import json
import string
from importlib import resources

import pytest

from paperflow.i18n import DEFAULT_LOCALE, t
from paperflow.i18n.keys import M

LOCALES = ("en", "ru")


def _load(locale: str) -> dict[str, str]:
    text = resources.files("paperflow.i18n").joinpath("locales", f"{locale}.json").read_text(
        encoding="utf-8"
    )
    result: dict[str, str] = json.loads(text)
    return result


def _placeholders(template: str) -> set[str]:
    return {
        field
        for _, field, _, _ in string.Formatter().parse(template)
        if field is not None and field != ""
    }


@pytest.mark.parametrize("locale", LOCALES)
def test_catalogue_has_exactly_the_keys_the_enum_declares(locale):
    catalogue = _load(locale)
    assert set(catalogue) == {member.value for member in M}


def test_default_locale_is_one_of_the_shipped_catalogues():
    assert DEFAULT_LOCALE in LOCALES


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("member", list(M))
def test_every_value_formats_with_its_own_placeholders(locale, member):
    template = _load(locale)[member.value]
    kwargs = dict.fromkeys(_placeholders(template), "x")
    template.format(**kwargs)  # raises KeyError/IndexError on a mismatch


def test_t_renders_a_known_key_in_each_locale():
    from paperflow.i18n import set_locale

    for locale in LOCALES:
        set_locale(locale)
        assert t(M.TRIGGER_DONE, ok=1, failed=0) == _load(locale)[M.TRIGGER_DONE.value].format(
            ok=1, failed=0
        )
    set_locale("ru")  # restore the project default so later tests are not order-dependent


def test_a_missing_key_falls_back_to_english_then_to_the_key_itself(monkeypatch):
    from paperflow.i18n import set_locale

    monkeypatch.setattr(
        "paperflow.i18n._catalogue",
        lambda locale: {} if locale == "ru" else {"trigger_done": "done: {ok}/{failed}"},
    )
    set_locale("ru")
    assert t(M.TRIGGER_DONE, ok=1, failed=0) == "done: 1/0"
    set_locale("en")
