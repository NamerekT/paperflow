"""User-facing text in ru/en, loaded once, chosen once, never touched by the log.

Two JSON catalogues plus the typed key enum in `paperflow.i18n.keys` - no gettext, no
build step. Values use named `{}` placeholders and `str.format`, so a translator never
has to read code to update a message.

Locale selection happens once, in an application's `main()`, immediately after
`paperflow.logging_setup.configure_logging` - see each app for its own precedence
(a `--locale` flag, then `PAPERFLOW_LOCALE`, then a default). The service has no
user-facing output and never calls `set_locale`.

Logs are never localised - `docflow` does not depend on this package at all, and
`paperflow.adapters`/`paperflow.pipeline` are forbidden from importing it by the
`banned-api` rule in the root `pyproject.toml`. A user-facing string and a log record
answer to different readers, and conflating them is how a debugging session ends up
grepping logs in a language its own author does not read.
"""

import json
import logging
from functools import cache
from importlib import resources
from typing import Any

from paperflow.i18n.keys import M

logger = logging.getLogger(__name__)

DEFAULT_LOCALE = "en"

_current_locale = DEFAULT_LOCALE
#: (locale, key) pairs already warned about - a missing key logs once, not once per call.
_warned: set[tuple[str, str]] = set()


@cache
def _catalogue(locale: str) -> dict[str, str]:
    # Addressed as paperflow.i18n / "locales" / ..., not as the dotted package
    # paperflow.i18n.locales, so that locales/ can stay a plain data directory with no
    # __init__.py of its own.
    try:
        text = resources.files("paperflow.i18n").joinpath("locales", f"{locale}.json").read_text(
            encoding="utf-8"
        )
    except FileNotFoundError:
        return {}
    result: dict[str, str] = json.loads(text)
    return result


def set_locale(locale: str) -> None:
    global _current_locale
    _current_locale = locale


def get_locale() -> str:
    return _current_locale


def _warn_once(locale: str, key: str, message: str) -> None:
    if (locale, key) in _warned:
        return
    _warned.add((locale, key))
    logger.warning(message)


def t(key: M, **kwargs: Any) -> str:
    """Render `key` in the current locale.

    Falls back to `en` and then to the bare key on a miss, each logged once - never
    raises, because a missing translation must not be able to crash the tool trying to
    report something else has already gone wrong.
    """
    locale = _current_locale
    template = _catalogue(locale).get(key.value)
    if template is None:
        _warn_once(
            locale, key.value, f"missing i18n key {key.value!r} for locale {locale!r} - using en"
        )
        template = _catalogue(DEFAULT_LOCALE).get(key.value)
    if template is None:
        _warn_once(
            DEFAULT_LOCALE,
            key.value,
            f"missing i18n key {key.value!r} in the en catalogue - using the key itself",
        )
        return key.value
    return template.format(**kwargs)
