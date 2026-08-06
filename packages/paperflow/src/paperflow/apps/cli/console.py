"""Console setup shared by the CLI tools."""

import sys

from paperflow.i18n import set_locale
from paperflow.settings import Settings


def configure_console() -> None:
    """UTF-8 output, unconditionally.

    A Windows console defaults to a code page that cannot encode much beyond Latin-1;
    printing a document title with an umlaut or a Cyrillic name used to raise
    UnicodeEncodeError and kill the whole run partway through.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def echo(message: str) -> None:
    print(message, flush=True)


def apply_locale(cli_locale: str | None, settings: Settings) -> None:
    """--locale flag, then PAPERFLOW_LOCALE (already resolved onto `settings.locale`
    by `Settings.load`), then its own "ru" default."""
    locale = (cli_locale or settings.locale).strip().lower() or "ru"
    set_locale(locale)
