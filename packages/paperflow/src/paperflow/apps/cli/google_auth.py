"""One-off interactive Google consent for Frist reminders in Google Tasks.

Run by hand, once, on a machine with a browser:

    paperflow-google-auth

An OAuth client (a Desktop app entry) must already exist in Google Cloud Console, with
its client_secret*.json placed in the credentials directory
(`PAPERFLOW_CREDENTIALS_DIR`). This opens a browser for the one-time consent and writes
token.json next to it. From then on, `paperflow.adapters.google.tasks.GoogleTasks`
refreshes the access token through its refresh_token as needed and never opens a
browser again - which is also why this flow is not, and cannot be, run from inside the
webhook service's container: it has none to open.
"""

import argparse
import glob
from datetime import date
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

from paperflow.adapters.google.tasks import SCOPES, GoogleTasks
from paperflow.apps.cli import console
from paperflow.i18n import t
from paperflow.i18n.keys import M
from paperflow.settings import Settings


def _find_client_secret_file(credentials_dir: Path) -> str:
    matches = glob.glob(str(credentials_dir / "client_secret*.json"))
    if not matches:
        raise FileNotFoundError(
            f"no client_secret*.json found in {credentials_dir} - need an OAuth client "
            "(Desktop app) from Google Cloud Console"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple client_secret*.json files in {credentials_dir} - keep exactly one: {matches}"
        )
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locale", default=None, choices=["ru", "en"], help="output language")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.load("cli")
    console.configure_console()
    console.apply_locale(args.locale, settings)

    settings.google.credentials_dir.mkdir(parents=True, exist_ok=True)
    client_secret_file = _find_client_secret_file(settings.google.credentials_dir)
    console.echo(t(M.GOOGLE_AUTH_USING_CLIENT, path=client_secret_file))

    flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
    # access_type="offline" is required - without it there is no refresh_token, and the
    # browser login would have to repeat every time the access token expires (~1h),
    # which a background service cannot do.
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    token_file = settings.google.credentials_dir / "token.json"
    token_file.write_text(creds.to_json(), encoding="utf-8")
    console.echo(t(M.GOOGLE_AUTH_SAVED, path=str(token_file)))

    console.echo(t(M.GOOGLE_AUTH_CREATING_TEST_TASK))
    google_tasks = GoogleTasks(settings.google)
    task_id = google_tasks.create_task(
        "Paperflow - test task",
        date.today(),
        notes="Safe to delete - this only checks the Google Tasks setup.",
    )
    console.echo(
        t(M.GOOGLE_AUTH_TEST_TASK_CREATED, task_id=task_id, list_name=settings.google.list_name)
    )


if __name__ == "__main__":
    main()
