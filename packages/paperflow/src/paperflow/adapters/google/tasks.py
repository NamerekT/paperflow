"""Deadline reminders in Google Tasks.

One Google account, one dedicated task list, confirmed sufficient with the user rather
than built for many. The OAuth client (a Desktop app entry) is created by hand once in
Google Cloud Console; its client_secret*.json and, after a one-off interactive consent,
token.json live in `GoogleTasksSettings.credentials_dir`. The consent flow itself is not
here - see `paperflow.apps.cli.google_auth` - because a background service has no
browser to open one with. Missing or broken credentials are a startup-time question for
that flow to answer, not something this class tries to fix by launching one.

Once token.json exists, this refreshes the access token through its refresh_token as
needed and never prompts for anything.
"""

import logging
from datetime import date
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import Resource, build

from paperflow.errors import PaperflowError
from paperflow.settings import GoogleTasksSettings

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/tasks"]


class GoogleTasksUnavailable(PaperflowError):
    """Credentials are missing or cannot be silently refreshed."""


class GoogleTasks:
    """Creates a reminder for a Frist deadline."""

    def __init__(self, settings: GoogleTasksSettings) -> None:
        self._settings = settings
        self._token_file = settings.credentials_dir / "token.json"
        self._service: Resource | None = None
        self._tasklist_id: str | None = None

    def _load_service(self) -> Resource:
        if not self._token_file.exists():
            raise GoogleTasksUnavailable(
                f"{self._token_file} does not exist - run the one-off consent flow first "
                "(a background service has no browser to open one with)"
            )
        # google-auth ships partial type information; this constructor itself carries
        # none, so the call is untyped even though the module resolves fine.
        creds = Credentials.from_authorized_user_file(  # type: ignore[no-untyped-call]
            str(self._token_file), SCOPES
        )
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                self._token_file.write_text(creds.to_json(), encoding="utf-8")
            else:
                raise GoogleTasksUnavailable(
                    f"{self._token_file} is invalid and has no refresh_token to silently "
                    "renew it with - redo the one-off consent flow"
                )
        return build("tasks", "v1", credentials=creds)

    def _tasklist_id_for(self, service: Resource) -> str:
        """Find `list_name`, case-insensitively, creating it if absent.

        Cached for the life of this instance - one process handles many documents
        against the same list, and re-listing every tasklist before every reminder
        would be a wasted round trip for a value that essentially never changes.
        """
        if self._tasklist_id is not None:
            return self._tasklist_id

        response = service.tasklists().list(maxResults=100).execute()
        for entry in response.get("items", []):
            if entry["title"].casefold() == self._settings.list_name.casefold():
                self._tasklist_id = entry["id"]
                return self._tasklist_id

        created = service.tasklists().insert(body={"title": self._settings.list_name}).execute()
        self._tasklist_id = created["id"]
        return self._tasklist_id

    def create_task(self, title: str, due: date, *, notes: str | None = None) -> str:
        """Create a reminder due on `due`; return the created task's id.

        The Tasks API wants an RFC3339 timestamp in `due`, but the interface only ever
        shows the date - the time of day is not surfaced anywhere - so this appends
        midnight UTC rather than tracking a time nobody will see.
        """
        if self._service is None:
            self._service = self._load_service()
        tasklist_id = self._tasklist_id_for(self._service)

        body: dict[str, Any] = {"title": title, "due": f"{due.isoformat()}T00:00:00.000Z"}
        if notes:
            body["notes"] = notes
        created = self._service.tasks().insert(tasklist=tasklist_id, body=body).execute()
        return str(created["id"])
