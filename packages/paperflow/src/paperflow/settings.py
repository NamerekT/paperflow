"""Application configuration for the Paperless binding.

Composes the engine's settings with everything Paperless-, watcher- and
service-specific, and validates the lot in one pass so that a misconfigured
process reports every missing variable at once instead of one per restart.

Nothing here has a secret default. The Paperless token and the model gateway
key used to be spelled out in source, which was convenient right up to the
first commit.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from docflow.errors import ConfigError
from docflow.settings import CascadeSettings, EnvReader, LlmSettings

#: Which application is asking. Each needs a different set of variables, and
#: the watcher must not fail to start because the service's webhook secret is
#: unset - it does not know the webhook exists.
Profile = Literal["watcher", "service", "cli"]

_ENV_HINT = (
    "Copy .env.example to .env and fill it in, "
    "or set the variables in the container environment."
)


def load_dotenv_if_present(start: Path | None = None) -> Path | None:
    """Load a .env file found at or above `start`, if python-dotenv is here.

    The container has no .env - variables arrive from docker-compose - so a
    missing file is normal and never an error. python-dotenv does not
    overwrite variables that are already set, so an explicit export always
    beats the file.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        env_file = candidate / ".env"
        if env_file.is_file():
            load_dotenv(env_file)
            return env_file
    return None


@dataclass(frozen=True, slots=True)
class PaperlessSettings:
    """Where the archive is and how patient to be with it.

    The four timeouts replace values that were hardcoded at each of fifteen
    call sites, where a download and a tag lookup had wildly different limits
    for no stated reason.
    """

    url: str
    token: str
    timeout_short: float = 20.0  # list, create, poll a task
    timeout_default: float = 30.0  # fetch or patch one document
    timeout_long: float = 60.0  # walk a paginated collection, download a file
    timeout_upload: float = 120.0
    task_wait_timeout: float = 120.0

    #: Requested page size when walking a collection. Paperless paginates
    #: everything; asking for a large page keeps the common case to a single
    #: round trip while the next-link handling stays as the safety net.
    page_size: int = 1000

    @classmethod
    def from_env(cls, env: EnvReader) -> "PaperlessSettings":
        return cls(
            url=env.required("PAPERLESS_URL").rstrip("/"),
            token=env.required("PAPERLESS_API_TOKEN"),
            timeout_short=env.number("PAPERLESS_TIMEOUT_SHORT", 20.0),
            timeout_default=env.number("PAPERLESS_TIMEOUT_DEFAULT", 30.0),
            timeout_long=env.number("PAPERLESS_TIMEOUT_LONG", 60.0),
            timeout_upload=env.number("PAPERLESS_TIMEOUT_UPLOAD", 120.0),
            task_wait_timeout=env.number("PAPERLESS_TASK_WAIT_TIMEOUT", 120.0),
            page_size=env.integer("PAPERLESS_PAGE_SIZE", 1000),
        )


@dataclass(frozen=True, slots=True)
class WatcherSettings:
    watch_folder: Path
    log_file: Path

    #: The scanner does not write the file atomically, so a newly created path
    #: may still be growing. Wait this long after the last change before
    #: touching it.
    write_settle_delay: float = 4.0

    #: How often the queue worker looks for work it missed - a file that
    #: appeared while Paperless was down, or an event the observer dropped.
    idle_check_interval: float = 20.0

    #: How long to wait for Paperless to finish consuming an upload, so the
    #: notification can carry a direct link to the finished document.
    paperless_wait_timeout: float = 60.0

    #: How many background confirmations (waiting on paperless_wait_timeout,
    #: then the toast) may run at once. Pure I/O - a GET poll every few
    #: seconds, touches no file - so this can be generous; it has no bearing
    #: on upload throughput, which stays sequential in the dispatcher.
    max_concurrent_confirmations: int = 20

    @property
    def queue_folder(self) -> Path:
        """Scans accepted but not yet uploaded."""
        return self.watch_folder / "queue"

    @property
    def archive_folder(self) -> Path:
        """Local copies kept after a successful upload."""
        return self.watch_folder / "Uploaded to Paperless"

    @classmethod
    def from_env(cls, env: EnvReader) -> "WatcherSettings":
        watch = env.required("PAPERFLOW_WATCH_FOLDER")
        return cls(
            watch_folder=Path(watch),
            log_file=Path(env.optional("PAPERFLOW_LOG_FILE", "") or (Path.cwd() / "paperflow.log")),
            write_settle_delay=env.number("PAPERFLOW_WRITE_SETTLE_DELAY", 4.0),
            idle_check_interval=env.number("PAPERFLOW_IDLE_CHECK_INTERVAL", 20.0),
            paperless_wait_timeout=env.number("PAPERFLOW_PAPERLESS_WAIT_TIMEOUT", 60.0),
            max_concurrent_confirmations=env.integer("PAPERFLOW_MAX_CONCURRENT_CONFIRMATIONS", 20),
        )


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    #: Shared with the catch-up CLI, which authenticates to this service with
    #: it. No default: an unset secret must stop the process, because the
    #: endpoint's only protection is this token and starting without one would
    #: let anything that reaches the port enqueue arbitrary document ids.
    webhook_secret: str
    port: int = 8000

    #: When false, everything runs except the writes back to Paperless. The
    #: only dry-run switch in the system.
    apply_changes: bool = True

    progress_path: Path = Path("/data/cascade_progress.jsonl")

    @classmethod
    def from_env(cls, env: EnvReader) -> "ServiceSettings":
        return cls(
            webhook_secret=env.required("WEBHOOK_SHARED_SECRET"),
            port=env.integer("WEBHOOK_PORT", 8000),
            apply_changes=env.flag("OCR_CASCADE_APPLY", True),
            progress_path=Path(
                env.optional("CASCADE_PROGRESS_FILE", "/data/cascade_progress.jsonl")
            ),
        )


@dataclass(frozen=True, slots=True)
class TriggerSettings:
    """Talking to this deployment's own webhook service, as a client.

    Separate from `ServiceSettings` even though the secret is the same value: the
    service reads it to check an incoming token, the nightly catch-up job reads it to
    send one. And separate from the shared `Settings.load("cli")` path entirely - of
    the four CLI tools, only the catch-up job touches the webhook at all, and it must
    not force an unset WEBHOOK_SHARED_SECRET to fail startup for duplicates, tag-colour
    or Google auth runs that have no use for it.
    """

    webhook_url: str
    webhook_secret: str

    @classmethod
    def from_env(cls, env: EnvReader) -> "TriggerSettings":
        return cls(
            webhook_url=env.optional("WEBHOOK_URL", "http://localhost:8002/webhook/ocr-cascade"),
            webhook_secret=env.required("WEBHOOK_SHARED_SECRET"),
        )

    @classmethod
    def load(cls) -> "TriggerSettings":
        """Load and validate on its own, failing once for just these variables."""
        env = EnvReader(hint=_ENV_HINT)
        settings = cls.from_env(env)
        env.raise_if_incomplete()
        return settings


@dataclass(frozen=True, slots=True)
class GoogleTasksSettings:
    #: Holds client_secret*.json and, after a one-off consent run, token.json.
    #:
    #: Explicit rather than derived from the package location, and that is the
    #: single most important line in this module. It used to be computed as
    #: "two directories above the config module", which resolved to /app in
    #: the container - exactly where the compose volume mounts it. Moving the
    #: code into src/ would have quietly repointed it at
    #: /app/src/paperflow/credentials, and the only symptom would have been
    #: reminders silently no longer being created.
    credentials_dir: Path

    list_name: str = "Paperflow Fristen"
    create_tasks: bool = False

    @classmethod
    def from_env(cls, env: EnvReader) -> "GoogleTasksSettings":
        return cls(
            credentials_dir=Path(
                env.optional("PAPERFLOW_CREDENTIALS_DIR", "") or (Path.cwd() / "credentials")
            ),
            list_name=env.optional("GOOGLE_TASKS_LIST_NAME", "Paperflow Fristen"),
            create_tasks=env.flag("FRIST_CREATE_TASKS", False),
        )


@dataclass(frozen=True, slots=True)
class Settings:
    paperless: PaperlessSettings
    google: GoogleTasksSettings
    family_members_file: Path
    llm: LlmSettings | None = None
    cascade: CascadeSettings = field(default_factory=CascadeSettings)
    watcher: WatcherSettings | None = None
    service: ServiceSettings | None = None
    locale: str = "ru"
    log_level: str = "INFO"

    @classmethod
    def load(cls, profile: Profile, *, load_dotenv: bool = True) -> "Settings":
        """Build settings for one application, failing once for all problems."""
        if load_dotenv:
            load_dotenv_if_present()

        env = EnvReader(hint=_ENV_HINT)

        paperless = PaperlessSettings.from_env(env)
        google = GoogleTasksSettings.from_env(env)
        family_members_file = Path(
            env.optional("PAPERFLOW_FAMILY_MEMBERS_FILE", "")
            or (Path.cwd() / "family_members.json")
        )

        # Only the service runs the engine. The watcher is a dumb uploader and
        # the CLI tools delegate all processing to the service over HTTP, so
        # neither should fail to start over an unset model-gateway key.
        llm = LlmSettings.from_env(env) if profile == "service" else None
        cascade = CascadeSettings.from_env(env)
        watcher = WatcherSettings.from_env(env) if profile == "watcher" else None
        service = ServiceSettings.from_env(env) if profile == "service" else None

        env.raise_if_incomplete()

        return cls(
            paperless=paperless,
            google=google,
            family_members_file=family_members_file,
            llm=llm,
            cascade=cascade,
            watcher=watcher,
            service=service,
            locale=os.environ.get("PAPERFLOW_LOCALE", "ru").strip().lower() or "ru",
            log_level=os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        )

    def require_llm(self) -> LlmSettings:
        if self.llm is None:
            raise ConfigError("LLM settings were not loaded for this profile")
        return self.llm

    def require_service(self) -> ServiceSettings:
        if self.service is None:
            raise ConfigError("service settings were not loaded for this profile")
        return self.service

    def require_watcher(self) -> WatcherSettings:
        if self.watcher is None:
            raise ConfigError("watcher settings were not loaded for this profile")
        return self.watcher
