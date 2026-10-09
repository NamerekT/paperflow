from pathlib import Path

import pytest

from paperflow.settings import Settings

_PATH_VARS = (
    "PAPERFLOW_WATCH_FOLDER",
    "PAPERFLOW_CREDENTIALS_DIR",
    "PAPERFLOW_FAMILY_MEMBERS_FILE",
    "PAPERFLOW_LOG_FILE",
)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "project"
    (root / "nested").mkdir(parents=True)
    for name in _PATH_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PAPERLESS_URL", "https://paperless.test")
    monkeypatch.setenv("PAPERLESS_API_TOKEN", "token")
    monkeypatch.chdir(root / "nested")
    return root


def _write_env(root: Path, body: str) -> None:
    (root / ".env").write_text(body, encoding="utf-8")


def test_defaults_live_under_the_env_file_directory(project: Path) -> None:
    _write_env(project, "")

    settings = Settings.load("watcher")

    assert settings.root == project
    assert settings.require_watcher().watch_folder == project / "scans"
    assert settings.require_watcher().log_file == project / "paperflow.log"
    assert settings.google.credentials_dir == project / "credentials"
    assert settings.family_members_file == project / "family_members.json"


def test_relative_paths_resolve_against_the_env_file_not_the_cwd(project: Path) -> None:
    _write_env(project, "PAPERFLOW_WATCH_FOLDER=inbox/scans\nPAPERFLOW_LOG_FILE=logs/w.log\n")

    watcher = Settings.load("watcher").require_watcher()

    assert watcher.watch_folder == project / "inbox" / "scans"
    assert watcher.log_file == project / "logs" / "w.log"


def test_absolute_paths_are_kept(project: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    _write_env(project, f"PAPERFLOW_CREDENTIALS_DIR={elsewhere}\n")

    assert Settings.load("cli").google.credentials_dir == elsewhere


def test_without_env_file_paths_resolve_against_the_cwd(project: Path) -> None:
    settings = Settings.load("cli", load_dotenv=False)

    assert settings.root == project / "nested"
    assert settings.family_members_file == project / "nested" / "family_members.json"
