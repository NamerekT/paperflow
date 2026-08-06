"""The watcher's queue numbering and its upload/backup/toast cycle.

`process_pdf` is exercised against fakes for the Paperless adapters and a monkeypatched
`send_toast` - no real network, no real Windows notification. What is pinned here is
orchestration: which failures leave a file in the queue for a retry, which ones still
count as success without a direct link, and that a toast always fires on a genuine
failure (the old code's silent failure paths, which the plan called out explicitly).
"""

from pathlib import Path

import pytest

from paperflow.apps.watcher import main as watcher_main
from paperflow.apps.watcher.queue import ScanQueue
from paperflow.errors import PaperlessUnavailable, TaskFailed, TaskTimeout
from paperflow.settings import WatcherSettings


def make_settings(tmp_path: Path) -> WatcherSettings:
    return WatcherSettings(watch_folder=tmp_path / "watch", log_file=tmp_path / "watcher.log")


def test_queue_numbers_are_sequential_and_start_at_one(tmp_path):
    settings = make_settings(tmp_path)
    queue = ScanQueue(settings)
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    a.write_bytes(b"x")
    b.write_bytes(b"y")

    first = queue.enqueue(a)
    second = queue.enqueue(b)

    assert first.name == "scan_0001.pdf"
    assert second.name == "scan_0002.pdf"
    assert queue.pending() == [first, second]


def test_queue_numbering_resumes_after_files_are_removed(tmp_path):
    settings = make_settings(tmp_path)
    queue = ScanQueue(settings)
    settings.queue_folder.mkdir(parents=True)
    (settings.queue_folder / "scan_0003.pdf").write_bytes(b"leftover")

    src = tmp_path / "new.pdf"
    src.write_bytes(b"z")
    queued = queue.enqueue(src)

    assert queued.name == "scan_0004.pdf"


class FakeDocuments:
    def __init__(self, *, title=None, get_error=None):
        self._title = title
        self._get_error = get_error
        self.uploaded: list[Path] = []

    def upload(self, path):
        self.uploaded.append(path)
        return "task-123"

    def get(self, doc_id):
        if self._get_error is not None:
            raise self._get_error
        return {"title": self._title}

    def url_for(self, doc_id):
        return f"https://paperless.test/documents/{doc_id}/details"


class FakeTasks:
    def __init__(self, *, doc_id=42, error=None):
        self._doc_id = doc_id
        self._error = error

    def wait_for(self, task_id, *, timeout):
        if self._error is not None:
            raise self._error
        return self._doc_id


@pytest.fixture
def recorded_toasts(monkeypatch):
    calls = []

    def fake_send_toast(title, body, *, open_target=None):
        calls.append((title, body, open_target))

    monkeypatch.setattr(watcher_main, "send_toast", fake_send_toast)
    return calls


def make_scan(tmp_path) -> Path:
    path = tmp_path / "queue" / "scan_0001.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.7 fake")
    return path


def test_happy_path_uploads_archives_and_toasts_with_a_link(tmp_path, recorded_toasts):
    settings = make_settings(tmp_path)
    path = make_scan(tmp_path)
    documents = FakeDocuments(title="Bescheid Januar")
    tasks = FakeTasks(doc_id=42)

    result = watcher_main.process_pdf(path, documents, tasks, settings)

    assert result is True
    assert documents.uploaded == [path]
    assert not path.exists()  # moved into the archive folder
    assert (settings.archive_folder / "Bescheid Januar.pdf").exists()
    assert recorded_toasts == [
        ("📄 Uploaded, processing...", "Bescheid Januar", "https://paperless.test/documents/42/details")
    ]


def test_a_task_timeout_still_archives_but_toasts_without_a_link(tmp_path, recorded_toasts):
    settings = make_settings(tmp_path)
    path = make_scan(tmp_path)
    documents = FakeDocuments()
    tasks = FakeTasks(error=TaskTimeout("still pending"))

    result = watcher_main.process_pdf(path, documents, tasks, settings)

    assert result is True
    assert not path.exists()
    assert recorded_toasts[0][2] is None  # no open_target - doc id was never confirmed


def test_a_task_failure_still_archives_but_toasts_without_a_link(tmp_path, recorded_toasts):
    settings = make_settings(tmp_path)
    path = make_scan(tmp_path)
    documents = FakeDocuments()
    tasks = FakeTasks(error=TaskFailed("rejected"))

    result = watcher_main.process_pdf(path, documents, tasks, settings)

    assert result is True
    assert recorded_toasts[0][2] is None


def test_paperless_unavailable_leaves_the_file_queued_with_no_toast(
    tmp_path, recorded_toasts, monkeypatch
):
    monkeypatch.setattr(watcher_main.time, "sleep", lambda seconds: None)
    settings = make_settings(tmp_path)
    path = make_scan(tmp_path)

    class BrokenDocuments:
        def upload(self, path):
            raise PaperlessUnavailable("tunnel down")

    result = watcher_main.process_pdf(path, BrokenDocuments(), FakeTasks(), settings)

    assert result is False
    assert path.exists()  # never left the queue
    assert recorded_toasts == []


def test_an_unexpected_error_does_not_leave_the_document_silent(tmp_path, recorded_toasts):
    settings = make_settings(tmp_path)
    path = make_scan(tmp_path)

    class BrokenDocuments:
        def upload(self, path):
            raise ValueError("something this code did not anticipate")

    result = watcher_main.process_pdf(path, BrokenDocuments(), FakeTasks(), settings)

    assert result is False
    assert len(recorded_toasts) == 1
    assert recorded_toasts[0][0] == "Upload failed"
