"""The watcher's queue numbering and its upload/backup/toast cycle.

`upload_pdf` and `finish_pdf` are exercised against fakes for the Paperless adapters and
a monkeypatched `send_toast` - no real network, no real Windows notification. What is
pinned here is orchestration: which upload failures leave a file in the queue for a
retry, which still count as a successful upload despite a missing confirmation, and that
a toast always fires on a genuine failure (the old code's silent failure paths, which the
plan called out explicitly).
"""

import os
import threading
import time
from pathlib import Path

import pytest

from paperflow.apps.watcher import main as watcher_main
from paperflow.apps.watcher.queue import ScanQueue
from paperflow.apps.watcher.worker import QueueWorker
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
    def __init__(self, *, upload_error=None):
        self._upload_error = upload_error
        self.uploaded: list[Path] = []

    def upload(self, path):
        if self._upload_error is not None:
            raise self._upload_error
        self.uploaded.append(path)
        return "task-123"

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


def test_upload_pdf_archives_under_the_queue_name(tmp_path):
    settings = make_settings(tmp_path)
    path = make_scan(tmp_path)
    documents = FakeDocuments()

    result = watcher_main.upload_pdf(path, documents, settings)

    assert result == ("task-123", "scan_0001.pdf")
    assert documents.uploaded == [path]
    assert not path.exists()  # moved into the archive folder
    assert (settings.archive_folder / "scan_0001.pdf").exists()


def test_finish_pdf_toasts_with_a_link_once_confirmed(tmp_path, recorded_toasts):
    settings = make_settings(tmp_path)
    documents = FakeDocuments()
    tasks = FakeTasks(doc_id=42)

    watcher_main.finish_pdf("task-123", "scan_0001.pdf", documents, tasks, settings)

    assert recorded_toasts == [
        ("📄 Uploaded, processing...", "scan_0001.pdf", "https://paperless.test/documents/42/details")
    ]


def test_a_task_timeout_still_toasts_but_without_a_link(tmp_path, recorded_toasts):
    settings = make_settings(tmp_path)
    documents = FakeDocuments()
    tasks = FakeTasks(error=TaskTimeout("still pending"))

    watcher_main.finish_pdf("task-123", "scan_0001.pdf", documents, tasks, settings)

    assert recorded_toasts[0][2] is None  # no open_target - doc id was never confirmed


def test_a_task_failure_still_toasts_but_without_a_link(tmp_path, recorded_toasts):
    settings = make_settings(tmp_path)
    documents = FakeDocuments()
    tasks = FakeTasks(error=TaskFailed("rejected"))

    watcher_main.finish_pdf("task-123", "scan_0001.pdf", documents, tasks, settings)

    assert recorded_toasts[0][2] is None


def test_paperless_unavailable_leaves_the_file_queued_with_no_toast(
    tmp_path, recorded_toasts, monkeypatch
):
    monkeypatch.setattr(watcher_main.time, "sleep", lambda seconds: None)
    settings = make_settings(tmp_path)
    path = make_scan(tmp_path)
    documents = FakeDocuments(upload_error=PaperlessUnavailable("tunnel down"))

    result = watcher_main.upload_pdf(path, documents, settings)

    assert result is None
    assert path.exists()  # never left the queue
    assert recorded_toasts == []


def test_an_unexpected_upload_error_does_not_leave_the_document_silent(tmp_path, recorded_toasts):
    settings = make_settings(tmp_path)
    path = make_scan(tmp_path)
    documents = FakeDocuments(upload_error=ValueError("something this code did not anticipate"))

    result = watcher_main.upload_pdf(path, documents, settings)

    assert result is None
    assert len(recorded_toasts) == 1
    assert recorded_toasts[0][0] == "Upload failed"


# ── autonomy: queueing must not depend on Paperless being up ────────────────


def test_stray_pdfs_left_in_the_watch_folder_are_adopted_at_startup(tmp_path):
    settings = make_settings(tmp_path)
    settings.watch_folder.mkdir(parents=True)
    stray = settings.watch_folder / "Scan.pdf"
    stray.write_bytes(b"%PDF-1.7 left over from a previous session")
    os.utime(stray, (0, 0))
    queue = ScanQueue(settings)

    adopted = watcher_main.adopt_strays(queue, settings)

    assert [path.name for path in adopted] == ["scan_0001.pdf"]
    assert not stray.exists()


def test_a_scan_still_being_written_is_left_to_the_observer(tmp_path):
    settings = make_settings(tmp_path)
    settings.watch_folder.mkdir(parents=True)
    fresh = settings.watch_folder / "Scan.pdf"
    fresh.write_bytes(b"%PDF-1.7 half written")
    queue = ScanQueue(settings)

    assert watcher_main.adopt_strays(queue, settings) == []
    assert fresh.exists()


def test_adopting_never_reaches_into_the_queue_or_archive_folders(tmp_path):
    settings = make_settings(tmp_path)
    settings.queue_folder.mkdir(parents=True)
    settings.archive_folder.mkdir(parents=True)
    queued = settings.queue_folder / "scan_0001.pdf"
    queued.write_bytes(b"already queued")
    archived = settings.archive_folder / "scan_0001.pdf"
    archived.write_bytes(b"already uploaded")
    os.utime(queued, (0, 0))
    os.utime(archived, (0, 0))

    assert watcher_main.adopt_strays(ScanQueue(settings), settings) == []
    assert queued.exists()
    assert archived.exists()


def test_enqueue_survives_the_source_being_taken_by_a_racing_pass(tmp_path):
    settings = make_settings(tmp_path)
    queue = ScanQueue(settings)

    assert queue.enqueue(tmp_path / "gone.pdf") is None


def test_the_worker_queues_but_does_not_upload_until_paperless_answers(tmp_path):
    settings = WatcherSettings(
        watch_folder=tmp_path / "watch",
        log_file=tmp_path / "watcher.log",
        idle_check_interval=0.05,
    )
    make_scan(tmp_path / "watch")
    uploaded: list[Path] = []
    ready = threading.Event()

    worker = QueueWorker(
        ScanQueue(settings),
        settings,
        upload=lambda path: (uploaded.append(path), ("task-1", path.name))[1],
        finish=lambda task_id, name: None,
        paperless_ready=ready,
    )
    worker.start()

    time.sleep(0.3)
    assert uploaded == []  # Paperless is still down; the scan waits on disk

    ready.set()
    deadline = time.monotonic() + 3.0
    while not uploaded and time.monotonic() < deadline:
        time.sleep(0.05)
    assert [path.name for path in uploaded] == ["scan_0001.pdf"]
