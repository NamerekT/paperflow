"""The worker's queue surviving a restart of the container.

A webhook is answered 202 before its document is processed, so a job that lives only in
memory is lost when the container is recreated mid-queue - and Paperless already logged
the call as a success. These tests pin the journal on disk that prevents that.
"""

import json

from paperflow.apps.service.journal import Job, JobJournal
from paperflow.apps.service.worker import MAX_ATTEMPTS, QueueWorker


class RecordingProcessor:
    def __init__(self, calls):
        self._calls = calls

    def process(self, doc_id, *, force):
        self._calls.append(("cascade", doc_id, force))


class RecordingReviewer:
    def __init__(self, calls):
        self._calls = calls

    def review(self, doc_id):
        self._calls.append(("review", doc_id))


def make_worker(tmp_path, calls):
    class StubSettings:
        progress_path = tmp_path / "progress.jsonl"
        queue_dir = tmp_path / "queue"

    return QueueWorker(RecordingProcessor(calls), RecordingReviewer(calls), StubSettings())


def run_to_empty(worker):
    worker.start()
    worker._queue.join()


def test_jobs_accepted_before_a_restart_are_processed_after_it_in_order(tmp_path):
    calls: list[tuple] = []
    before_restart = make_worker(tmp_path, calls)
    before_restart.enqueue(1, force=False)
    before_restart.enqueue_tag_review(2)
    before_restart.enqueue(3, force=True)
    # never started: the container is recreated before the worker got to them

    after_restart = make_worker(tmp_path, calls)
    after_restart.enqueue(4, force=False)
    run_to_empty(after_restart)

    assert calls == [
        ("cascade", 1, False),
        ("review", 2),
        ("cascade", 3, True),
        ("cascade", 4, False),
    ]


def test_a_finished_job_leaves_nothing_on_disk(tmp_path):
    calls: list[tuple] = []
    worker = make_worker(tmp_path, calls)
    worker.enqueue(1, force=False)
    run_to_empty(worker)

    assert list((tmp_path / "queue").iterdir()) == []
    assert worker.pending() == 0

    run_to_empty(make_worker(tmp_path, calls))
    assert calls == [("cascade", 1, False)]


def test_pending_counts_queued_jobs(tmp_path):
    worker = make_worker(tmp_path, [])
    worker.enqueue(1, force=False)
    worker.enqueue_tag_review(2)

    assert worker.pending() == 2


def test_a_job_the_process_keeps_dying_on_is_dropped_and_logged(tmp_path):
    journal = JobJournal(tmp_path / "queue")
    path = journal.add(Job(kind="cascade", doc_id=7, force=False))
    job = journal.read(path)
    for _ in range(MAX_ATTEMPTS):
        job = journal.mark_started(path, job)  # each start ended with the process dying
    journal.add(Job(kind="tag_review", doc_id=8))

    calls: list[tuple] = []
    run_to_empty(make_worker(tmp_path, calls))

    assert calls == [("review", 8)]
    assert list((tmp_path / "queue").iterdir()) == []
    records = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert records[0]["doc_id"] == 7
    assert "dropped" in records[0]["error"]


def test_a_job_interrupted_once_is_retried(tmp_path):
    journal = JobJournal(tmp_path / "queue")
    path = journal.add(Job(kind="cascade", doc_id=7, force=True))
    journal.mark_started(path, journal.read(path))

    calls: list[tuple] = []
    run_to_empty(make_worker(tmp_path, calls))

    assert calls == [("cascade", 7, True)]


def test_an_unreadable_job_file_is_set_aside_and_does_not_stop_the_rest(tmp_path):
    journal = JobJournal(tmp_path / "queue")
    (tmp_path / "queue" / "00000000000000000001.json").write_text("{not json", encoding="utf-8")
    journal.add(Job(kind="cascade", doc_id=5, force=False))

    calls: list[tuple] = []
    run_to_empty(make_worker(tmp_path, calls))

    assert calls == [("cascade", 5, False)]
    assert [p.name for p in (tmp_path / "queue").iterdir()] == ["00000000000000000001.json.bad"]
