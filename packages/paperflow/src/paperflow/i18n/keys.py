"""Every user-facing message key, typed.

`t()` takes an `M`, not a bare string, so a typo in a key is a mypy error caught before
the message is ever needed - not a `None` silently reaching a user at runtime. Nothing
outside `apps/` may hold one of these; see the `banned-api` rule on `paperflow.i18n` in
the root `pyproject.toml`.
"""

from enum import StrEnum


class M(StrEnum):
    # ── watcher toasts (paperflow.apps.watcher) ─────────────────────────────
    TOAST_UPLOADED_TITLE = "toast_uploaded_title"
    TOAST_UPLOADED_BODY = "toast_uploaded_body"
    #: New: the old code's failure paths in process_pdf() never told the user anything -
    #: a scan could sit failing in the queue with nothing but a log line to show for it.
    TOAST_FAILED_TITLE = "toast_failed_title"
    TOAST_FAILED_BODY = "toast_failed_body"

    # ── trigger (nightly catch-up) ───────────────────────────────────────────
    TRIGGER_ONLY_MODE = "trigger_only_mode"
    TRIGGER_DOC_COUNT = "trigger_doc_count"
    TRIGGER_SKIPPING_WATCH_FOLDER = "trigger_skipping_watch_folder"
    TRIGGER_ALREADY_UPLOADED = "trigger_already_uploaded"
    TRIGGER_UPLOADING = "trigger_uploading"
    TRIGGER_UPLOAD_FAILED = "trigger_upload_failed"
    TRIGGER_SENDING_TO_WEBHOOK = "trigger_sending_to_webhook"
    TRIGGER_WEBHOOK_RESULT = "trigger_webhook_result"
    TRIGGER_WEBHOOK_ERROR = "trigger_webhook_error"
    TRIGGER_DONE = "trigger_done"

    # ── duplicates ────────────────────────────────────────────────────────────
    DUP_FETCHING = "dup_fetching"
    DUP_TOTAL_DOCS = "dup_total_docs"
    DUP_TEXT_COUNT = "dup_text_count"
    DUP_IMAGE_COUNT = "dup_image_count"
    DUP_MISSING_IMAGE_DEPS = "dup_missing_image_deps"
    DUP_HASHING = "dup_hashing"
    DUP_NONE_FOUND = "dup_none_found"
    DUP_TEXT_MATCHES_HEADER = "dup_text_matches_header"
    DUP_TEXT_SIMILARITY = "dup_text_similarity"
    DUP_IMAGE_MATCHES_HEADER = "dup_image_matches_header"
    DUP_IMAGE_DISTANCE = "dup_image_distance"
    DUP_DOC_LINE = "dup_doc_line"
    DUP_DOC_LINK = "dup_doc_link"

    # ── tag colours ───────────────────────────────────────────────────────────
    TAGCOLOR_COUNT = "tagcolor_count"
    TAGCOLOR_DRY_RUN_LINE = "tagcolor_dry_run_line"
    TAGCOLOR_APPLIED_LINE = "tagcolor_applied_line"
    TAGCOLOR_SUMMARY_DRY_RUN = "tagcolor_summary_dry_run"
    TAGCOLOR_SUMMARY_APPLIED = "tagcolor_summary_applied"

    # ── Google Tasks one-off consent flow ───────────────────────────────────
    GOOGLE_AUTH_USING_CLIENT = "google_auth_using_client"
    GOOGLE_AUTH_SAVED = "google_auth_saved"
    GOOGLE_AUTH_CREATING_TEST_TASK = "google_auth_creating_test_task"
    GOOGLE_AUTH_TEST_TASK_CREATED = "google_auth_test_task_created"
