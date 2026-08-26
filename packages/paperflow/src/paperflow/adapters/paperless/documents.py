import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from docflow.adapters.pdf.unwrap import unwrap_pkcs7_pdf
from paperflow.adapters.paperless.tasks import PaperlessTasks
from paperflow.adapters.paperless.transport import PaperlessTransport
from paperflow.domain.models import DocumentSummary

logger = logging.getLogger(__name__)


class PaperlessDocuments:
    def __init__(self, transport: PaperlessTransport, tasks: PaperlessTasks) -> None:
        self._transport = transport
        self._tasks = tasks

    # ── reading ───────────────────────────────────────────────────────────

    def get(self, doc_id: int) -> dict[str, Any]:
        document: dict[str, Any] = self._transport.get(
            f"/api/documents/{doc_id}/", versioned=True
        ).json()
        return document

    def iter_ids(self) -> Iterator[int]:
        for row in self._transport.paginate("/api/documents/", params={"fields": "id"}):
            yield int(row["id"])

    def iter_summaries(self, fields: Sequence[str]) -> Iterator[DocumentSummary]:
        """Walk the archive fetching only the fields named.

        Far cheaper than fetching each document in full, which is what the
        duplicate finder would otherwise have to do.

        Note that asking for "checksum" is pointless, and not only because the
        serialiser does not expose it for any API version, so the parameter is
        silently ignored. Even if it did: the column is declared unique, so two
        documents cannot share a checksum in the first place - Paperless
        already refuses a byte-identical re-upload. Finding visually similar
        documents needs perceptual hashing, not checksums.
        """
        wanted = list(fields)
        for row in self._transport.paginate(
            "/api/documents/", params={"fields": ",".join(wanted)}
        ):
            yield DocumentSummary(
                id=int(row["id"]),
                title=row.get("title") or "",
                original_file_name=row.get("original_file_name"),
                created=row.get("created"),
                mime_type=row.get("mime_type"),
                content=row.get("content") or "",
            )

    def download_original(self, doc_id: int, dest: Path) -> Path:
        """Fetch the original file as uploaded, not the archived version."""
        if dest.exists():
            return dest
        response = self._transport.get(
            f"/api/documents/{doc_id}/download/",
            params={"original": "true"},
            timeout=self._transport.settings.timeout_long,
        )
        dest.write_bytes(response.content)
        return dest

    # ── writing ───────────────────────────────────────────────────────────

    def upload(
        self,
        file_path: Path,
        *,
        title: str | None = None,
        correspondent_id: int | None = None,
        document_type_id: int | None = None,
        tag_ids: Sequence[int] | None = None,
        created: str | None = None,
    ) -> str:
        """Upload a file; return the id of the consume task.

        Every metadata argument is optional: a caller may hand over a bare file
        and classify it later, or supply everything up front.
        """
        file_path = Path(file_path)
        form: list[tuple[str, str]] = []
        if title:
            form.append(("title", title))
        if created:
            form.append(("created", created))
        if correspondent_id:
            form.append(("correspondent", str(correspondent_id)))
        if document_type_id:
            form.append(("document_type", str(document_type_id)))
        for tag_id in tag_ids or ():
            if tag_id:
                # A repeated field, not one comma-separated string.
                form.append(("tags", str(tag_id)))

        payload, unwrapped = self._readable_pdf(file_path)
        with payload.open("rb") as handle:
            try:
                response = self._transport.post(
                    "/api/documents/post_document/",
                    data=form or None,
                    # The original name, even when what is being sent is the
                    # unwrapped temporary copy: Paperless should record
                    # something a human recognises.
                    files={"document": (file_path.name, handle, "application/pdf")},
                    timeout=self._transport.settings.timeout_upload,
                )
            finally:
                handle.close()
                if unwrapped:
                    payload.unlink(missing_ok=True)

        return str(response.json())

    def upload_and_wait(self, file_path: Path, **kwargs: Any) -> int | None:
        task_id = self.upload(file_path, **kwargs)
        return self._tasks.wait_for(
            task_id, timeout=self._transport.settings.task_wait_timeout
        )

    def _readable_pdf(self, file_path: Path) -> tuple[Path, bool]:
        """Return a path Paperless will accept, and whether it is temporary.

        Some files - bank statements carrying a qualified electronic signature
        - do not begin with the PDF header even though the PDF inside is
        perfectly readable. Paperless rejects them as an unsupported mime type
        before it ever opens the file. See `docflow.adapters.pdf.unwrap`.
        """
        extracted = unwrap_pkcs7_pdf(file_path.read_bytes())
        if extracted is None:
            return file_path, False

        temp_path = file_path.with_name(f".{file_path.stem}__unwrapped.pdf")
        temp_path.write_bytes(extracted)
        logger.info(
            f"{file_path.name}: does not start with a PDF header (looks like a PKCS#7 "
            f"signature container) - extracted the real PDF, {len(extracted)} bytes"
        )
        return temp_path, True

    def patch(self, doc_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        """Raw PATCH. Callers writing custom fields must go through
        `PaperlessCustomFields.patch` instead - see its docstring."""
        updated: dict[str, Any] = self._transport.patch(
            f"/api/documents/{doc_id}/", versioned=True, json=payload
        ).json()
        return updated

    def url_for(self, doc_id: int) -> str:
        return self._transport.document_url(doc_id)
