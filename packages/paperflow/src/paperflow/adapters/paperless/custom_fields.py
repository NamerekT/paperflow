import logging
from typing import Any

from paperflow.adapters.paperless.transport import PaperlessTransport
from paperflow.domain.models import CustomField, CustomFieldSet, CustomFieldSpec, CustomFieldValue
from paperflow.errors import PaperlessError

logger = logging.getLogger(__name__)


class PaperlessCustomFields:
    def __init__(self, transport: PaperlessTransport) -> None:
        self._transport = transport

    # ── schema ────────────────────────────────────────────────────────────

    def fetch_all(self) -> dict[str, CustomField]:
        fields: dict[str, CustomField] = {}
        for row in self._transport.paginate("/api/custom_fields/"):
            fields[row["name"]] = CustomField(
                id=int(row["id"]),
                name=row["name"],
                data_type=row["data_type"],
                extra_data=row.get("extra_data"),
            )
        return fields

    def ensure(self, specs: tuple[CustomFieldSpec, ...]) -> CustomFieldSet:
        """Create what is missing, verify what is already there.

        The type check is the point. A pre-existing field of the wrong type
        would otherwise be discovered at the first write - which happens after
        the document has been downloaded, OCR'd and classified, so the cost of
        finding out late is the whole expensive part of the pipeline.
        """
        existing = self.fetch_all()
        resolved: dict[str, CustomField] = {}

        for spec in specs:
            field = existing.get(spec.name)

            if field is None:
                if spec.externally_managed:
                    # Deliberately not created here. A field the operator is
                    # expected to define by hand should be missing loudly
                    # rather than conjured with settings nobody chose.
                    raise PaperlessError(
                        f"custom field {spec.name!r} (type {spec.data_type}) does not exist in "
                        f"Paperless - create it by hand under Settings, Custom fields"
                    )
                field = self._create(spec)
            elif field.data_type != spec.data_type:
                raise PaperlessError(
                    f"custom field {spec.name!r} is of type {field.data_type!r} in Paperless, "
                    f"but this application writes {spec.data_type!r} into it"
                )
            else:
                logger.debug(f"custom field present: {spec.name!r} (id={field.id})")

            resolved[spec.name] = field

        return CustomFieldSet(by_name=resolved)

    def _create(self, spec: CustomFieldSpec) -> CustomField:
        payload: dict[str, Any] = {"name": spec.name, "data_type": spec.data_type}
        if spec.select_options is not None:
            payload["extra_data"] = {
                "select_options": [{"label": label} for label in spec.select_options]
            }
        row = self._transport.post("/api/custom_fields/", versioned=True, json=payload).json()
        logger.info(f"created custom field {spec.name!r} (id={row['id']})")
        return CustomField(
            id=int(row["id"]),
            name=row["name"],
            data_type=row["data_type"],
            extra_data=row.get("extra_data"),
        )

    # ── values ────────────────────────────────────────────────────────────

    def select_option_value(self, field: CustomField, label: str) -> Any:
        """The value to write into a select field for a given label.

        From API v7 the server returns select options as {id, label} objects
        and wants the id, a string. Before v7 they are bare label strings and
        the value is the label's index in that list, a number.
        """
        if field.extra_data is None:
            raise PaperlessError(f"custom field {field.name!r} has no select options")
        options = field.extra_data["select_options"]

        if self._transport.api_version >= 7:
            for option in options:
                if option["label"] == label:
                    return option["id"]
        else:
            for index, option_label in enumerate(options):
                if option_label == label:
                    return index

        raise PaperlessError(f"option {label!r} not found on field {field.name!r}: {options}")

    @staticmethod
    def value_on(document: dict[str, Any], field_id: int) -> Any:
        for entry in document.get("custom_fields") or []:
            if entry.get("field") == field_id:
                return entry.get("value")
        return None

    def patch(
        self,
        doc_id: int,
        values: list[CustomFieldValue],
        *,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write custom-field values, preserving the ones not mentioned.

        On this instance, PATCHing a document with custom_fields REPLACES the
        whole list rather than merging the entries sent. Reading the
        paperless-ngx sources suggests it should merge - the serialiser calls
        update_or_create per field and leaves the rest alone - but the deployed
        server does not behave that way, whether through a version difference
        or because that code path is not the one being taken.

        The consequence was that separate writes to different fields of the
        same document erased each other. One document lost all four cascade
        fields when its deadline was written by a later, separate PATCH.

        So the merge happens on this side: read the current full list, drop the
        entries being overwritten, append the new values, send the complete
        result. `extra_payload` carries the ordinary fields - content, title,
        tags - which Paperless does merge correctly; the defect is specific to
        custom_fields.
        """
        current = self._transport.get(
            f"/api/documents/{doc_id}/", versioned=True
        ).json().get("custom_fields") or []

        overwritten = {value.field for value in values}
        merged = [entry for entry in current if entry["field"] not in overwritten]
        merged.extend({"field": value.field, "value": value.value} for value in values)

        payload = dict(extra_payload or {})
        payload["custom_fields"] = merged
        updated: dict[str, Any] = self._transport.patch(
            f"/api/documents/{doc_id}/", versioned=True, json=payload
        ).json()
        return updated
