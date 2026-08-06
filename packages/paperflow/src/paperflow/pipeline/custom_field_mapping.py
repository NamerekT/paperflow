"""Turning a `CascadeResult` into the custom-field values Paperless stores.

Kept separate from `document_processor` because it is pure translation - domain values
in, `CustomFieldValue` objects out - with none of the surrounding I/O or error handling
that writing them actually involves.
"""

from docflow.domain.models import CascadeResult
from paperflow.adapters.paperless.custom_fields import PaperlessCustomFields
from paperflow.domain.models import CustomFieldSet, CustomFieldValue


def cascade_field_values(
    fields: CustomFieldSet,
    custom_fields: PaperlessCustomFields,
    result: CascadeResult,
    *,
    mark_done: bool,
) -> list[CustomFieldValue]:
    """The values for the four cascade fields, plus "Mixed Created by" when it applies."""
    values = [
        CustomFieldValue(
            field=fields.content_by.id,
            value=custom_fields.select_option_value(fields.content_by, result.content_by.value),
        ),
        CustomFieldValue(field=fields.avg_confidence.id, value=result.avg_confidence_pct),
        CustomFieldValue(field=fields.avg_accuracy.id, value=result.avg_accuracy_pct),
        CustomFieldValue(field=fields.cascade_flag.id, value=mark_done),
    ]
    if result.mixed_by is not None:
        values.append(CustomFieldValue(field=fields.mixed_created_by.id, value=result.mixed_by))
    return values
