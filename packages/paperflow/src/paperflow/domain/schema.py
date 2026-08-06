"""The custom-field schema this application requires of Paperless.

Declarative on purpose: startup walks this list, creates what is missing and
checks the type of what already exists. `Frist` used to sit outside it as an
imperative special case in the service's own startup path, which meant one
field was verified by different code from the other five and could drift.
"""

from docflow.domain.models import ContentSource

from paperflow.domain import constants
from paperflow.domain.models import CustomFieldSpec

CUSTOM_FIELD_SPECS: tuple[CustomFieldSpec, ...] = (
    CustomFieldSpec(
        name=constants.FIELD_CONTENT_BY,
        data_type="select",
        # Derived from the enum rather than restated, because these labels are
        # what actually gets written into the field. Two hand-kept lists would
        # only have to agree; one list cannot disagree with itself.
        select_options=tuple(source.value for source in ContentSource),
    ),
    CustomFieldSpec(name=constants.FIELD_MIXED_CREATED_BY, data_type="longtext"),
    CustomFieldSpec(name=constants.FIELD_AVG_CONFIDENCE, data_type="float"),
    CustomFieldSpec(name=constants.FIELD_AVG_ACCURACY, data_type="float"),
    CustomFieldSpec(name=constants.FIELD_CASCADE_FLAG, data_type="boolean"),
    # Created by hand in the Paperless UI. Startup verifies its type but will
    # not create it: a date field that silently appeared with the wrong type
    # would be discovered only when a deadline failed to save.
    CustomFieldSpec(name=constants.FIELD_FRIST, data_type="date", externally_managed=True),
)

SPECS_BY_NAME = {spec.name: spec for spec in CUSTOM_FIELD_SPECS}
