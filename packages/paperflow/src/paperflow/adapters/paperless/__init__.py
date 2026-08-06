"""The Paperless-ngx REST client, split by concern.

One transport and four repositories, rather than a single class with
twenty-five methods covering HTTP, taxonomy, schema management, document
CRUD, PDF byte surgery and task polling. Callers receive the repositories
they actually use, which is also what makes the pipeline exercisable with
small fakes instead of a whole archive.

Every hard-won workaround in here - the rewritten pagination links, the
client-side merge of custom fields, the lower-case task statuses - is
documented where it happens. They are the reason this is one client and not a
handful of requests calls spread across the applications.
"""

from paperflow.adapters.paperless.custom_fields import PaperlessCustomFields
from paperflow.adapters.paperless.documents import PaperlessDocuments
from paperflow.adapters.paperless.tasks import PaperlessTasks
from paperflow.adapters.paperless.taxonomy import PaperlessTaxonomy
from paperflow.adapters.paperless.transport import PaperlessTransport

__all__ = [
    "PaperlessCustomFields",
    "PaperlessDocuments",
    "PaperlessTasks",
    "PaperlessTaxonomy",
    "PaperlessTransport",
]
