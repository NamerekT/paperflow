"""Identifiers that are already persisted in the archive.

Every literal in this module names something that exists in the live
Paperless instance - a custom field, a tag, a correspondent - and is written
into documents. None of them can be renamed by editing this file alone:
Paperless matches entities by name, so a new spelling creates a second entity
beside the old one and leaves every document processed so far pointing at the
original. Changing one of these is a data migration, not a rename.

Deliberately import-free so that anything can depend on it.
"""

# ── Custom fields written by the OCR cascade ──────────────────────────────

FIELD_CONTENT_BY = "Content by"
FIELD_MIXED_CREATED_BY = "Mixed Created by"
FIELD_AVG_CONFIDENCE = "Avg. confidence"
FIELD_AVG_ACCURACY = "Avg. accuracy"

#: Boolean marking a document as fully processed. This is the single source of
#: truth for "already done": the webhook service checks it on arrival, and the
#: nightly catch-up job selects documents that lack it. There is no local
#: progress file to disagree with it.
FIELD_CASCADE_FLAG = "Updated by OCR Cascade"

#: The deadline extracted from the document text. Created by hand in the
#: Paperless UI rather than by this application - see the spec table.
FIELD_FRIST = "Frist"


# ── Fallback entity names ─────────────────────────────────────────────────
#
# Used when the model cannot determine a value. These are German because the
# archive is German - they sit alongside Frist, Bescheid and Widerspruch - and
# they are real correspondent, document-type and tag names in Paperless, not
# display strings. Translating them would create an English duplicate of each
# and orphan every document already filed under the German one.

FALLBACK_CORRESPONDENT = "Unbekannt"
FALLBACK_DOCUMENT_TYPE = "Sonstiges"
FALLBACK_TAG = "Unsortiert"

#: Applied to every document, without exception.
TAG_PERSONAL = "Personal"

#: The family-member tag for shared or unattributable documents: rent,
#: Kfz-Schein and the like. Must be present in the family member list.
FAMILY_FALLBACK = "Familie"


# ── Tag review ────────────────────────────────────────────────────────────

#: Parent of every tag the classifier invented, until a human has looked at it:
#: moved elsewhere to keep it, or renamed with `MERGE_PREFIX` to fold it into
#: an existing one. Documents carrying such a tag carry this one too.
TAG_REVIEW_PARENT = "🆕 Neue Tags"

#: A tag renamed to "→ Target" is an instruction to replace it with Target on
#: every document, carried out by the tag-review webhook.
MERGE_PREFIX = "→"

#: Part of the name of tags that drive Paperless workflows rather than describe
#: documents. Never offered to the classifier.
TECHNICAL_TAG_MARKER = "Technical tag"


# ── Page attribution ──────────────────────────────────────────────────────

#: Prefix in the "Mixed Created by" field value, one entry per page.
PAGE_LABEL = "p."
