"""Carrying out, on one document, what a human decided about its tags.

The classifier files every tag it invents under the review parent
(`paperflow.domain.constants.TAG_REVIEW_PARENT`). Reviewing happens in the Paperless UI
and is nothing but ordinary tag editing: move a tag out from under the parent to keep it,
or rename it to "→ Target" to fold it into Target. Neither edit touches the documents, so
Paperless calls back - one webhook per document still carrying the review parent - and
this module brings that document in line.

A write happens only when something actually changes. The webhook is fired by "document
updated" among other things, so an unconditional PATCH here would trigger itself forever.
"""

import logging
from typing import Any

from paperflow.adapters.paperless.documents import PaperlessDocuments
from paperflow.adapters.paperless.taxonomy import PaperlessTaxonomy
from paperflow.domain.tags import merge_target
from paperflow.errors import PaperlessError
from paperflow.logging_setup import DocumentLogAdapter

logger = logging.getLogger(__name__)


class TagReviewer:
    def __init__(
        self,
        documents: PaperlessDocuments,
        taxonomy: PaperlessTaxonomy,
        review_parent_id: int,
        *,
        apply_changes: bool,
    ) -> None:
        self._documents = documents
        self._taxonomy = taxonomy
        self._review_parent_id = review_parent_id
        self._apply_changes = apply_changes

    def review(self, doc_id: int) -> None:
        log = DocumentLogAdapter(logger, doc_id)
        current = [int(tag_id) for tag_id in self._documents.get(doc_id).get("tags") or []]
        tags = self._taxonomy.tags_by_id(current)

        wanted, merged = self._merge(current, tags, log)
        wanted = self._without_idle_review_parent(wanted, tags)
        if wanted == current:
            return

        if not self._apply_changes:
            log.info(f"apply_changes is off - would set tags {current} -> {wanted} (dry run)")
            return

        self._documents.patch(doc_id, {"tags": wanted})
        log.info(f"tags reviewed: {current} -> {wanted}")
        self._delete_unused(merged, log)

    def _merge(
        self, current: list[int], tags: dict[int, dict[str, Any]], log: DocumentLogAdapter
    ) -> tuple[list[int], list[int]]:
        """Swap each "→ Target" tag for Target; return the new ids and the ids swapped out.

        `tags` gains a record for every target, so the caller can see their parents.
        """
        wanted: list[int] = []
        merged: list[int] = []
        for tag_id in current:
            name = tags[tag_id]["name"] if tag_id in tags else ""
            target_name = merge_target(name)
            if target_name is None:
                wanted.append(tag_id)
                continue

            target = self._taxonomy.find_tag(target_name)
            if target is None:
                log.warning(f"tag {name!r} names no existing tag - leaving it in place")
                wanted.append(tag_id)
                continue

            target_id = int(target["id"])
            tags[target_id] = target
            wanted.append(target_id)
            merged.append(tag_id)
        return list(dict.fromkeys(wanted)), merged

    def _without_idle_review_parent(
        self, tag_ids: list[int], tags: dict[int, dict[str, Any]]
    ) -> list[int]:
        under_review = any(
            tag_id in tags and tags[tag_id].get("parent") == self._review_parent_id
            for tag_id in tag_ids
        )
        if under_review:
            return tag_ids
        return [tag_id for tag_id in tag_ids if tag_id != self._review_parent_id]

    def _delete_unused(self, merged: list[int], log: DocumentLogAdapter) -> None:
        # Its own narrow `try`: the document is already correct, and a merge tag that
        # outlives its last document is clutter, not a failure of this job.
        try:
            for tag_id, tag in self._taxonomy.tags_by_id(merged).items():
                if tag.get("document_count") == 0:
                    self._taxonomy.delete_tag(tag_id)
                    log.info(f"deleted the merged tag {tag['name']!r}: no documents left")
        except PaperlessError as exc:
            log.warning(f"could not clean up merged tags {merged}: {exc}")
