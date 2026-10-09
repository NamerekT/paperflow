import logging
from collections.abc import Iterable
from typing import Any

from docflow.ports.taxonomy import TaxonomySnapshot
from paperflow.adapters.paperless.transport import PaperlessTransport
from paperflow.domain.models import TagRef
from paperflow.domain.tag_colors import tag_color_for_name
from paperflow.domain.tags import is_technical_tag
from paperflow.errors import PaperlessError, TaxonomyUnavailable

logger = logging.getLogger(__name__)

Resource = str  # "tags" | "correspondents" | "document_types"


class PaperlessTaxonomy:
    """Reads and creates the entities documents are filed under.

    Implements `docflow.ports.taxonomy.TaxonomySource`.
    """

    def __init__(self, transport: PaperlessTransport) -> None:
        self._transport = transport

    def list_names(self, resource: Resource) -> tuple[str, ...]:
        """Every name of `resource`, following pagination.

        Two things changed here, and both mattered.

        It now paginates. The previous version asked for a page size of 200 and
        stopped, so an archive with more than 200 tags silently showed the
        classifier a truncated list - and a classifier that cannot see an
        existing tag creates a near-duplicate beside it. With 172 tags already
        filed, that ceiling was close.

        And it no longer swallows failures. Returning an empty list on a
        network error made a blip indistinguishable from an empty archive,
        which produces the same duplicate-everything behaviour, only harder to
        explain afterwards.
        """
        try:
            return tuple(row["name"] for row in self._transport.paginate(f"/api/{resource}/"))
        except PaperlessError as exc:
            raise TaxonomyUnavailable(f"could not list {resource}: {exc}") from exc

    def snapshot(self) -> TaxonomySnapshot:
        """Everything the classifier needs to avoid inventing duplicates."""
        return TaxonomySnapshot(
            tags=tuple(name for name in self.list_names("tags") if not is_technical_tag(name)),
            correspondents=self.list_names("correspondents"),
            document_types=self.list_names("document_types"),
        )

    def list_tags_full(self) -> list[dict[str, Any]]:
        """Whole tag records - id, name, colour - rather than just names."""
        return list(self._transport.paginate("/api/tags/", versioned=False))

    def set_color(self, tag_id: int, color: str) -> None:
        """Used by the one-off `paperflow-tag-colors` recolour tool."""
        self._transport.patch(f"/api/tags/{tag_id}/", json={"color": color})

    def tags_by_id(self, tag_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
        wanted = sorted(set(tag_ids))
        if not wanted:
            return {}
        rows = self._transport.paginate("/api/tags/", params={"id__in": ",".join(map(str, wanted))})
        return {int(row["id"]): row for row in rows}

    def find_tag(self, name: str) -> dict[str, Any] | None:
        """The tag called `name`, ignoring case. Raises if Paperless cannot be asked."""
        results: list[dict[str, Any]] = (
            self._transport.get("/api/tags/", params={"name__iexact": name}, versioned=True)
            .json()
            .get("results", [])
        )
        return results[0] if results else None

    def ensure_tag(self, name: str) -> int:
        """Find or create a tag, raising on failure - for startup, not for documents."""
        found = self.find_tag(name)
        return int(found["id"]) if found else self._create("tags", name)

    def delete_tag(self, tag_id: int) -> None:
        self._transport.request("DELETE", f"/api/tags/{tag_id}/")

    def get_or_create_tag(self, name: str, *, new_parent_id: int | None) -> TagRef | None:
        """`get_or_create_id` for a tag, reporting where in the hierarchy it sits.

        A tag that has to be created is created under `new_parent_id`. An existing
        one is returned with whatever parent it already has, so the caller can tell
        a tag still awaiting review from an accepted one. None on the same terms as
        `get_or_create_id`.
        """
        if not name:
            return None

        try:
            found = self.find_tag(name)
            if found:
                parent = found.get("parent")
                return TagRef(id=int(found["id"]), parent_id=int(parent) if parent else None)
            return TagRef(
                id=self._create("tags", name, parent_id=new_parent_id), parent_id=new_parent_id
            )
        except PaperlessError as exc:
            logger.warning(f"could not resolve tag {name!r}: {exc}")
            return None

    def get_or_create_id(self, resource: Resource, name: str) -> int | None:
        """Find an entity by name, case-insensitively, creating it if absent.

        Returns None for an empty name or when creation failed. The failure is
        logged at warning: the caller drops the tag and files the document
        without it, which is the right trade - but it must be visible, since
        the result is a document that is subtly less findable than it should
        be.
        """
        if not name:
            return None

        try:
            found = self._transport.get(
                f"/api/{resource}/", params={"name__iexact": name}
            ).json()
            results = found.get("results", [])
            if results:
                return int(results[0]["id"])
            return self._create(resource, name)
        except PaperlessError as exc:
            logger.warning(f"could not resolve {resource} {name!r}: {exc}")
            return None

    def _create(self, resource: Resource, name: str, *, parent_id: int | None = None) -> int:
        payload: dict[str, Any] = {
            # 0 = Automatic. Documents uploaded by hand through the Paperless
            # UI keep getting metadata suggestions from its own classifier,
            # like the rest of the archive.
            #
            # A correspondent or document type passed explicitly at upload
            # bypasses that classifier entirely - Paperless returns early if
            # the value is already set. Tags have no such guard: setting them
            # always runs matching, so Paperless may add tags of its own
            # alongside ours. That is expected rather than a defect, and the
            # noise falls as the labelled archive grows.
            "matching_algorithm": 0,
            "name": name,
        }
        if resource == "tags":
            # Otherwise every new tag is the same default grey and none of them
            # can be told apart at a glance.
            payload["color"] = tag_color_for_name(name)
        if parent_id is not None:
            payload["parent"] = parent_id

        created = self._transport.post(f"/api/{resource}/", json=payload, versioned=True).json()
        logger.info(f"created {resource[:-1]}: {name}")
        return int(created["id"])
