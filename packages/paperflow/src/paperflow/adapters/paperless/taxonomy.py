import logging
from typing import Any

from docflow.ports.taxonomy import TaxonomySnapshot
from paperflow.adapters.paperless.transport import PaperlessTransport
from paperflow.domain.tag_colors import tag_color_for_name
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
            tags=self.list_names("tags"),
            correspondents=self.list_names("correspondents"),
            document_types=self.list_names("document_types"),
        )

    def list_tags_full(self) -> list[dict[str, Any]]:
        """Whole tag records - id, name, colour - rather than just names."""
        return list(self._transport.paginate("/api/tags/", versioned=False))

    def set_color(self, tag_id: int, color: str) -> None:
        """Used by the one-off `paperflow-tag-colors` recolour tool."""
        self._transport.patch(f"/api/tags/{tag_id}/", json={"color": color})

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

    def _create(self, resource: Resource, name: str) -> int:
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

        created = self._transport.post(f"/api/{resource}/", json=payload).json()
        logger.info(f"created {resource[:-1]}: {name}")
        return int(created["id"])
