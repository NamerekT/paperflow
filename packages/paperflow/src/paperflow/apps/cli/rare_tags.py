"""Report: the tags that only a handful of documents carry.

Read-only. A rare tag is not wrong in itself - a one-off topic deserves its one tag - but
most near-duplicates the classifier ever invented are in this list, so it is the place to
look for candidates to merge. Merging is done by hand in Paperless: rename the tag to
"→ Target" (see `paperflow.pipeline.tag_review`).
"""

import argparse
from typing import Any

from paperflow.adapters.paperless.documents import PaperlessDocuments
from paperflow.adapters.paperless.tasks import PaperlessTasks
from paperflow.adapters.paperless.taxonomy import PaperlessTaxonomy
from paperflow.adapters.paperless.transport import PaperlessTransport
from paperflow.apps.cli import console
from paperflow.domain.tags import is_technical_tag
from paperflow.i18n import t
from paperflow.i18n.keys import M
from paperflow.settings import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-documents",
        type=int,
        default=1,
        help="report tags carried by at most this many documents (default: 1)",
    )
    parser.add_argument("--locale", default=None, choices=["ru", "en"], help="output language")
    return parser


def select_rare(tags: list[dict[str, Any]], *, max_documents: int) -> list[dict[str, Any]]:
    rare = [
        tag
        for tag in tags
        if tag.get("document_count", 0) <= max_documents and not is_technical_tag(tag["name"])
    ]
    return sorted(rare, key=lambda tag: (tag.get("document_count", 0), tag["name"].casefold()))


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.load("cli")
    console.configure_console()
    console.apply_locale(args.locale, settings)

    transport = PaperlessTransport(settings.paperless)
    transport.detect_api_version()
    taxonomy = PaperlessTaxonomy(transport)
    documents = PaperlessDocuments(transport, PaperlessTasks(transport))

    tags = taxonomy.list_tags_full()
    rare = select_rare(tags, max_documents=args.max_documents)
    console.echo(
        t(M.RARETAGS_COUNT, rare=len(rare), total=len(tags), max_documents=args.max_documents)
    )

    for tag in rare:
        console.echo(
            t(
                M.RARETAGS_TAG_LINE,
                name=tag["name"],
                count=tag.get("document_count", 0),
                url=transport.tag_documents_url(tag["id"]),
            )
        )
        if not tag.get("document_count"):
            continue
        for document in documents.iter_summaries(
            ["id", "title"], filters={"tags__id__all": tag["id"]}
        ):
            console.echo(
                t(
                    M.RARETAGS_DOC_LINE,
                    doc_id=document.id,
                    title=document.title,
                    url=transport.document_url(document.id),
                )
            )


if __name__ == "__main__":
    main()
