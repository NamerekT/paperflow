"""One-off: recolours every existing tag in Paperless to a bright, distinguishable
colour instead of the default grey.

The colour is chosen DETERMINISTICALLY from the tag's name
(`paperflow.domain.tag_colors.tag_color_for_name` - the same function
`PaperlessTaxonomy.get_or_create_id` uses when creating NEW tags on the fly during
classification), so recolouring old tags here can never disagree with what a new tag
gets at the moment it is created.
"""

import argparse

from paperflow.adapters.paperless.taxonomy import PaperlessTaxonomy
from paperflow.adapters.paperless.transport import PaperlessTransport
from paperflow.apps.cli import console
from paperflow.domain.tag_colors import tag_color_for_name
from paperflow.i18n import t
from paperflow.i18n.keys import M
from paperflow.settings import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="actually recolour (default: dry run)")
    parser.add_argument("--locale", default=None, choices=["ru", "en"], help="output language")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.load("cli")
    console.configure_console()
    console.apply_locale(args.locale, settings)

    transport = PaperlessTransport(settings.paperless)
    transport.detect_api_version()
    taxonomy = PaperlessTaxonomy(transport)

    tags = taxonomy.list_tags_full()
    console.echo(t(M.TAGCOLOR_COUNT, count=len(tags)))

    changed = 0
    for tag in tags:
        color = tag_color_for_name(tag["name"])
        current = tag.get("color")
        if current == color:
            continue
        if not args.apply:
            console.echo(t(M.TAGCOLOR_DRY_RUN_LINE, name=tag["name"], old=current, new=color))
            changed += 1
            continue
        taxonomy.set_color(tag["id"], color)
        console.echo(t(M.TAGCOLOR_APPLIED_LINE, name=tag["name"], old=current, new=color))
        changed += 1

    summary_key = M.TAGCOLOR_SUMMARY_APPLIED if args.apply else M.TAGCOLOR_SUMMARY_DRY_RUN
    console.echo(t(summary_key, changed=changed, total=len(tags)))


if __name__ == "__main__":
    main()
