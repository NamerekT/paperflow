r"""
set_tag_colors.py
────────────────────────────────────────────────────────────────────────────
Разово красит ВСЕ существующие теги в Paperless в яркие, различимые цвета
вместо дефолтного серого.

Цвет выбирается ДЕТЕРМИНИРОВАННО по имени тега (docsorter_common.paperless_client
.tag_color_for_name — та же функция, которую get_or_create_id() теперь
использует при создании НОВЫХ тегов "на лету" во время классификации). Так
покраска старых и будущих тегов не расходится.

ЗАПУСК:
    python set_tag_colors.py            # dry-run — только покажет, что покрасил бы
    python set_tag_colors.py --apply    # реально красит
"""

import argparse

from docsorter_common.paperless_client import PaperlessClient, tag_color_for_name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="реально красить (по умолчанию dry-run)")
    args = ap.parse_args()

    client = PaperlessClient(log_fn=print)
    client.detect_api_version()

    tags = client.list_tags_full()
    print(f"  ..  тегов в Paperless: {len(tags)}")

    changed = 0
    for tag in tags:
        color = tag_color_for_name(tag["name"])
        current = tag.get("color")
        if current == color:
            continue
        if not args.apply:
            print(f"  [DRY-RUN] {tag['name']!r}: {current} -> {color}")
            changed += 1
            continue
        client.patch_resource("tags", tag["id"], {"color": color})
        print(f"  OK  {tag['name']!r}: {current} -> {color}")
        changed += 1

    print(f"  ..  {'покрасил бы' if not args.apply else 'покрашено'}: {changed} из {len(tags)}")


if __name__ == "__main__":
    main()
