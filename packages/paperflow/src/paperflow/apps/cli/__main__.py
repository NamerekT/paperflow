"""`python -m paperflow.apps.cli <tool> [args...]`.

Convenience for a checkout without the console scripts on PATH; each tool is equally
reachable directly (`paperflow-trigger`, `paperflow-duplicates`, `paperflow-tag-colors`,
`paperflow-rare-tags`, `paperflow-google-auth` - see `[project.scripts]`).
"""

import importlib
import sys

_TOOLS = {
    "trigger": "paperflow.apps.cli.trigger",
    "duplicates": "paperflow.apps.cli.duplicates",
    "tag-colors": "paperflow.apps.cli.tag_colors",
    "rare-tags": "paperflow.apps.cli.rare_tags",
    "google-auth": "paperflow.apps.cli.google_auth",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _TOOLS:
        names = ", ".join(sorted(_TOOLS))
        print(f"usage: python -m paperflow.apps.cli <{names}> [args...]", file=sys.stderr)
        raise SystemExit(2)

    tool = sys.argv[1]
    sys.argv = [f"paperflow-{tool}", *sys.argv[2:]]
    module = importlib.import_module(_TOOLS[tool])
    module.main()


if __name__ == "__main__":
    main()
