"""An append-only JSONL log of what happened to each processed document.

Not a source of truth for anything - the "already done" question is answered by the
`Updated by OCR Cascade` field on the document itself, see
`paperflow.domain.constants.FIELD_CASCADE_FLAG`. This is only a human-readable record for
whoever is watching the container's `/data` volume, one line per document processed.
"""

import json
from pathlib import Path
from typing import Any


def append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
