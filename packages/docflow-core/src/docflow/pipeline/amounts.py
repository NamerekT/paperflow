"""Amounts that lost their decimal comma in a form's cells.

RapidOCR reads "100,00" in a form cell as "10000" and scores it 1.0 (PPF-18), so a page
full of such cells keeps mean and p10 near 1.0. A bare whole number alone says nothing -
postcodes, customer and article numbers look the same - but sitting in a table row next to
an amount written with a comma, it usually is an amount too. On the archive this fired with
two or more such numbers only on the form it was found on; invoices with an article number
in the row have one.
"""

import re
from collections.abc import Sequence

from docflow.domain.models import OcrLine

_AMOUNT = re.compile(r"(?:(?:€|EUR)\s*)?[-+]?\d{1,3}(?:\.\d{3})*,\d{2}(?:\s*(?:€|EUR))?")

#: Four to seven digits: three would also catch every house number and code, and eight or
#: more are account, document and EAN numbers rather than amounts.
_BARE_NUMBER = re.compile(r"\d{4,7}")
_YEAR = re.compile(r"(?:19|20)\d\d")

#: Two boxes are in one row when they overlap vertically by at least this share of the
#: smaller one's height.
_SAME_ROW_OVERLAP = 0.5


def _is_bare_number(text: str) -> bool:
    return bool(_BARE_NUMBER.fullmatch(text)) and not _YEAR.fullmatch(text)


def _same_row(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return overlap >= _SAME_ROW_OVERLAP * min(a[3] - a[1], b[3] - b[1])


def bare_numbers_beside_amounts(lines: Sequence[OcrLine]) -> int:
    """How many lines are a bare whole number in the same row as an amount with a comma."""
    amounts = [line.box for line in lines if line.box and _AMOUNT.fullmatch(line.text.strip())]
    return sum(
        1
        for line in lines
        if line.box
        and _is_bare_number(line.text.strip())
        and any(_same_row(line.box, amount) for amount in amounts)
    )
