"""PyMuPDF code goes through `pymupdf`, not the deprecated `fitz` alias (PPF-14).

`fitz` warns on import that it will be removed; a module that still imports it would
break on the PyMuPDF release that drops it.
"""

import subprocess
import sys

import pytest

MODULES = [
    "docflow.adapters.pdf.fitz_document",
    "paperflow.apps.cli.duplicates",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_does_not_import_the_fitz_alias(module: str) -> None:
    code = f"import sys, {module}; print('fitz' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
