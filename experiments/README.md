# experiments/

Diagnostic and exploration scripts, not tests. Nothing here runs in CI, and none of it
is held to the rest of the repository's standards (see the `extend-exclude` in the root
`pyproject.toml`).

Each script was written to answer a specific question that came up while building the
OCR/classification pipeline, usually by comparing a few real documents against different
engines or prompts:

| File | What it explored |
|---|---|
| `docling_ocr_test.py` | Docling as an alternative local OCR backend |
| `gemini_vision_ocr_test.py` | Direct Gemini vision-model OCR, outside the LiteLLM proxy |
| `gemma_stream_diagnose.py` | Streaming a Gemma vision-OCR request token by token, to tell a hung network connection apart from a model stuck repeating itself |
| `local_vision_ocr_test.py` | A locally hosted vision model as an OCR backend |
| `rapidocr_compare_test.py` | RapidOCR variants/settings against each other |
| `rapidocr_confidence_calibration.py` | Whether RapidOCR's confidence score can be trusted as an escalation signal - see `docflow.domain.policies.EscalationPolicy` for where that investigation ended up |
| `vision_ocr_test.py` | Vision-model OCR quality across sample documents |

They talk to the live Paperless archive and to paid model endpoints, and write their
output to `data/` and `results/` (both gitignored - real documents and generated
artifacts, not source). Run them individually, by hand, when re-investigating something
in this space; do not expect them to still work unmodified as the surrounding code
changes underneath them.
