# tests/

Automated tests. Deliberately small: this is a smoke suite, not coverage.

Each test guards a piece of logic that is pure, non-obvious, and expensive to
get wrong in production:

| File | Guards |
|---|---|
| `test_pdf_unwrap.py` | Slicing a real PDF out of a PKCS#7-wrapped file |
| `test_webhook_body.py` | Paperless double-encoding its webhook JSON body |
| `test_i18n.py` | Locale catalogues staying in sync with the key enum |
| `test_llm_client.py` | Retry/error taxonomy of the model gateway client |
| `test_native_text.py` | Born-digital detection, against fake pages |
| `test_cascade.py` | The OCR escalation cascade, against fake pages and tasks |
| `test_fitz_document.py` | The same detection against real PyMuPDF - skipped unless `test/data/` is present locally |
| `test_document_processor.py` | The download/engine/resolve/write cycle, against fake Paperless adapters |
| `test_classify.py` | Dropping tags that only repeat the correspondent or the document type |
| `test_tag_review.py` | New tags going under the review parent, and `→ Target` merges, against fake adapters |
| `test_rare_tags.py` | Which tags the `paperflow-rare-tags` report includes |

Run with `pytest` from the repository root.

The diagnostic scripts under `experiments/` are **not** tests. They talk to
the live archive and to paid model endpoints, and none of them is run here.
