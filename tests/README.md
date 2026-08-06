# tests/

Automated tests. Deliberately small: this is a smoke suite, not coverage.

Each test guards a piece of logic that is pure, non-obvious, and expensive to
get wrong in production:

| File | Guards |
|---|---|
| `test_pdf_unwrap.py` | Slicing a real PDF out of a PKCS#7-wrapped file |
| `test_webhook_body.py` | Paperless double-encoding its webhook JSON body |
| `test_i18n.py` | Locale catalogues staying in sync with the key enum |

Run with `pytest` from the repository root.

The diagnostic scripts under `experiments/` are **not** tests. They talk to
the live archive and to paid model endpoints, and none of them is run here.
