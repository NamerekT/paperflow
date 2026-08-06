"""docflow - a document normalisation and classification engine.

Given a PDF, it decides whether the file is born-digital or a scan, corrects
page orientation, recognises text, escalates pages the local OCR is not
confident about to a vision LLM, classifies the resulting text and extracts a
deadline from it.

It knows nothing about where the document came from or where the result goes.
Everything outside the engine is reached through a Protocol in `docflow.ports`,
so binding it to a document management system means writing adapters, not
editing the pipeline. `paperflow` is the Paperless-ngx binding.

Layers, strictly one-directional:

    domain  <-  ports  <-  pipeline
       ^          ^
       +-- adapters (implement ports; nothing here imports them except the
           composition root of the application using this package)

`domain` is pure data and pure functions. `pipeline` speaks only in terms of
ports - it never imports `requests`, `fitz` or `rapidocr`, which is what makes
it exercisable with fakes.
"""

__version__ = "0.1.0"
