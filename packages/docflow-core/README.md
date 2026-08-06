# docflow-core

Document normalization, OCR-escalation and classification engine, independent of any document
management system. See the [repository root README](../../README.md) for the full picture of how
this fits together with [`paperflow`](../paperflow), the Paperless-ngx binding that consumes it.

## What it does

Given a PDF, `docflow.pipeline.document_workflow.DocumentWorkflow` decides whether it's
born-digital or a scan, runs OCR where needed, escalates poorly-recognized pages to a
vision-capable LLM, classifies the result (title, correspondent, document type, tags, date), and
extracts a deadline if the text names one. It returns a single typed `DocumentAnalysis` -
nothing is written back anywhere, and nothing beyond the LLM gateway and the file on disk is
touched.

Everything crosses the engine's boundary by **name** (`DocumentClassification.correspondent: str
| None`, never an id) - which document management system, if any, turns a name into an id is
entirely up to the caller. Implementing `docflow.ports.taxonomy.TaxonomySource` for a document
store other than Paperless-ngx is on the order of a few dozen lines.

## Installing

The base package is standard-library only:

```bash
pip install -e .
```

The pieces that perform I/O are behind extras:

```bash
pip install -e ".[llm]"   # the LLM gateway client and llm_tasks
pip install -e ".[ocr]"   # PDF rendering (PyMuPDF) + RapidOCR
```
