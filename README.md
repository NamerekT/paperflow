# Paperflow

An automated document pipeline for [Paperless-ngx](https://docs.paperless-ngx.com/): a scan, an
emailed PDF, or a manual upload gets OCR'd, classified by an LLM (title, sender, document type,
tags, whom it belongs to), and scanned for a deadline (due date, response window, payment date) -
with an optional reminder in Google Tasks. Everything lands in Paperless with metadata already
filled in; there is nothing left to file by hand.

This file is the onboarding doc for whoever - human or an AI coding agent - picks this project up
without prior context: what exists, why, how the pieces connect, and which mistakes have already
been made so they aren't repeated.

## Why

A household that receives a steady stream of official mail (government agencies, banks,
insurers, landlords) in more than one language ends up with either a shoebox of paper or a
half-searchable folder tree - neither scales, and neither is searchable by content. Paperless-ngx
solves storage and full-text search, but it does not by itself: (a) read mixed-language or
handwritten text on a mediocre scan reliably, (b) figure out what a letter *is* and who it is
from well enough to file it, or (c) notice that a letter has a deadline attached. This project is
the layer that does those three things.

## Two packages

The pipeline is split into a reusable engine and a Paperless-specific binding, on purpose: the
engine has zero knowledge of Paperless, so a different document store could reuse it without a
rewrite. The boundary is physical, not a convention someone could accidentally violate - `docflow`
does not depend on `paperflow`, so an accidental cross-import simply fails to install.

```
packages/
  docflow-core/   ← engine: open a PDF, OCR/escalate/classify/extract-deadline, no I/O beyond
                     the LLM gateway and the file itself
  paperflow/      ← Paperless-ngx binding + the applications (watcher, webhook service, CLI)
```

`docflow` returns everything by **name** (`DocumentClassification(title, correspondent, ...)`),
never by Paperless id - resolving a name to an id, and deciding what already exists vs. what
needs creating, is entirely `paperflow`'s job. That is the one decision that keeps the engine
portable: implementing `docflow.ports.taxonomy.TaxonomySource` for a different document manager
is a few dozen lines, not a rewrite.

```
src/docflow/
  domain/  ports/  adapters/{pdf,ocr,llm}/  llm_tasks/  pipeline/

src/paperflow/
  domain/  adapters/{paperless,google,windows}/  pipeline/  i18n/
  apps/  service/  watcher/  cli/
```

## How a document gets processed

```
scan / email / web upload
        │
        ▼
Paperless "Document Added" workflow (no source filter) ──► webhook
        │                                                     │
        ▼                                                     ▼
  (nightly safety net, in case the                    paperflow webhook service
   webhook was unreachable when the                    (FastAPI, one document at
   document was added)                                  a time - RapidOCR/the LLM
                                                          gateway are not built for
                                                          concurrent calls)
                                                                │
                                                                ▼
                                              download → OCR cascade → classify →
                                              PATCH back to Paperless → extract
                                              deadline → PATCH + optional reminder
```

Inside the OCR cascade, per page:

```
born-digital? (text layer present, and the page is not mostly a scanned image)
  ├─ yes ──► use the text layer as-is, no OCR
  └─ no  ──► RapidOCR (two language models) ──► good enough? ──► done
                                                       │
                                                       no
                                                       ▼
                                     escalate: render the page, ask a vision LLM
```

A single Docker service (`docker/service.Dockerfile`) is the only place this cascade runs. The
watcher and the nightly catch-up job are both thin: they only ever tell Paperless "a document
exists" - the same webhook handles a scan, an email import, and a manual upload identically.

### Tag review

Every tag the classifier invents is created as a child of the tag `🆕 Neue Tags`, and the
document gets that parent tag too, so new tags collect in one place instead of dissolving into
the list. Reviewing them is ordinary tag editing in the Paperless UI:

- **keep** a tag - move it out from under `🆕 Neue Tags`;
- **merge** a tag - rename it to `→ Target`, where Target is an existing tag. To merge a tag that
  is not under review, also move it under `🆕 Neue Tags`: Paperless then adds the parent to its
  documents, which is how the workflow below finds them.

Neither edit touches the documents. A second Paperless workflow does: for every document that
carries `🆕 Neue Tags` it calls `POST /webhook/tag-review` (triggers: Scheduled, daily, and
Document Updated; filter: has that tag; a webhook action with the same header and body as the
cascade's). For that document the service replaces each `→ Target` tag with Target, deletes a
merge tag once no document carries it, and removes `🆕 Neue Tags` when none of the document's
tags is under it any more. A `→ Target` tag whose target does not exist is left alone and logged.

Technical tags - the review parent, `→ ...` tags, and any tag with "Technical tag" in its name -
are never shown to the classifier. A tag that only repeats the correspondent or the document
type is dropped before the document is written.

## Applications

- **`paperflow.apps.watcher`** - runs continuously next to a physical scanner. Watches a folder,
  uploads new files to Paperless with no metadata (that arrives later, asynchronously, from the
  webhook service), keeps a local backup copy, and sends a desktop notification. Accepting a scan
  does not depend on Paperless being up: the file reaches the queue folder immediately, even with
  Docker off, and survives a reboot - the upload starts once the archive answers. Scans left in
  the watch folder by a previous session are picked up at startup.
- **`paperflow.apps.service`** - the FastAPI webhook service described above. The only
  containerized part of the system.
- **`paperflow.apps.cli`** - five small tools: a nightly catch-up trigger (`paperflow-trigger`,
  covers documents the webhook missed), a duplicate-document finder that compares OCR text and
  perceptual hashes rather than byte-identical checksums (`paperflow-duplicates`), a Paperless
  tag-color utility (`paperflow-tag-colors`), a read-only report of tags carried by at most N
  documents, with links to them - the candidates for a merge (`paperflow-rare-tags`), and the
  one-time Google OAuth consent flow (`paperflow-google-auth`).

CLI and watcher output is localized (`ru`/`en`, via `paperflow.i18n`); log records are always
English, enforced by a ruff rule that keeps `paperflow.i18n` out of library code.

## Configuration

Everything is read from the environment (frozen dataclasses, `Settings.load(profile)` for
`"watcher"` / `"service"` / `"cli"` - each profile only requires what it actually uses). Copy
`.env.example` to `.env` and fill it in; see that file for the full list of variables and what
each one does. No secret or hostname has a default in code - a missing one is a single
`ConfigError` naming every missing variable at once, not a 401 discovered mid-document.

`family_members.example.json` documents the one piece of local configuration outside environment
variables: a JSON list naming who documents in the archive get filed under.

## External services

Nothing here computes anything "in the cloud" on its own - it talks to a self-hosted Paperless-ngx
instance, a self-hosted LiteLLM proxy that fronts one or more model providers, and (for
reminders) Google Tasks. See `.env.example` for the shape of that configuration; there is no
hardcoded endpoint anywhere in `packages/`.

Independent LLM tasks run through the proxy with separate models and separate quota, so a
heavy day of OCR escalation cannot starve classification or vice versa: page-transcription
escalation uses a larger vision-capable model, while deadline extraction and classification
use faster, cheaper models. Which model handles which task is entirely a `.env` setting
(`VISION_PROXY_MODEL`, `FRIST_LLM_MODEL`, `CLASSIFIER_LLM_MODEL`).

## Development

```bash
pip install -e ./packages/docflow-core -e "./packages/paperflow[service,dev]"
ruff check .
mypy packages/docflow-core/src packages/paperflow/src
pytest
```

`experiments/` holds one-off diagnostic scripts (comparing OCR backends, calibrating escalation
thresholds, and so on) - not tests, not held to the rest of the repository's standards, and not
required to still run unmodified as the surrounding code changes. See `experiments/README.md`.

## Docker

The webhook service is the only containerized piece:

```bash
docker build -f docker/service.Dockerfile -t paperflow-service .
```

Real build gotcha, worth knowing before re-deriving it the hard way: `rapidocr` depends on
`opencv-python` (not the headless build), which needs system graphics libraries that a minimal
`python:3.11-slim` doesn't have. Without them, the import fails with `ImportError:
libxcb.so.1: cannot open shared object file` - and because the OCR adapter's import fallback
catches `ImportError` broadly, the real cause shows up disguised as an unrelated
`ModuleNotFoundError`. Fixed by installing `libgl1 libglib2.0-0 libsm6 libxext6 libxrender1`
before `pip install` (see the Dockerfile).

Neither `credentials/` (Google OAuth token) nor `family_members.json` is baked into the image -
both hold data that has no business persisting in an image layer, so both are bind-mounted at run
time instead. The credentials mount must **not** be read-only: the token is refreshed and
rewritten roughly once an hour, and a read-only mount turns that into a silent failure - reminders
keep working right up until the first token expiry and then stop without an error visible
anywhere but the refresh attempt itself.

## Rakes we've already stepped on

Short version here; full reasoning lives in the relevant module's comments/docstrings.

1. **A hung HTTP gateway looks identical to a slow one.** `requests`' `timeout=` only measures
   gaps *between* bytes, not total wall-clock time - a proxy that trickles keep-alive bytes without
   ever finishing the response sails straight through it. Fixed with a hard wall-clock deadline
   enforced from the client side via a background thread pool, independent of what the socket is
   doing.

2. **A vision LLM can loop on generation** instead of reaching a stop token - specifically on
   long, visually repetitive elements with no real text content (barcodes, QR codes, blank
   signature fields). At `temperature=0` there's no randomness to break the loop. The real fix
   isn't `max_tokens`/`frequency_penalty` (those only bound the damage) - it's an explicit prompt
   instruction not to transcribe such elements character by character, with a short placeholder
   marker instead, plus a rule against repeating the same character more than a few times.

3. **A PATCH of `custom_fields` on this Paperless server replaces the whole list rather than
   merging into it** - confirmed empirically, not by reading the server's source (which describes
   a different, safer default for most field types). A real document lost every field written by
   an earlier step when a later step's PATCH overwrote it. Fixed by always doing a GET → merge →
   PATCH cycle instead of a bare partial PATCH, in exactly one place.

4. **Some PDFs aren't PDFs at the first byte** - certain bank statements arrive wrapped in a
   PKCS#7/CMS signature container, with the real PDF starting at a nonzero offset. Detected and
   unwrapped by searching for `%PDF-`/`%%EOF` inside the file before handing it to Paperless.

5. **A full-page perceptual hash struggles to tell apart official letters from the same sender**
   that share a header/template but differ in content - naive transitive clustering ("A looks like
   B, B looks like C, so all three are one group") made this worse by merging unrelated letters
   together. Fixed by making OCR-text similarity (shingling + Jaccard) the primary duplicate
   signal, keeping the perceptual hash only as a fallback for near-textless documents (passport
   scans, photos), and reporting pairs with a similarity score instead of transitive groups.

6. **A Windows console can't encode non-ASCII output** by default - a bare `print()`/
   `logging.StreamHandler()` raises `UnicodeEncodeError` the moment localized text hits a
   terminal using a legacy code page. Fixed once, centrally, by reconfiguring stdout to UTF-8
   before any logging happens.

7. **An unhandled exception under `pythonw.exe` (no console, e.g. a Task Scheduler `LogonTrigger`)
   is completely invisible** - no traceback anywhere, just a silently dead process. Fixed with
   `sys.excepthook`/`threading.excepthook` installed at startup, so a crash is written to the log
   before the process dies, in both the watcher and the service's worker thread.

8. **A `LogonTrigger` scheduled task is not guaranteed to fire immediately at logon** - network
   services the task depends on may not be up yet either way. A short startup delay plus a
   frequent repetition combined with "ignore new instance if one is already running" makes the
   task self-healing: if the process is alive, repeated triggers are no-ops; if it died for any
   reason, the next repetition restarts it within minutes.

9. **A workflow webhook's JSON body can be double-encoded** - if the webhook action is configured
   with a template body, some Paperless-ngx versions render the template into a JSON string and
   then pass that string through as the value of a `json=` request parameter, which encodes it
   *again*. The body ends up being a JSON string containing JSON, not the object itself. Fixed by
   detecting that shape after the first decode and parsing again rather than depending on getting
   every configuration checkbox exactly right in a specific server version.

10. **The document-id placeholder in a webhook body template is not always the field name you'd
    guess** - and is only populated on a "document added" trigger, not on an earlier
    "consumption started" one where the document doesn't exist in the database yet. Check the
    actual template variables your Paperless-ngx version exposes rather than assuming.

11. **A Docker network alias defined on one network is not visible on another network the same
    container also sits on.** If a service resolves an internal hostname correctly from one
    container but gets a connection refused from another container on a different Docker network,
    check whether the alias was only declared under the first network's block.

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.
