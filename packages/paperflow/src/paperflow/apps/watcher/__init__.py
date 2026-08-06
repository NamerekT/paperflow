"""The scanner-folder watcher: a dumb uploader, nothing more.

Watches for new PDFs, queues them, uploads each to Paperless with no metadata, keeps a
local backup, and shows a toast. It does not run the engine and does not know the
webhook exists - classification, OCR escalation and the deadline search all happen later,
asynchronously, once Paperless's own "Document Added" workflow calls the service. That
split is what keeps this process installable on Python 3.14 with nothing heavier than
`requests` and `watchdog`: the OCR/LLM half of the stack never has to run on this host.
"""
