"""Turning one engine run into writes against a Paperless document.

Where `docflow.pipeline` decides what a document is, this decides what that means for
the archive: which custom-field values to send, what counts as "already done", and how
one document's outcome gets appended to the progress log both the live webhook and the
nightly catch-up job share.
"""
