"""The webhook service Paperless calls after a document is added.

Asynchronous by design: queue the document id, answer 202, process it on a background
thread. Paperless's own timeout for a webhook action is not documented anywhere public,
and a page escalated to the vision model can take minutes across its retries - holding
the HTTP connection open for that long is not a risk worth taking when a queue and one
worker thread inside this same process do the job with no extra infrastructure.
"""
