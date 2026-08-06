"""One module per thing we ask a model to do.

Each is a prompt, a reply schema and a failure policy - nothing else. The
transport, the retries and the error classification all live in
`docflow.adapters.llm`, so these modules stay short enough to read alongside
the prompt they carry.
"""
