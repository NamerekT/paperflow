"""The engine's orchestration layer: pure logic over `docflow.ports`, nothing else.

Nothing here imports an adapter directly - only the Protocols in
`docflow.ports` and the `docflow.llm_tasks`, which are themselves built on
those Protocols. That is what makes `docflow.pipeline.document_workflow`
runnable against a real archive or against a handful of in-memory fakes with
no change to the code under test.
"""
