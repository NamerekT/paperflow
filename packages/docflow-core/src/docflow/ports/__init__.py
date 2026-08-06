"""The Protocols the pipeline speaks to.

Everything outside the engine - the model gateway, the PDF library, the OCR
backend, the archive's taxonomy - is reached through a Protocol defined here.
The pipeline imports these and never an adapter, which is what allows the
engine to run against a different document manager, a different OCR backend
or, in a test, against a handful of fakes.
"""
