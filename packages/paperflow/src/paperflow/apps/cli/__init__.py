"""One-shot operator tools: the nightly catch-up job, duplicate detection, tag
recolouring, and the Google Tasks one-off consent flow.

Each is runnable two ways: as its own console script (`paperflow-trigger`, and so on -
see `[project.scripts]` in this package's pyproject.toml) or as
`python -m paperflow.apps.cli <tool> ...` through this package's dispatcher. Output is
localised through `paperflow.i18n`; nothing else in this project is.
"""
