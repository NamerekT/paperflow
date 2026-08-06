"""paperflow - the Paperless-ngx binding for the docflow engine.

Three applications live here:

  * `paperflow.apps.watcher` - a Windows daemon that watches the scanner
    output folder and uploads new PDFs to Paperless. It does nothing else;
    in particular it does not import docflow.
  * `paperflow.apps.service` - the HTTP service Paperless calls through a
    workflow webhook. It runs the docflow engine over the document and writes
    the result back.
  * `paperflow.apps.cli` - the scheduled catch-up job and the maintenance
    tools.

Everything Paperless-specific - the REST client, the custom-field schema, the
tag colour policy - lives in this package. The engine stays unaware of it.
"""

__version__ = "0.1.0"
