# Build from the repository ROOT as the build context (needs access to
# packages/docflow-core and packages/paperflow, which live there, not inside
# docker/):
#
#   docker build -f docker/service.Dockerfile -t ocr-cascade-webhook .
#
# (run from the repository root)

FROM python:3.11-slim

WORKDIR /app

# rapidocr depends on opencv-python (not the headless build), and that package needs
# several system shared libraries at import time (import cv2) that a minimal
# python:3.11-slim does not have (Debian slim is built without the graphics stack).
# Without them, "from rapidocr import RapidOCR, ..." fails with an ImportError
# ("libxcb.so.1: cannot open shared object file") before rapidocr gets to do anything -
# and the try/except around that import (docflow/adapters/ocr/rapidocr_dual.py) catches
# exactly ImportError and silently falls back to the older rapidocr_onnxruntime package,
# which is not installed either, so the real error (missing system libraries) shows up
# disguised as "ModuleNotFoundError: rapidocr_onnxruntime". Confirmed by reproducing it
# in a clean python:3.11-slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Both distributions, installed from the local source tree rather than PyPI - this image
# is the only consumer that needs them built, and there is nothing published to install
# from instead.
COPY packages/docflow-core/ ./packages/docflow-core/
COPY packages/paperflow/ ./packages/paperflow/
RUN pip install --no-cache-dir ./packages/docflow-core "./packages/paperflow[service]"

ENV PYTHONUNBUFFERED=1
# Explicit rather than derived from the package's own file location - a src/ layout
# would otherwise silently repoint them at /app/src/paperflow/... with no error anywhere.
ENV PAPERFLOW_CREDENTIALS_DIR=/app/credentials
ENV PAPERFLOW_FAMILY_MEMBERS_FILE=/app/family_members.json
# WEBHOOK_SHARED_SECRET, PAPERLESS_API_TOKEN, LLM_API_KEY and the rest of the secrets are
# required and have no default - passed through env_file/docker-compose at run time,
# never baked into the image.

# Neither credentials/ nor family_members.json is copied into the image - both hold
# real personal data (an OAuth token, a family's names) that has no business persisting
# in an image layer, so both are bind-mounted at run time instead:
#
#   - credentials/ (client_secret*.json and, after a one-off consent run on the Windows
#     host, token.json for Google Tasks - see paperflow.adapters.google.tasks): there is
#     no browser inside the container to complete the interactive consent, so this reads
#     the token generated on the host. The mount must NOT be read-only - GoogleTasks
#     writes the refreshed access token back to the same token.json roughly once an
#     hour, and a read-only mount would turn that into a silent PermissionError:
#     reminders would appear to work right up until the first token expiry and then stop
#     without any error visible outside the container log.
#   - family_members.json, read-only: this process only ever reads it.

EXPOSE 8000

# ${WEBHOOK_PORT:-8000} so the variable actually controls what uvicorn listens on, not
# just the EXPOSE line (which only documents a port - it opens nothing by itself).
CMD ["sh", "-c", "uvicorn paperflow.apps.service.app:app --host 0.0.0.0 --port ${WEBHOOK_PORT:-8000}"]
