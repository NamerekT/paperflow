# Build from the repository ROOT as the build context (needs access to
# packages/docflow-core and packages/paperflow, which live there, not inside
# docker/):
#
#   docker build -f docker/service.Dockerfile -t ocr-cascade-webhook .
#
# (run from the repository root)

# The base is pinned by digest and apt reads a dated snapshot.debian.org, so a rebuild
# gets the same Python and the same Debian packages as the deployed image (5a4f4f5):
# the digest alone is not enough - `apt-get install` against the live mirror also
# upgrades libraries already in the base (libexpat1 and the like).
#
# Moving the base forward (Debian point releases and deb13uN security fixes land here)
# is a deliberate change, done together:
#   1. docker pull python:3.11-slim, take its digest
#      (docker image inspect --format '{{index .RepoDigests 0}}' python:3.11-slim);
#   2. set DEBIAN_SNAPSHOT to a moment at or after that image's build time
#      (its /etc/apt/sources.list.d/debian.sources names the snapshot it was built on);
#   3. build with --no-cache, compare `python --version` and `dpkg-query -W` with the
#      deployed image, process a sample scan, deploy.
FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce

ARG DEBIAN_SNAPSHOT=20261010T000000Z

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
#
# The grep fails the build if a new base lists its mirrors differently and the sed missed
# one - apt would otherwise quietly install from the live mirror. A snapshot's Release
# files expire like the live ones do, hence Check-Valid-Until=false; the signatures are
# still checked.
RUN sed -i -E \
        "s#^URIs: http://deb.debian.org/(debian(-security)?)\$#URIs: https://snapshot.debian.org/archive/\1/${DEBIAN_SNAPSHOT}#" \
        /etc/apt/sources.list.d/debian.sources \
    && ! grep -q '^URIs:.*deb\.debian\.org' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Both distributions, installed from the local source tree rather than PyPI - this image
# is the only consumer that needs them built, and there is nothing published to install
# from instead. The constraints file pins every third-party version, so a rebuild for a
# one-line fix does not also upgrade the whole stack.
COPY docker/service-constraints.txt /tmp/service-constraints.txt
COPY packages/docflow-core/ ./packages/docflow-core/
COPY packages/paperflow/ ./packages/paperflow/
RUN pip install --no-cache-dir -c /tmp/service-constraints.txt \
        ./packages/docflow-core "./packages/paperflow[service]"

# The latin and eslav recognition models are not in the rapidocr wheel - without this
# step they are downloaded when the first document arrives after every deploy.
COPY docker/preload_ocr_models.py /tmp/preload_ocr_models.py
RUN python /tmp/preload_ocr_models.py && rm /tmp/preload_ocr_models.py

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
