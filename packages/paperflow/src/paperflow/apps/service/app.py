"""The FastAPI application. Entry point: `uvicorn paperflow.apps.service.app:app`.

uvicorn imports this module directly - there is no `main()` of ours in between - so the
one setup step that has to run before anything pulls in numpy or onnxruntime
(`paperflow.runtime.limit_math_threads`) happens right here, at module scope, before any
other import in this file. See that function's docstring for why it cannot simply live
inside the OCR engine module instead.
"""

from paperflow import runtime

runtime.limit_math_threads()

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import ValidationError

from paperflow import logging_setup
from paperflow.apps.service import composition
from paperflow.apps.service.context import ServiceContext
from paperflow.apps.service.schemas import parse_webhook_body
from paperflow.settings import Settings

_settings = Settings.load("service")
_service_settings = _settings.require_service()
logging_setup.configure_logging(_settings.log_level)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    runtime.install_excepthooks()
    logger.info(
        f"starting paperflow-ocr-cascade-webhook (apply_changes={_service_settings.apply_changes})"
    )

    context = composition.build(_settings)
    context.worker.start()

    app.state.context = context
    app.state.ready = True
    logger.info(f"ready for webhooks (container port {_service_settings.port})")
    yield
    logger.info("shutting down")


app = FastAPI(title="paperflow-ocr-cascade-webhook", lifespan=lifespan)


@app.post("/webhook/ocr-cascade", status_code=202)
async def receive_webhook(
    request: Request,
    x_webhook_token: str | None = Header(default=None, alias="X-Webhook-Token"),
) -> dict[str, Any]:
    if x_webhook_token != _service_settings.webhook_secret:
        logger.info("rejected webhook: missing or invalid X-Webhook-Token")
        raise HTTPException(status_code=401, detail="invalid or missing X-Webhook-Token")

    raw_body = await request.body()
    try:
        body = parse_webhook_body(raw_body)
    except (ValueError, ValidationError) as exc:
        logger.info(f"rejected webhook: invalid body ({exc}); body: {raw_body[:300]!r}")
        raise HTTPException(status_code=400, detail="missing or invalid 'document_id'") from exc

    context: ServiceContext = app.state.context
    queue_length = context.worker.enqueue(body.document_id, force=body.force)
    logger.info(
        f"webhook: doc_id={body.document_id} force={body.force} queued "
        f"(queue length: {queue_length})"
    )
    return {"status": "queued", "document_id": body.document_id}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ok"}
