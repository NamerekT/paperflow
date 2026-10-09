"""The composition root: settings in, a running `ServiceContext` out.

Everything else in this package takes what it needs as constructor arguments and builds
nothing itself - this is the one place that decides which concrete adapter answers each
port, so that decision exists exactly once.
"""

import json
import logging
from pathlib import Path

from docflow.adapters.llm.litellm_client import LiteLlmClient
from docflow.adapters.ocr.rapidocr_dual import RapidOcrDualEngine
from docflow.adapters.pdf.fitz_document import FitzOpener
from docflow.llm_tasks.classify import ClassifyTask
from docflow.llm_tasks.frist import FristTask
from docflow.llm_tasks.vision_ocr import VisionOcrTask
from docflow.pipeline.cascade import OcrCascade
from docflow.pipeline.document_workflow import DocumentWorkflow
from docflow.ports.taxonomy import ClassificationFallback, ClassifierPolicy
from paperflow.adapters.google.tasks import GoogleTasks
from paperflow.adapters.paperless.custom_fields import PaperlessCustomFields
from paperflow.adapters.paperless.documents import PaperlessDocuments
from paperflow.adapters.paperless.tasks import PaperlessTasks
from paperflow.adapters.paperless.taxonomy import PaperlessTaxonomy
from paperflow.adapters.paperless.transport import PaperlessTransport
from paperflow.adapters.startup import startup_step_with_retry
from paperflow.apps.service.context import ServiceContext
from paperflow.apps.service.worker import QueueWorker
from paperflow.domain import constants
from paperflow.domain.schema import CUSTOM_FIELD_SPECS
from paperflow.pipeline.document_processor import DocumentProcessor
from paperflow.pipeline.tag_review import TagReviewer
from paperflow.settings import Settings

logger = logging.getLogger(__name__)

#: Used only if family_members.json is missing or malformed - the process must not fail
#: outright over a typo in a hand-edited config file, only warn loudly and carry on.
_FAMILY_MEMBERS_FALLBACK = ("Familie",)


def _load_family_members(path: Path) -> tuple[str, ...]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        members = data["members"]
        if not isinstance(members, list) or not members:
            raise ValueError("'members' is empty or not a list")
        if "Familie" not in members:
            members = [*members, "Familie"]
        return tuple(members)
    except Exception as exc:
        logger.warning(
            f"could not read {path} ({exc}) - using the fallback list: "
            f"{list(_FAMILY_MEMBERS_FALLBACK)}"
        )
        return _FAMILY_MEMBERS_FALLBACK


def build(settings: Settings) -> ServiceContext:
    service_settings = settings.require_service()
    llm_settings = settings.require_llm()

    transport = PaperlessTransport(settings.paperless)
    startup_step_with_retry(
        "connecting to Paperless (detect_api_version)", transport.detect_api_version
    )

    tasks = PaperlessTasks(transport)
    documents = PaperlessDocuments(transport, tasks)
    custom_fields = PaperlessCustomFields(transport)
    taxonomy = PaperlessTaxonomy(transport)

    logger.info("checking/creating the cascade's custom fields")
    fields = startup_step_with_retry(
        "cascade custom fields", lambda: custom_fields.ensure(CUSTOM_FIELD_SPECS)
    )
    review_parent_id = startup_step_with_retry(
        "tag review parent", lambda: taxonomy.ensure_tag(constants.TAG_REVIEW_PARENT)
    )

    family_members = _load_family_members(settings.family_members_file)
    llm_client = LiteLlmClient(
        base_url=llm_settings.base_url,
        api_key=llm_settings.api_key,
        deadline_margin=llm_settings.deadline_margin_seconds,
    )

    classifier_policy = ClassifierPolicy(
        family_members=family_members,
        fallback=ClassificationFallback(
            correspondent=constants.FALLBACK_CORRESPONDENT,
            document_type=constants.FALLBACK_DOCUMENT_TYPE,
            tag=constants.FALLBACK_TAG,
            family_member=constants.FAMILY_FALLBACK,
        ),
        always_tags=(constants.TAG_PERSONAL,),
    )

    logger.info("loading the RapidOCR models (latin + eslav) - may take a while on first run")
    ocr_engine = RapidOcrDualEngine()
    vision = VisionOcrTask(llm_client, llm_settings.vision)
    classify = ClassifyTask(
        llm_client,
        llm_settings.classifier,
        classifier_policy,
        truncation_chars=llm_settings.text_truncation_chars,
    )
    frist = FristTask(
        llm_client, llm_settings.frist, truncation_chars=llm_settings.text_truncation_chars
    )

    cascade = OcrCascade(
        ocr_engine,
        vision,
        render_dpi=settings.cascade.render_dpi,
        jpeg_quality=settings.cascade.jpeg_quality,
        escalation=settings.cascade.escalation,
        native_text=settings.cascade.native_text,
    )
    workflow = DocumentWorkflow(FitzOpener(), cascade, classify, frist, taxonomy)

    google_tasks = GoogleTasks(settings.google) if settings.google.create_tasks else None

    processor = DocumentProcessor(
        workflow,
        documents,
        custom_fields,
        taxonomy,
        fields,
        google_tasks,
        apply_changes=service_settings.apply_changes,
        progress_path=service_settings.progress_path,
        review_parent_id=review_parent_id,
    )
    tag_reviewer = TagReviewer(
        documents, taxonomy, review_parent_id, apply_changes=service_settings.apply_changes
    )

    worker = QueueWorker(processor, tag_reviewer, service_settings)
    return ServiceContext(worker=worker)
