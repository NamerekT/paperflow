"""The per-page OCR cascade: local recognition, escalated to a vision model where it is
not good enough, plus the confidence bookkeeping the archive stores against the
finished document.

This used to live only in a batch script; a live upload went through a plain local OCR
pass with no escalation and no confidence fields at all, so freshly uploaded documents
silently missed both the vision fallback and the "Content by"/"Avg. confidence"/"Avg.
accuracy" bookkeeping until a separate batch run caught up with them later. Both paths
now call the same `OcrCascade.run`, so the two cannot drift apart the way they did
before - it is the same code, not two copies kept in sync by hand.
"""

import logging

import numpy as np

from docflow.domain.models import CascadeResult, ContentSource, OcrLine, PageReport, PageStats
from docflow.domain.policies import EscalationPolicy, NativeTextConfig
from docflow.llm_tasks.vision_ocr import VisionOcrTask
from docflow.pipeline.native_text import extract_native_text, is_native_text_document
from docflow.ports.ocr import PageOcr
from docflow.ports.pdf import PdfDocument

logger = logging.getLogger(__name__)


def _page_stats(lines: list[OcrLine]) -> PageStats:
    if not lines:
        return PageStats(mean=float("nan"), p10=float("nan"), line_count=0)
    scores = np.array([line.score for line in lines], dtype=float)
    return PageStats(
        mean=float(scores.mean()), p10=float(np.percentile(scores, 10)), line_count=len(scores)
    )


class OcrCascade:
    """Runs one open document through native-text detection, local OCR and vision
    escalation. Holds only the rendering and escalation settings; the local OCR engine
    and the two model tasks are passed in, so a caller that already loaded the (heavy)
    OCR engine and built the (cheap) tasks shares them across documents instead of
    rebuilding either per call."""

    def __init__(
        self,
        ocr: PageOcr,
        vision: VisionOcrTask,
        *,
        render_dpi: int,
        jpeg_quality: int,
        escalation: EscalationPolicy,
        native_text: NativeTextConfig,
    ) -> None:
        self._ocr = ocr
        self._vision = vision
        self._render_dpi = render_dpi
        self._jpeg_quality = jpeg_quality
        self._escalation = escalation
        self._native_text = native_text

    def run(self, doc: PdfDocument) -> CascadeResult:
        page_texts: list[str] = []
        page_reports: list[PageReport] = []
        means: list[float] = []
        p10s: list[float] = []
        any_escalated = False
        any_local_kept = False

        # Looking at the WHOLE document rather than one page at a time is what stops a
        # foreign OCR layer, inherited on only part of a PDF (for example after a Print
        # to PDF over an already-recognised scan), from being mistaken for genuine
        # digital text - see docflow.pipeline.native_text.
        trust_native = is_native_text_document(doc, self._native_text)

        for page in doc.pages():
            native_text = extract_native_text(page, self._native_text) if trust_native else None
            if native_text is not None:
                logger.info(
                    f"p.{page.number}: found a PDF text layer "
                    f"({len(native_text)} chars) - OCR skipped"
                )
                page_texts.append(native_text)
                page_reports.append(
                    PageReport(page=page.number, source="Native PDF", mean=1.0, p10=1.0)
                )
                means.append(1.0)
                p10s.append(1.0)
                any_local_kept = True
                continue

            lines = self._ocr.read_lines(page, dpi=self._render_dpi)
            stats = _page_stats(lines)
            means.append(stats.mean if not stats.is_empty else 1.0)
            p10s.append(stats.p10 if not stats.is_empty else 1.0)

            if self._escalation.should_escalate(stats.mean, stats.p10):
                logger.info(
                    f"p.{page.number}: mean={stats.mean:.3f} p10={stats.p10:.3f} "
                    "-> escalating to the vision model"
                )
                image_b64 = page.render_jpeg_b64(dpi=self._render_dpi, quality=self._jpeg_quality)
                text, provider_label = self._vision.transcribe(image_b64)
                page_texts.append(text)
                page_reports.append(
                    PageReport(
                        page=page.number, source=provider_label, mean=stats.mean, p10=stats.p10
                    )
                )
                any_escalated = True
            else:
                text = "\n".join(line.text for line in lines)
                page_texts.append(text)
                page_reports.append(
                    PageReport(
                        page=page.number, source=self._ocr.name, mean=stats.mean, p10=stats.p10
                    )
                )
                any_local_kept = True

        return self._finish(
            page_texts, page_reports, means, p10s, any_escalated, any_local_kept
        )

    @staticmethod
    def _finish(
        page_texts: list[str],
        page_reports: list[PageReport],
        means: list[float],
        p10s: list[float],
        any_escalated: bool,
        any_local_kept: bool,
    ) -> CascadeResult:
        if any_escalated and any_local_kept:
            content_by = ContentSource.MIXED
            mixed_by: str | None = "\n".join(f"p.{r.page}: {r.source}" for r in page_reports)
        elif any_escalated:
            content_by = ContentSource.VISION
            mixed_by = None
        else:
            content_by = ContentSource.RAPIDOCR
            mixed_by = None

        return CascadeResult(
            content="\n\n---\n\n".join(page_texts),
            content_by=content_by,
            mixed_by=mixed_by,
            avg_confidence_pct=round(float(np.mean(means)) * 100, 1) if means else None,
            avg_accuracy_pct=round(float(np.mean(p10s)) * 100, 1) if p10s else None,
            page_reports=tuple(page_reports),
        )
