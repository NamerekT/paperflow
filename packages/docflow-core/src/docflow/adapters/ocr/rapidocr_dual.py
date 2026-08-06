"""Two RapidOCR passes per page - one per language group - merged line by line.

One RapidOCR engine reads only one language group at a time, so German and
Ukrainian/Russian/Belarusian documents each need their own pass: "latin" for
German and most of Western Europe, "eslav" for Ukrainian (trained specifically
on Russian/Belarusian/Ukrainian - closer to this archive than the more general
"cyrillic"). Concatenating the two passes' text would mix Cyrillic
misreadings into otherwise correct German text, so instead the merge below
picks, line by line, whichever pass the engine itself was more confident
about - see `_merge_best_per_line`.
"""

import logging

# Deliberately not setting OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS here:
# they must exist before numpy or onnxruntime are first imported, which happens a few
# lines below as a side effect of importing rapidocr. A library module is the wrong
# place to decide that for the whole process - the application does, once, before this
# module is ever imported. See `paperflow.runtime.limit_math_threads`.
try:
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
except ImportError:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore[no-redef]

    LangRec = OCRVersion = ModelType = None  # type: ignore[assignment, misc]  # older package

from docflow.domain.models import OcrLine
from docflow.ports.pdf import PageHandle

logger = logging.getLogger(__name__)


class RapidOcrDualEngine:
    """A `docflow.ports.ocr.PageOcr` running RapidOCR twice per page."""

    name = "RapidOCR"

    def __init__(self) -> None:
        # mypy resolves LangRec to the try-branch import unconditionally, so it
        # considers the fallback (older rapidocr_onnxruntime, LangRec is None) an
        # unreachable branch - it is real at runtime, just not something mypy can see
        # through a try/except ImportError.
        self._engine_latin = self._build_engine(
            LangRec.LATIN if LangRec else None,  # type: ignore[truthy-function]
            "latin (German)",
        )
        self._engine_eslav = self._build_engine(
            LangRec.ESLAV if LangRec else None,  # type: ignore[truthy-function]
            "eslav (Ukrainian)",
        )

    def _build_engine(self, lang_rec: object, label: str) -> RapidOCR:
        """Build the engine for one language group.

        There is no dedicated "German" model - German, and nearly all of
        Western Europe, is covered by the "latin" PP-OCRv5 recognition model.
        Ukrainian is best covered by "eslav" (trained specifically on
        Russian/Belarusian/Ukrainian) - more accurate than the more general
        "cyrillic". Falls back to RapidOCR's default model if the requested
        combination is unavailable (an older package version, or the model
        failed to download) rather than failing the whole run.
        """
        if LangRec is None:
            return RapidOCR()
        try:
            return RapidOCR(
                params={
                    "Rec.lang_type": lang_rec,
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.model_type": ModelType.MOBILE,
                }
            )
        except Exception:
            logger.warning(
                f"could not load the '{label}' model, falling back to the default", exc_info=True
            )
            return RapidOCR()

    @staticmethod
    def _ocr_lines(result: object) -> list[tuple[str, float]]:
        """[(text, score), ...] for one pass, allowing for RapidOCR's different reply shapes
        across package versions. Score defaults to 1.0 when a version does not report one -
        there is nothing to compare it against, so it is treated as confident."""
        if result is None:
            return []
        if isinstance(result, tuple):
            result = result[0]
        if hasattr(result, "txts"):
            txts = list(result.txts or [])
            scores = list(getattr(result, "scores", None) or [])
            if len(scores) != len(txts):
                scores = [1.0] * len(txts)
            return list(zip(txts, scores, strict=True))
        if isinstance(result, list):
            return [(item[1], item[2] if len(item) > 2 else 1.0) for item in result]
        return []

    @staticmethod
    def _merge_best_per_line(
        latin_lines: list[tuple[str, float]], eslav_lines: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Both engines use the same text-detection model, so on a normal page they find the
        same lines in the same order - only the recognition differs. Picks, line by line,
        whichever pass the engine was more confident about, instead of concatenating both
        texts one after the other."""
        if latin_lines and eslav_lines and len(latin_lines) == len(eslav_lines):
            return [
                lt_ls if lt_ls[1] >= et_es[1] else et_es
                for lt_ls, et_es in zip(latin_lines, eslav_lines, strict=True)
            ]
        return latin_lines + eslav_lines

    def read_lines(self, page: PageHandle, *, dpi: int) -> list[OcrLine]:
        img_bgr = page.render_bgr(dpi=dpi)
        latin_lines = self._ocr_lines(self._engine_latin(img_bgr))
        eslav_lines = self._ocr_lines(self._engine_eslav(img_bgr))
        merged = self._merge_best_per_line(latin_lines, eslav_lines)
        return [OcrLine(text=text, score=score) for text, score in merged]
