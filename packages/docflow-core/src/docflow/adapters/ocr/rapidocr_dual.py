"""Two RapidOCR passes per page - one per language group - merged line by line.

One RapidOCR engine reads only one language group at a time, so German and
Ukrainian/Russian/Belarusian documents each need their own pass: "latin" for
German and most of Western Europe, "eslav" for Ukrainian (trained specifically
on Russian/Belarusian/Ukrainian - closer to this archive than the more general
"cyrillic"). Concatenating the two passes' text would mix Cyrillic
misreadings into otherwise correct German text, so instead the merge below
picks, line by line, whichever pass the engine itself was more confident
about - see `_merge_best_per_line`.

Every detected line is returned, however low its score: dropping weak lines is the
cascade's decision (`EscalationPolicy.line_score`), so that a lost line can count
against the page instead of vanishing inside the engine.
"""

import logging
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any

import numpy as np

# Deliberately not setting OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS here:
# they must exist before numpy or onnxruntime are first imported, which happens a few
# lines below as a side effect of importing rapidocr. A library module is the wrong
# place to decide that for the whole process - the application does, once, before this
# module is ever imported. See `paperflow.runtime.limit_math_threads`.
try:
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
    from rapidocr.utils.process_img import get_rotate_crop_image
except ImportError:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore[no-redef]

    LangRec = OCRVersion = ModelType = None  # type: ignore[assignment, misc]  # older package
    get_rotate_crop_image = None  # type: ignore[assignment]

from docflow.domain.models import OcrLine
from docflow.ports.pdf import PageHandle

logger = logging.getLogger(__name__)

#: A line scored below this is recognised again from its crop, upright and turned by 180
#: degrees, with the angle classifier off. The classifier
#: (`ch_ppocr_mobile_v2.0_cls_mobile`) sometimes turns a straight line over, and the
#: recognition of the upside-down crop is garbage the engine then drops; it cannot simply
#: be switched off, because on a scan fed in upside down it is what turns the lines back.
#: Re-reading only the weak lines keeps both.
_REREAD_BELOW = 0.9

#: Two passes' lines are the same line when their bounding rectangles overlap this much
#: (intersection over union). Both passes run the same detection model on the same
#: image, so in practice their boxes are identical.
_SAME_LINE_IOU = 0.5


@dataclass(frozen=True, slots=True)
class _Line:
    text: str
    score: float
    box: np.ndarray | None


class RapidOcrDualEngine:
    """A `docflow.ports.ocr.PageOcr` running RapidOCR twice per page."""

    name = "RapidOCR"

    def __init__(self, *, latin: Any = None, eslav: Any = None) -> None:
        """`latin` and `eslav` are prebuilt engines - tests pass fakes; by default both
        are built here."""
        # mypy resolves LangRec to the try-branch import unconditionally, so it
        # considers the fallback (older rapidocr_onnxruntime, LangRec is None) an
        # unreachable branch - it is real at runtime, just not something mypy can see
        # through a try/except ImportError.
        if latin is None:
            latin = self._build_engine(
                LangRec.LATIN if LangRec else None,  # type: ignore[truthy-function]
                "latin (German)",
            )
        if eslav is None:
            eslav = self._build_engine(
                LangRec.ESLAV if LangRec else None,  # type: ignore[truthy-function]
                "eslav (Ukrainian)",
            )
        self._engine_latin = latin
        self._engine_eslav = eslav

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
    def _ocr_lines(result: object) -> list[_Line]:
        """The lines of one pass, allowing for RapidOCR's different reply shapes across
        package versions. Score defaults to 1.0 when a version does not report one -
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
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) != len(txts):
                boxes = [None] * len(txts)
            return [
                _Line(text, float(score), box)
                for text, score, box in zip(txts, scores, boxes, strict=True)
            ]
        if isinstance(result, list):
            return [
                _Line(item[1], float(item[2]) if len(item) > 2 else 1.0, np.asarray(item[0]))
                for item in result
            ]
        return []

    @staticmethod
    def _read_pass(engine: Any, img: np.ndarray) -> list[_Line]:
        """One full pass over the page. RapidOCR keeps call arguments as engine state, so
        every call names all of them - a crop read leaves det and cls switched off."""
        result = engine(img, use_det=True, use_cls=True, use_rec=True, text_score=0.0)
        return RapidOcrDualEngine._ocr_lines(result)

    def _reread_if_weak(self, img: np.ndarray, line: _Line) -> _Line:
        """Both engines, not only the pass the line came from: on an upside-down Cyrillic
        line the classifier may leave the crop as it is, eslav then reads it as empty -
        which RapidOCR drops - and the only reading left is latin's garbage."""
        if line.score >= _REREAD_BELOW or line.box is None or get_rotate_crop_image is None:
            return line
        crop = get_rotate_crop_image(img, np.asarray(line.box, dtype=np.float32))
        best = line
        for engine in (self._engine_latin, self._engine_eslav):
            for candidate in (crop, np.rot90(crop, 2)):
                result = engine(
                    np.ascontiguousarray(candidate),
                    use_det=False,
                    use_cls=False,
                    use_rec=True,
                    text_score=0.0,
                )
                txts = getattr(result, "txts", None) or ()
                scores = getattr(result, "scores", None) or ()
                if txts and scores and txts[0].strip() and scores[0] > best.score:
                    best = _Line(txts[0], float(scores[0]), line.box)
        return best

    @staticmethod
    def _iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        a_min, a_max = np.min(a, axis=0), np.max(a, axis=0)
        b_min, b_max = np.min(b, axis=0), np.max(b, axis=0)
        overlap = np.clip(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0, None)
        inter = float(overlap[0] * overlap[1])
        union = float(np.prod(a_max - a_min) + np.prod(b_max - b_min)) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _merge_best_per_line(latin: list[_Line], eslav: list[_Line]) -> list[_Line]:
        """Pairs the two passes' lines by where they are on the page and keeps, per pair,
        the reading the engine was more confident about, instead of concatenating both
        texts one after the other. A line only one pass returned (the other read it as
        empty) is kept once, in its place in the reading order."""
        match: dict[int, int] = {}
        taken: set[int] = set()
        for i, lt in enumerate(latin):
            candidates = [
                (RapidOcrDualEngine._iou(lt.box, es.box), j)
                for j, es in enumerate(eslav)
                if j not in taken
            ]
            iou, j = max(candidates, default=(0.0, -1))
            if iou >= _SAME_LINE_IOU:
                match[i] = j
                taken.add(j)

        merged: list[_Line] = []
        latin_only: list[_Line] = []
        next_eslav = 0

        def flush_gap(stop: int) -> None:
            gap = latin_only + [eslav[k] for k in range(next_eslav, stop) if k not in taken]
            merged.extend(RapidOcrDualEngine._in_reading_order(gap))
            latin_only.clear()

        for i, lt in enumerate(latin):
            pair = match.get(i)
            if pair is None:
                latin_only.append(lt)
                continue
            flush_gap(pair)
            next_eslav = max(next_eslav, pair + 1)
            merged.append(lt if lt.score >= eslav[pair].score else eslav[pair])
        flush_gap(len(eslav))
        return merged

    @staticmethod
    def _in_reading_order(lines: list[_Line]) -> list[_Line]:
        """Top to bottom, and left to right within a row - the order RapidOCR's detector
        sorts its boxes in, with the same 10 px tolerance for what counts as one row."""
        if len(lines) < 2 or any(line.box is None for line in lines):
            return lines

        def compare(a: _Line, b: _Line) -> int:
            (ax, ay), (bx, by) = np.asarray(a.box)[0], np.asarray(b.box)[0]
            delta = ax - bx if abs(ay - by) < 10 else ay - by
            return int(np.sign(delta))

        return sorted(lines, key=cmp_to_key(compare))

    def read_lines(self, page: PageHandle, *, dpi: int) -> list[OcrLine]:
        img_bgr = page.render_bgr(dpi=dpi)
        latin_lines = self._read_pass(self._engine_latin, img_bgr)
        eslav_lines = self._read_pass(self._engine_eslav, img_bgr)
        merged = self._merge_best_per_line(latin_lines, eslav_lines)
        reread = [self._reread_if_weak(img_bgr, line) for line in merged]
        return [OcrLine(text=line.text, score=line.score) for line in reread]
