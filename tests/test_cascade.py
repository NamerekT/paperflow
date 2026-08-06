"""The per-page OCR cascade: local recognition, escalated to a vision model where it is
not good enough, plus the "Content by"/"Mixed Created by"/confidence bookkeeping.

Exercised entirely against fakes - no PDF library, no model gateway - because the
decisions being pinned (when to escalate, how the three `content_by` labels are chosen,
what a rotation correction does to the page handle) are pure orchestration over the
ports and belong to this module alone.
"""

from docflow.domain.models import ContentSource, OcrLine
from docflow.domain.policies import EscalationPolicy, NativeTextConfig
from docflow.pipeline.cascade import OcrCascade

NATIVE_TEXT = NativeTextConfig(min_chars=40, max_image_coverage=0.5, min_page_ratio=0.8)
ESCALATION = EscalationPolicy(mean=0.95, p10=0.95)


class FakePage:
    def __init__(self, number, rotation=0):
        self.number = number
        self.rotation = rotation
        self.rotations_set: list[int] = []

    def set_rotation(self, degrees):
        self.rotation = degrees
        self.rotations_set.append(degrees)

    def text(self):
        return ""  # short -> never trusted as native text in these tests

    def image_coverage(self):
        return 0.0

    def render_jpeg_b64(self, *, dpi, quality):
        return f"jpeg-p{self.number}"

    def render_bgr(self, *, dpi):
        return f"bgr-p{self.number}"


class FakeDocument:
    def __init__(self, pages):
        self._pages = pages

    @property
    def page_count(self):
        return len(self._pages)

    def pages(self):
        return iter(self._pages)


class FakeOcr:
    """Returns preset lines per page number - {} means "no lines recognised"."""

    name = "RapidOCR"

    def __init__(self, lines_by_page):
        self._lines_by_page = lines_by_page
        self.calls = []

    def read_lines(self, page, *, dpi):
        self.calls.append((page.number, dpi))
        return self._lines_by_page.get(page.number, [])


class FakeVision:
    def __init__(self, label="Gemma 31B (LiteLLM: test)"):
        self.calls = []
        self._label = label

    def transcribe(self, image_b64):
        self.calls.append(image_b64)
        return f"vision text for {image_b64}", self._label


class FakeOrientation:
    def __init__(self, rotation_by_page=None):
        self._rotation_by_page = rotation_by_page or {}
        self.calls = []

    def detect(self, image_b64):
        self.calls.append(image_b64)
        return self._rotation_by_page.get(image_b64, 0)


def make_cascade(ocr, vision=None, orientation=None, *, escalation=ESCALATION):
    return OcrCascade(
        ocr,
        vision or FakeVision(),
        orientation or FakeOrientation(),
        render_dpi=300,
        jpeg_quality=85,
        escalation=escalation,
        native_text=NATIVE_TEXT,
    )


def good_lines(text="hallo welt"):
    return [OcrLine(text=text, score=0.99)]


def bad_lines(text="hallo welt"):
    return [OcrLine(text=text, score=0.5)]


def test_confident_page_is_kept_locally_and_not_escalated():
    doc = FakeDocument([FakePage(1)])
    ocr = FakeOcr({1: good_lines("hallo welt")})
    vision = FakeVision()

    result = make_cascade(ocr, vision).run(doc)

    assert result.content == "hallo welt"
    assert result.content_by == ContentSource.RAPIDOCR
    assert result.mixed_by is None
    assert vision.calls == []
    assert result.page_reports[0].source == "RapidOCR"


def test_low_confidence_page_is_escalated_to_vision():
    doc = FakeDocument([FakePage(1)])
    ocr = FakeOcr({1: bad_lines()})
    vision = FakeVision("Gemma 31B (LiteLLM: test)")

    result = make_cascade(ocr, vision).run(doc)

    assert vision.calls == ["jpeg-p1"]
    assert result.content == "vision text for jpeg-p1"
    assert result.content_by == ContentSource.VISION
    assert result.mixed_by is None
    assert result.page_reports[0].source == "Gemma 31B (LiteLLM: test)"


def test_page_with_no_recognised_lines_is_escalated():
    # NaN mean/p10 - EscalationPolicy treats "nothing recognised" as needing the vision
    # model rather than silently filing a blank page.
    doc = FakeDocument([FakePage(1)])
    ocr = FakeOcr({1: []})
    vision = FakeVision()

    result = make_cascade(ocr, vision).run(doc)

    assert vision.calls == ["jpeg-p1"]
    assert result.content_by == ContentSource.VISION


def test_mixed_document_reports_both_sources_per_page():
    doc = FakeDocument([FakePage(1), FakePage(2)])
    ocr = FakeOcr({1: good_lines(), 2: bad_lines()})
    vision = FakeVision("Gemma 31B (LiteLLM: test)")

    result = make_cascade(ocr, vision).run(doc)

    assert result.content_by == ContentSource.MIXED
    assert result.mixed_by == "p.1: RapidOCR\np.2: Gemma 31B (LiteLLM: test)"
    assert result.content == "hallo welt\n\n---\n\nvision text for jpeg-p2"


def test_rotation_is_applied_before_ocr_and_recorded_relative():
    page = FakePage(1, rotation=90)
    doc = FakeDocument([page])
    ocr = FakeOcr({1: good_lines()})
    orientation = FakeOrientation({"jpeg-p1": 180})

    result = make_cascade(ocr, orientation=orientation).run(doc)

    # Absolute /Rotate on the page handle is current (90) + detected (180) = 270.
    assert page.rotations_set == [270]
    # What the archive needs is the relative correction the model actually reported.
    assert result.page_rotations == {1: 180}


def test_no_rotation_detected_leaves_the_page_untouched():
    page = FakePage(1, rotation=0)
    doc = FakeDocument([page])
    ocr = FakeOcr({1: good_lines()})

    result = make_cascade(ocr).run(doc)

    assert page.rotations_set == []
    assert result.page_rotations == {}


def test_confidence_and_accuracy_are_averaged_as_percentages():
    doc = FakeDocument([FakePage(1), FakePage(2)])
    ocr = FakeOcr(
        {
            1: [OcrLine(text="a", score=1.0)],
            2: [OcrLine(text="b", score=1.0)],
        }
    )

    result = make_cascade(ocr).run(doc)

    assert result.avg_confidence_pct == 100.0
    assert result.avg_accuracy_pct == 100.0


def test_empty_document_produces_empty_result_with_no_averages():
    result = make_cascade(FakeOcr({})).run(FakeDocument([]))

    assert result.content == ""
    assert result.content_by == ContentSource.RAPIDOCR
    assert result.avg_confidence_pct is None
    assert result.avg_accuracy_pct is None
    assert result.page_reports == ()
