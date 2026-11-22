"""The dual RapidOCR engine: re-reading lines the angle classifier spoiled, and merging the
latin and eslav passes by where a line is on the page rather than by how many lines each
pass returned.

Runs against fake RapidOCR engines. The page image is a gradient that encodes position:
channel 0 is the x coordinate and channel 1 the y coordinate, so a fake can tell from a
crop's corner pixels which line it was given and whether the crop is upright or turned
by 180 degrees.
"""

from types import SimpleNamespace

import numpy as np

from docflow.adapters.ocr.rapidocr_dual import RapidOcrDualEngine

WIDTH, HEIGHT = 240, 200


def page_image():
    xs = np.tile(np.arange(WIDTH, dtype=np.uint8), (HEIGHT, 1))
    ys = np.tile(np.arange(HEIGHT, dtype=np.uint8)[:, None], (1, WIDTH))
    return np.dstack([xs, ys, np.zeros_like(xs)])


def box(top, left=20, right=220, height=20):
    bottom = top + height
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=float)


class FakePage:
    number = 1

    def render_bgr(self, *, dpi):
        return page_image()


class FakeRapidOcr:
    """`page` is what the full det+cls+rec pipeline returns: [(top, text, score)].
    `crops` is what recognition alone returns for a crop: {(top, upright): (text, score)}."""

    def __init__(self, page, crops=None):
        self._page = page
        self._crops = crops or {}
        self.calls = []

    def __call__(self, img, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("use_det"):
            return SimpleNamespace(
                boxes=np.array([box(top) for top, _, _ in self._page]),
                txts=tuple(text for _, text, _ in self._page),
                scores=tuple(score for _, _, score in self._page),
            )
        upright = int(img[0, 0, 0]) < int(img[0, -1, 0])
        top = int(img[0, 0, 1]) if upright else int(img[-1, -1, 1])
        text, score = self._crops.get((top, upright), ("", 0.0))
        return SimpleNamespace(txts=(text,), scores=(score,))

    @property
    def crop_calls(self):
        return [call for call in self.calls if not call.get("use_det")]


def engine(latin, eslav=None):
    return RapidOcrDualEngine(latin=latin, eslav=eslav or FakeRapidOcr([]))


def read(dual):
    return [(line.text, line.score) for line in dual.read_lines(FakePage(), dpi=300)]


def test_a_line_the_classifier_turned_over_is_read_again_upright():
    latin = FakeRapidOcr(
        page=[(10, "Erste Zeile", 0.99), (50, "ue6ue6 lsd", 0.37), (90, "Dritte Zeile", 0.99)],
        crops={(50, True): ("Das Ticket kann gedruckt werden", 0.98)},
    )

    assert read(engine(latin)) == [
        ("Erste Zeile", 0.99),
        ("Das Ticket kann gedruckt werden", 0.98),
        ("Dritte Zeile", 0.99),
    ]


def test_a_line_left_upside_down_is_read_again_turned_over():
    latin = FakeRapidOcr(
        page=[(50, "ue6ue6 lsd", 0.30)],
        crops={(50, True): ("ue6ue6 lsd", 0.30), (50, False): ("Zeile auf dem Kopf", 0.97)},
    )

    assert read(engine(latin)) == [("Zeile auf dem Kopf", 0.97)]


def test_a_weak_line_only_one_pass_returned_is_read_again_by_the_other_pass_too():
    """On an upside-down Cyrillic line the classifier may leave the crop as it is: the
    eslav pass then reads it as empty, which RapidOCR drops, and latin returns garbage."""
    latin = FakeRapidOcr(page=[(50, "e", 0.47)], crops={(50, False): ("з з 90", 0.80)})
    eslav = FakeRapidOcr(page=[], crops={(50, False): ("Дякуємо за вашу заяву", 0.96)})

    assert read(engine(latin, eslav)) == [("Дякуємо за вашу заяву", 0.96)]


def test_a_confident_line_is_not_read_again():
    latin = FakeRapidOcr(page=[(10, "Erste Zeile", 0.99)])

    read(engine(latin))

    assert latin.crop_calls == []


def test_a_line_no_orientation_reads_well_is_returned_with_its_low_score():
    latin = FakeRapidOcr(page=[(50, "~~", 0.20)], crops={(50, True): ("-", 0.10)})

    assert read(engine(latin)) == [("~~", 0.20)]


def test_every_call_sets_the_pipeline_flags_and_disables_the_engines_own_filter():
    """RapidOCR keeps call arguments as engine state, so a crop read with det and cls off
    would leave them off for the next page unless every call names them again."""
    latin = FakeRapidOcr(page=[(50, "x", 0.30)])
    dual = engine(latin)

    read(dual)
    read(dual)

    page_calls = [call for call in latin.calls if call.get("use_det")]
    assert len(page_calls) == 2
    assert all(call["use_cls"] and call["text_score"] == 0.0 for call in page_calls)
    assert all(not call["use_cls"] and call["text_score"] == 0.0 for call in latin.crop_calls)


def test_passes_are_merged_by_line_position_and_the_more_confident_reading_wins():
    latin = FakeRapidOcr(page=[(10, "Rechnung", 0.99), (50, "Pewehko", 0.60)])
    eslav = FakeRapidOcr(page=[(10, "Pechhyhg", 0.70), (50, "Решенко", 0.97)])

    assert read(engine(latin, eslav)) == [("Rechnung", 0.99), ("Решенко", 0.97)]


def test_lines_only_one_pass_returned_appear_once_and_in_page_order():
    latin = FakeRapidOcr(page=[(10, "eins", 0.99), (90, "drei", 0.99)])
    eslav = FakeRapidOcr(page=[(10, "eihs", 0.80), (50, "zwei", 0.99), (130, "vier", 0.99)])

    assert read(engine(latin, eslav)) == [
        ("eins", 0.99),
        ("zwei", 0.99),
        ("drei", 0.99),
        ("vier", 0.99),
    ]
