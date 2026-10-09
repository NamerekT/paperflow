"""Build step of service.Dockerfile: puts the RapidOCR recognition models into the image.

The rapidocr wheel ships only its default models; the latin and eslav ones are fetched
on first use. Its sessions are built lazily, so "first use" is the first document after
every deploy, not the service start - a download failure there costs a document.
Reading one rendered line here downloads both models into the image layer, and fails
the build if the OCR stack cannot read it.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from docflow.adapters.ocr.rapidocr_dual import RapidOcrDualEngine

SAMPLE_TEXT = "Rechnung Nr. 12345 vom 01.10.2026"


class _RenderedLine:
    number = 1

    def render_bgr(self, *, dpi: int) -> np.ndarray:
        image = Image.new("RGB", (900, 160), "white")
        ImageDraw.Draw(image).text(
            (20, 40), SAMPLE_TEXT, fill="black", font=ImageFont.load_default(size=44)
        )
        return np.array(image)[:, :, ::-1]


def main() -> None:
    lines = RapidOcrDualEngine().read_lines(_RenderedLine(), dpi=200)
    text = " ".join(line.text for line in lines)
    if "12345" not in text:
        raise SystemExit(f"OCR smoke test failed: read {text!r}")
    print(f"RapidOCR models are in place, read: {text!r}")


if __name__ == "__main__":
    main()
