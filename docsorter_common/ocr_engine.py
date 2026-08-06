"""
ocr_engine.py — связка двух RapidOCR-движков (latin + eslav) для German/
Ukrainian документов, без сервера и без Docker.

Единая точка правды для DocSorter.py (плоский текст всего документа сразу)
и apply_ocr_cascade.py (постраничные (текст, score) для решения об
эскалации в Gemma).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import fitz  # PyMuPDF
import numpy as np

try:
    from rapidocr import RapidOCR, LangRec, OCRVersion, ModelType
except ImportError:
    from rapidocr_onnxruntime import RapidOCR
    LangRec = OCRVersion = ModelType = None   # старая версия пакета


class DualRapidOcrEngine:
    """Два прохода RapidOCR на страницу: "latin" (немецкий и большинство
    западноевропейских языков) и "eslav" (украинский/русский/белорусский).
    Один движок RapidOCR за раз читает только одну языковую группу, поэтому
    гоняем оба прохода и берём построчно тот, что увереннее (см.
    _merge_best_per_line) — склеивание текста напрямую подмешивало
    кириллические опечатки в правильный немецкий текст.
    """

    def __init__(self, render_dpi: int = 200, log_fn=print):
        self.render_dpi = render_dpi
        self.log = log_fn
        self._engine_latin = self._build_engine(LangRec.LATIN if LangRec else None, "latin (German)")
        self._engine_eslav = self._build_engine(LangRec.ESLAV if LangRec else None, "eslav (Ukrainian)")

    def _build_engine(self, lang_rec, label: str):
        """Собирает движок под конкретный язык. Нет отдельной "немецкой"
        модели — немецкий (и почти вся Западная Европа) покрывается моделью
        распознавания "latin" PP-OCRv5. Украинский лучше всего покрывает
        "eslav" (обучена именно на русском/белорусском/украинском) — точнее,
        чем более общий "cyrillic". Откатывается на дефолтную модель
        RapidOCR, если нужная комбинация недоступна (старая версия пакета,
        не скачалась модель) — вместо падения всего скрипта."""
        if LangRec is None:
            return RapidOCR()
        try:
            return RapidOCR(params={
                "Rec.lang_type": lang_rec,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE,
            })
        except Exception as e:
            self.log(f"  !!  Не удалось загрузить модель '{label}' ({e}) — использую дефолтную")
            return RapidOCR()

    @staticmethod
    def _ocr_lines(result) -> list[tuple[str, float]]:
        """[(text, score), ...] для одного прохода, учитывая разные формы
        ответа RapidOCR в разных версиях пакета. score по умолчанию 1.0,
        если версия его не отдаёт (сравнивать не с чем — считаем уверенным)."""
        if result is None:
            return []
        if isinstance(result, tuple):
            result = result[0]
        if hasattr(result, "txts"):
            txts = list(result.txts or [])
            scores = list(getattr(result, "scores", None) or [])
            if len(scores) != len(txts):
                scores = [1.0] * len(txts)
            return list(zip(txts, scores))
        if isinstance(result, list):
            return [(item[1], item[2] if len(item) > 2 else 1.0) for item in result]
        return []

    @staticmethod
    def _merge_best_per_line(latin_lines: list[tuple[str, float]],
                              eslav_lines: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Оба движка используют одну и ту же модель детекции текста, так что
        на нормальной странице находят одни и те же строки в том же порядке —
        отличается только распознавание. Берём построчно ту версию, где
        движок увереннее, вместо склейки обоих текстов подряд."""
        if latin_lines and eslav_lines and len(latin_lines) == len(eslav_lines):
            return [
                lt_ls if lt_ls[1] >= et_es[1] else et_es
                for lt_ls, et_es in zip(latin_lines, eslav_lines)
            ]
        return latin_lines + eslav_lines

    def _page_to_bgr(self, page):
        zoom = self.render_dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
        img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return img_rgb[:, :, ::-1]   # RapidOCR/OpenCV-style порядок каналов

    @staticmethod
    def _page_image_coverage(page) -> float:
        """Доля площади страницы, занятая встроенными картинками (0..1+).
        Скан — это буквально одна картинка на весь лист (~1.0), ДАЖЕ ЕСЛИ
        поверх неё заведён невидимый текстовый слой (т.н. "searchable PDF" —
        именно так выглядела "EK Valentyna (Printed to PDF)": на странице 1
        текст есть, но сама страница на 100% состоит из картинки под ним).
        У подлинно цифрового документа (Word/SAP/LaTeX-экспорт) картинок
        либо нет вовсе, либо только мелкие логотипы/подписи — покрытие
        низкое. Это надёжнее, чем смотреть только на наличие текста."""
        page_area = abs(page.rect)
        if not page_area:
            return 0.0
        total = 0.0
        for img in page.get_images(full=True):
            for rect in page.get_image_rects(img[0]):
                total += abs(rect)
        return total / page_area

    @staticmethod
    def extract_native_text(page, min_chars: int = 40, max_image_coverage: float = 0.5) -> str | None:
        """Если у PDF-страницы уже есть встроенный текстовый слой И страница
        не является по сути картинкой (см. _page_image_coverage) — это не
        скан бумаги, а программно сгенерированный/экспортированный PDF
        (например, официальное письмо, скачанное как PDF, а не отсканированное
        на принтере) — OCR ей вообще не нужен: текст уже точный на 100%,
        бесплатный и мгновенный. Прогонять такую страницу через RapidOCR или,
        тем более, Gemma — тратить время (и для Gemma — деньги/токены) ради
        текста ЗАВЕДОМО ХУЖЕ, чем уже есть в файле.

        Реальный случай, на котором поймана прошлая версия этой проверки
        (когда она смотрела только на наличие текста): скан, который потом
        кто-то "распечатал в PDF" (Print to PDF) — у страницы появился
        текстовый слой, но это оказался ЧУЖОЙ, встроенный откуда-то OCR, а
        не подлинный цифровой текст: "Valenign" вместо "Valentyna",
        "welteren" вместо "weiteren" — классические OCR-ошибки, включая
        испорченное имя. При этом картинка под текстом покрывала 100%
        страницы — что и выдало её как скан, несмотря на наличие текста.

        Возвращает текст, если он длиннее min_chars символов И картинки
        занимают не больше max_image_coverage площади страницы, иначе None
        — тогда страница действительно скан и её нужно распознавать как обычно."""
        text = page.get_text().strip()
        if len(text) < min_chars:
            return None
        if DualRapidOcrEngine._page_image_coverage(page) > max_image_coverage:
            return None
        return text

    @staticmethod
    def is_native_text_document(doc, min_chars: int = 40, min_page_ratio: float = 0.8) -> bool:
        """Второй, подстраховочный уровень поверх extract_native_text()'s
        картиночного чек-апа — решает НА УРОВНЕ ВСЕГО ДОКУМЕНТА, можно ли
        доверять текстовому слою PDF, а не только на уровне одной страницы.

        Картиночная проверка в extract_native_text() уже сама по себе ловит
        основной случай (searchable PDF — текст поверх картинки на весь
        лист), но не полагаемся ТОЛЬКО на неё: если какой-то другой
        инструмент когда-нибудь встроит невидимый OCR-слой без картинки на
        весь лист под ним (например, векторизованный скан) — эта проверка
        подстрахует, требуя, чтобы текст был последовательно почти на ВСЕХ
        страницах документа, что типично именно для born-digital документов
        (Word/SAP/LaTeX-экспорт), а не для скана, у которого страницы с
        текстом (если такие вообще есть) обычно единичны/случайны.

        Считаем документ born-digital, только если доля страниц с текстовым
        слоем >= min_page_ratio. Иначе весь документ считается сканом целиком
        (даже те отдельные страницы, где текст всё же нашёлся) — и идёт по
        обычному OCR-каскаду, как и должно быть для скана."""
        if doc.page_count == 0:
            return False
        with_text = sum(
            1 for page in doc
            if DualRapidOcrEngine.extract_native_text(page, min_chars) is not None
        )
        return (with_text / doc.page_count) >= min_page_ratio

    def ocr_page_lines(self, page) -> list[tuple[str, float]]:
        """(текст, score) на строку — то, что нужно apply_ocr_cascade.py
        для постраничного решения об эскалации в Gemma."""
        img_bgr = self._page_to_bgr(page)
        latin_lines = self._ocr_lines(self._engine_latin(img_bgr))
        eslav_lines = self._ocr_lines(self._engine_eslav(img_bgr))
        return self._merge_best_per_line(latin_lines, eslav_lines)

    def ocr_page_text(self, page) -> str:
        """Просто текст страницы (без score) — то, что нужно DocSorter.py."""
        return "\n".join(t for t, _ in self.ocr_page_lines(page))

    def extract_document_text(self, pdf_path: str) -> str:
        """Весь документ разом, плоским текстом (страницы разделены пустой
        строкой) — прямой аналог старого DocSorter.py::extract_text_rapidocr.
        Текстовый слой PDF (без OCR) используется, только если он есть почти
        на всех страницах документа — см. is_native_text_document()."""
        texts: list[str] = []
        doc = fitz.open(pdf_path)
        try:
            trust_native = self.is_native_text_document(doc)
            for page in doc:
                native_text = self.extract_native_text(page) if trust_native else None
                page_text = native_text or self.ocr_page_text(page)
                if page_text:
                    texts.append(page_text)
        finally:
            doc.close()
        return "\n\n".join(texts).strip()
