"""
ocr_cascade.py — постраничный OCR-каскад RapidOCR -> Gemma (vision-LLM) +
кастомные поля Paperless ("Content by" / "Avg. confidence" / "Avg.
accuracy" / "Updated by OCR Cascade").

Раньше вся эта логика жила ТОЛЬКО в apply_ocr_cascade.py — DocSorter.py
(боевой watcher, реальные загрузки "с колёс") звал только "голый" RapidOCR
без эскалации в Gemma на плохих страницах и вообще не трогал кастомные
поля. Из-за этого свежезагруженные через DocSorter документы:
  - не получали Gemma на страницах, где RapidOCR действительно плохо
    справился (то же самое, из-за чего изначально понадобился каскад —
    см. историю ESCALATE_MEAN/ESCALATE_P10 ниже);
  - оставались без "Content by"/"Avg. confidence"/"Avg. accuracy"/
    "Updated by OCR Cascade" насовсем, пока их не подбирал отдельным
    прогоном apply_ocr_cascade.py.

Теперь оба скрипта используют один и тот же класс OcrCascade — поведение
(пороги эскалации, промт для Gemma, расчёт confidence/accuracy, набор
кастомных полей) не может разойтись между "живой" загрузкой и пакетным
прогоном, потому что это буквально один и тот же код.
"""

import base64
import concurrent.futures
import time

import fitz  # PyMuPDF
import numpy as np
import requests

from . import config
from .http_utils import post_with_deadline
from .ocr_engine import DualRapidOcrEngine
from .orientation import detect_page_rotation
from .paperless_client import PaperlessClient

CONTENT_BY_RAPIDOCR = "RapidOCR"
CONTENT_BY_VISION   = "Vision-LLM (Gemma 31B)"
CONTENT_BY_MIXED    = "Mixed"

# Кастомное поле (boolean/чекбокс), которым помечается документ после
# обработки каскадом — источник правды для "уже сделано". apply_ocr_cascade.py
# пропускает документ, если оно уже true (если только не --force); теперь
# DocSorter.py тоже ставит его в true сразу после живой загрузки, так что
# batch-прогон каскада больше не будет зря переобрабатывать документы,
# которые уже прошли полный каскад "с колёс".
CASCADE_FLAG_FIELD_NAME = "Updated by OCR Cascade"

CUSTOM_FIELD_DEFS = {
    "Content by": {
        "data_type": "select",
        "extra_data": {"select_options": [
            {"label": CONTENT_BY_RAPIDOCR},
            {"label": CONTENT_BY_VISION},
            {"label": CONTENT_BY_MIXED},
        ]},
    },
    "Mixed Created by": {"data_type": "longtext", "extra_data": None},
    "Avg. confidence":  {"data_type": "float", "extra_data": None},
    "Avg. accuracy":    {"data_type": "float", "extra_data": None},
    CASCADE_FLAG_FIELD_NAME: {"data_type": "boolean", "extra_data": None},
}

# Два реальных случая одного и того же семейства (диагностика через
# test/gemma_stream_diagnose.py, август 2026):
#   1. стр.2 scan_0001.pdf — залипла на штрихкоде внизу страницы, пытаясь
#      побуквенно "прочитать" полосатый узор как текст ("||||||..." до
#      упора в max_tokens).
#   2. Schweigepflichtentbindung — залипла на ПУСТОЙ строке для подписи/
#      заполнения ("befreie hiermit Frau / Herrn ____________"), пытаясь
#      дорисовать черту символами "_" на всю ширину страницы.
# Общая причина: любой длинный визуально повторяющийся элемент БЕЗ
# собственного текстового содержимого (штрихкод, QR, пустая линия для
# подписи/заполнения, пунктирная/сплошная рамка таблицы) не имеет
# естественной "длины" — сколько символов ни генерируй, визуально это
# никогда не будет "готово", и без явного запрета модель способна
# генерировать такой символ до потолка токенов. max_tokens/frequency_penalty
# ниже — подстраховка на случай ещё не встреченного варианта той же
# болезни, но РЕАЛЬНЫЙ фикс — явно перечислить эту категорию целиком и
# сказать не транскрибировать её посимвольно. Это убирает причину, а не
# только обрезает симптом — точность настоящего текста не может пострадать
# от лимита токенов, если модель вообще не тратит их на подобные элементы.
PROMPT_VISION = (
    "Распознай ВЕСЬ текст на этом изображении документа — печатный и "
    "рукописный (документ может быть на украинском или немецком). Выведи "
    "результат простым текстом, сохраняя порядок блоков на странице и "
    "структуру таблиц (можно через табуляцию или ' | '). "
    "Ничего не придумывай и не пропускай — если что-то неразборчиво, "
    "отметь это как [нерозбірливо], но не оставляй пропуски молча и не "
    "подставляй правдоподобные, но выдуманные слова вместо реального текста. "
    "Штрихкоды, QR-коды, водяные знаки, логотипы и любые другие чисто "
    "графические элементы — это НЕ текст: не пытайся распознавать их "
    "посимвольно и не воспроизводи узор линий/точек как последовательность "
    "символов. Просто обозначь их одной короткой пометкой вроде [штрихкод] "
    "или [QR-код] и сразу переходи дальше. То же самое с ПУСТЫМИ полями для "
    "заполнения или подписи (длинная черта/линия/точки без текста на ней, "
    "например 'befreie hiermit Frau / Herrn ___________') — не рисуй черту "
    "на всю видимую длину, просто обозначь короткой пометкой вроде "
    "[пустое поле] или [пропуск для подписи] и переходи дальше. Общее "
    "правило: НИКОГДА не выводи один и тот же символ (в т.ч. '_', '-', '.', "
    "'|') подряд больше 3-5 раз, даже если на изображении этот символ/линия "
    "визуально длиннее — как только заметил, что вот-вот начнёшь повторять "
    "один и тот же символ или короткий фрагмент текста, останавливайся и "
    "переходи к следующему элементу страницы. "
    "Не добавляй собственных комментариев или пояснений — только "
    "распознанный текст."
)

# Постраничный порог эскалации в Gemma. Изначально поднят с 0.80 до 0.85
# после реального случая (doc 26, Ergänzung zum Diplom — таблица оценок):
# страницы с плотной мелкой таблицей (кириллица+латиница+цифры вперемешку,
# короткие ячейки — коды предметов, оценки вида "82/B/Добре/Good") дали
# mean=0.938-0.940, p10=0.834-0.835 — ВЫШЕ старого порога 0.80, при этом
# реальный текст был грубо перепутан ("82/B/ope/Good", "poepayeann a Pyihon"
# вместо "Програмування Python"). Классический случай "уверен, но неправ" —
# RapidOCR не мог служить сигналом качества именно на этом типе страниц.
# Дальше поднят до 0.90 по результатам реального прогона: на p10=0.85
# в Gemma уходило почти всё, кроме последних 2 страниц одного документа —
# и то, что RapidOCR выдал на этих 2 страницах, оказалось некачественным.
# То есть ниже 0.90 RapidOCR практически нигде не был по-настоящему
# надёжен — держать порог ниже смысла не было.
ESCALATE_MEAN = 0.95
ESCALATE_P10  = 0.95


def _page_stats(lines: list[tuple[str, float]]) -> tuple[float, float, int]:
    scores = np.array([s for _, s in lines], dtype=float)
    if len(scores) == 0:
        return (float("nan"), float("nan"), 0)
    return (float(scores.mean()), float(np.percentile(scores, 10)), len(scores))


def _render_page_jpeg_b64(page, dpi: int) -> str:
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
    return base64.b64encode(pix.tobytes("jpeg", jpg_quality=85)).decode("ascii")


def gemma_vision_ocr(image_b64: str, log_fn=print, timeout: int = 90, max_retries: int = 4) -> tuple[str, str]:
    """Эскалация плохой страницы в vision-LLM через собственный self-hosted
    LiteLLM-прокси (docker-compose в D:\\Projects\\YAML\\ollama-openwebui-compose,
    см. litellm-config/config.yaml) вместо прямого обращения к OpenRouter.

    Раньше при сбое OpenRouter (429/HTTP-ошибка/таймаут/зависание без
    ответа) фолбэк на прямой Google AI Studio был написан руками прямо
    здесь (две отдельные функции + оркестратор) — теперь этим занимается
    сам роутер LiteLLM: модель "openrouter-gemma-4-31b-it" при сбое сама
    переключается на "google-gemma-4-31b-it" (router_settings.fallbacks в
    litellm-config/config.yaml). Проверено вручную 2 августа 2026 — рабочий
    ответ содержал "provider":"Google AI Studio", т.е. фолбэк реально
    сработал сам, без единой строчки нашего кода.

    Здесь остаётся только то, что нужно НЕЗАВИСИМО от того, что происходит
    за прокси: retry на 429/5xx и жёсткий дедлайн на тот случай, если сам
    прокси зависнет так же, как раньше зависал прямой OpenRouter (см.
    http_utils.py) — самостоятельный self-hosted сервис не застрахован от
    того же класса проблем, что и любой другой HTTP-эндпоинт.

    Возвращает (text, provider_label) — provider_label берётся из поля
    "provider" в ответе LiteLLM (сам сообщает, кто в итоге ответил).

    max_tokens/frequency_penalty: вторая рабочая теория зависаний (не сеть/
    прокси, а сама генерация) — на плотном, визуально повторяющемся
    контенте (ряды чекбоксов, как на стр.2 scan_0001.pdf) vision-LLM может
    зациклиться на повторе одного и того же куска текста и никогда не
    дойти до стоп-токена. При temperature=0 (полностью жадный decoding)
    сбежать из такого локального повторяющегося цикла нечем — это
    известная болячка greedy decoding на затруднённом вводе. max_tokens
    ставит жёсткий потолок на длину генерации (раньше его не было вообще —
    зависание могло быть не "тишиной", а РЕАЛЬНОЙ бесконечной генерацией
    мусора). frequency_penalty снижает шанс залипания, не мешая точности
    OCR — drop_params: true в litellm-config/config.yaml гарантирует, что
    если конкретный провайдер этот параметр не поддерживает, он просто
    молча отбрасывается, а не роняет запрос ошибкой. См.
    test/gemma_stream_diagnose.py — стриминговый скрипт, который прямо
    показывает разницу между "тишина в сети" и "модель повторяет одно и то
    же"."""
    payload = {
        "model": config.VISION_PROXY_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_VISION},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 8192,
        "frequency_penalty": 0.3,
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = post_with_deadline(
                f"{config.VISION_PROXY_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.VISION_PROXY_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload, timeout=timeout,
            )
        except requests.exceptions.ReadTimeout:
            log_fn(f"       .. LiteLLM-прокси не ответил за {timeout}с, попытка {attempt}/{max_retries}")
            continue
        except concurrent.futures.TimeoutError:
            # requests' собственный timeout почему-то не сработал — тот же
            # класс зависания, что раньше ловили на прямом OpenRouter (см.
            # http_utils.py). Жёсткий дедлайн подстраховал и здесь.
            log_fn(f"       .. LiteLLM-прокси завис без ответа и без ошибки за "
                   f"{timeout}с — попытка {attempt}/{max_retries}")
            continue

        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 10 * attempt))
            log_fn(f"       .. 429 Too Many Requests (LiteLLM), жду {wait:.0f}с (попытка {attempt}/{max_retries}): {resp.text[:300]}")
            if attempt < max_retries:
                time.sleep(wait)
                continue
            raise RuntimeError("LiteLLM-прокси: rate limit после всех попыток (видимо, исчерпаны оба провайдера за прокси)")

        if resp.status_code >= 400:
            raise RuntimeError(f"LiteLLM-прокси: HTTP {resp.status_code} — {resp.text[:500]}")

        data = resp.json()
        if "choices" not in data:
            raise RuntimeError(f"Неожиданный ответ от LiteLLM-прокси: {data}")
        text = data["choices"][0]["message"]["content"].strip()
        provider = data.get("provider") or config.VISION_PROXY_MODEL
        return text, f"Gemma 31B (LiteLLM: {provider})"

    raise RuntimeError(f"LiteLLM-прокси: не удалось получить ответ за {max_retries} попыток")


class OcrCascade:
    """Постраничный каскад RapidOCR -> Gemma + сборка кастомных полей
    Paperless. Один инстанс держит только render_dpi/log_fn — RapidOCR
    движки (тяжёлые, грузятся один раз) передаются снаружи (DualRapidOcrEngine),
    чтобы DocSorter.py и apply_ocr_cascade.py продолжали использовать один и
    тот же уже загруженный движок, а не грузили его дважды."""

    def __init__(self, ocr_engine: DualRapidOcrEngine, render_dpi: int = 300, log_fn=print):
        self.ocr_engine = ocr_engine
        self.render_dpi = render_dpi
        self.log = log_fn

    def process_document(self, pdf_path) -> dict:
        """Возвращает {"content": str, "content_by": str, "mixed_by": str|None,
        "avg_confidence_pct": float, "avg_accuracy_pct": float,
        "page_reports": [...], "page_rotations": {номер_страницы: градусы}}."""
        doc = fitz.open(str(pdf_path))
        page_texts = []      # финальный текст по каждой странице (после каскада)
        page_reports = []    # для "Mixed Created by" и логов
        page_rotations = {}  # страницы, где обнаружен и исправлен поворот — для edit_pdf в Paperless
        means, p10s = [], []
        any_escalated = False
        any_rapidocr_kept = False

        try:
            # Смотрим на ВЕСЬ документ, а не на страницу по отдельности — см.
            # docsorter_common/ocr_engine.py::is_native_text_document(). Иначе
            # чужой (и не факт что качественный) OCR-слой, случайно
            # унаследованный только на части страниц PDF (например, после
            # Print to PDF поверх уже распознанного скана), молча выдавался бы
            # за подлинный цифровой текст.
            trust_native = self.ocr_engine.is_native_text_document(doc)
            for i, page in enumerate(doc, start=1):
                native_text = self.ocr_engine.extract_native_text(page) if trust_native else None
                if native_text is not None:
                    self.log(f"       стр.{i}: найден текстовый слой PDF ({len(native_text)} симв.) — OCR пропущен")
                    page_texts.append(native_text)
                    page_reports.append((i, "Native PDF", 1.0, 1.0))
                    means.append(1.0)
                    p10s.append(1.0)
                    any_rapidocr_kept = True
                    continue

                # Определение ориентации ДО OCR — отдельный узкий промт
                # (docsorter_common/orientation.py), не смешиваем с
                # gemma_vision_ocr() ниже. Реальная проблема: сканер/NAPS2
                # не переворачивают неправильно вложенные листы (чеки)
                # автоматически. page.set_rotation() правит рендер СРАЗУ
                # для всех дальнейших обращений к этой странице (и
                # ocr_page_lines() ниже, и _render_page_jpeg_b64() при
                # эскалации) — fitz учитывает /Rotate при рендере.
                rotation = detect_page_rotation(_render_page_jpeg_b64(page, self.render_dpi), log_fn=self.log)
                if rotation:
                    # rotation — ОТНОСИТЕЛЬНАЯ поправка ("на сколько ещё
                    # повернуть ТО, ЧТО СЕЙЧАС ВИДНО на рендере"), именно
                    # её и нужно передать в page_rotations для
                    # PaperlessClient.edit_pdf_rotate_pages() ниже по
                    # стеку — тот вызов Paperless API "rotate" тоже
                    # относительный (крутит от текущего состояния
                    # СОХРАНЁННОГО файла, как кнопка "повернуть на X°" в
                    # интерфейсе), а не абсолютная установка /Rotate.
                    # Для ЛОКАЛЬНОГО fitz-объекта (эта копия, для OCR)
                    # нужно ДРУГОЕ — set_rotation() принимает абсолютное
                    # значение, поэтому здесь складываем с текущим
                    # page.rotation отдельно, не переиспользуя rotation
                    # напрямую.
                    new_rotation = (page.rotation + rotation) % 360
                    self.log(f"       стр.{i}: обнаружен поворот {rotation}° — "
                             f"исправляю перед OCR (локально /Rotate={new_rotation}°)")
                    page.set_rotation(new_rotation)
                    page_rotations[i] = rotation

                lines = self.ocr_engine.ocr_page_lines(page)
                mean, p10, n = _page_stats(lines)
                means.append(mean if n else 1.0)
                p10s.append(p10 if n else 1.0)

                needs_gemma = n > 0 and (mean < ESCALATE_MEAN or p10 < ESCALATE_P10)
                if needs_gemma:
                    self.log(f"       стр.{i}: mean={mean:.3f} p10={p10:.3f} -> эскалирую в Gemma")
                    img_b64 = _render_page_jpeg_b64(page, self.render_dpi)
                    text, provider_label = gemma_vision_ocr(img_b64, log_fn=self.log, timeout=180)
                    page_texts.append(text)
                    page_reports.append((i, provider_label, mean, p10))
                    any_escalated = True
                else:
                    text = "\n".join(t for t, _ in lines)
                    page_texts.append(text)
                    page_reports.append((i, "RapidOCR", mean, p10))
                    any_rapidocr_kept = True
        finally:
            doc.close()

        if any_escalated and any_rapidocr_kept:
            content_by = CONTENT_BY_MIXED
            mixed_by = "\n".join(f"стр.{i}: {who}" for i, who, _, _ in page_reports)
        elif any_escalated:
            content_by = CONTENT_BY_VISION
            mixed_by = None
        else:
            content_by = CONTENT_BY_RAPIDOCR
            mixed_by = None

        full_content = "\n\n---\n\n".join(page_texts)
        avg_confidence_pct = round(float(np.mean(means)) * 100, 1) if means else None
        avg_accuracy_pct = round(float(np.mean(p10s)) * 100, 1) if p10s else None

        return {
            "content": full_content,
            "content_by": content_by,
            "mixed_by": mixed_by,
            "avg_confidence_pct": avg_confidence_pct,
            "avg_accuracy_pct": avg_accuracy_pct,
            "page_reports": page_reports,
            "page_rotations": page_rotations,
        }

    @staticmethod
    def ensure_fields(client: PaperlessClient) -> dict:
        """Создаёт (если ещё нет) все 5 кастомных полей каскада и возвращает
        {name: field_dict} — то, что нужно build_custom_field_values()."""
        return client.ensure_custom_fields(CUSTOM_FIELD_DEFS)

    @staticmethod
    def build_custom_field_values(client: PaperlessClient, fields: dict, result: dict,
                                   mark_done: bool = True) -> list[dict]:
        """custom_fields[] payload для PATCH /api/documents/{id}/ из результата
        process_document()."""
        content_by_field   = fields["Content by"]
        confidence_field   = fields["Avg. confidence"]
        accuracy_field     = fields["Avg. accuracy"]
        cascade_flag_field = fields[CASCADE_FLAG_FIELD_NAME]

        values = [
            {"field": content_by_field["id"], "value": client.select_option_value(content_by_field, result["content_by"])},
            {"field": confidence_field["id"], "value": result["avg_confidence_pct"]},
            {"field": accuracy_field["id"], "value": result["avg_accuracy_pct"]},
            {"field": cascade_flag_field["id"], "value": mark_done},
        ]
        if result["mixed_by"] is not None:
            values.append({"field": fields["Mixed Created by"]["id"], "value": result["mixed_by"]})
        return values

    @staticmethod
    def document_has_cascade_flag(doc_meta: dict, flag_field_id) -> bool:
        """Проверяет, стоит ли уже true в кастомном поле CASCADE_FLAG_FIELD_NAME."""
        return bool(PaperlessClient.custom_field_value(doc_meta, flag_field_id))
