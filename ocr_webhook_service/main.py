r"""
ocr_webhook_service/main.py
────────────────────────────────────────────────────────────────────────────
HTTP-сервис — ЕДИНСТВЕННАЯ оставшаяся реализация OCR-каскада +
классификации + Frist-извлечения в проекте. Раньше эта логика была
продублирована в трёх местах (DocSorter.py, apply_ocr_cascade.py и здесь);
теперь:
  - DocSorter.py — тупой загрузчик (только upload в Paperless, каскад не
    трогает вообще);
  - trigger_missing_cascade.py — тонкий вызывающий (находит id без флага
    каскада, дёргает этот же вебхук);
  - Paperless workflow (триггер "Document Added", БЕЗ фильтра по
    источнику — срабатывает на скан/email/веб-загрузку одинаково,
    действие "Webhook") — основной, самый частый путь сюда.

Использует низкоуровневые классы docsorter_common/ (PaperlessClient,
OcrCascade, DualRapidOcrEngine, Classifier) как есть.

ПОЧЕМУ АСИНХРОННО (очередь + фоновый воркер, а не обработка прямо в хендлере
запроса): реальный таймаут Paperless на вебхук-действие не задокументирован
публично. OCR-каскад с эскалацией "плохих" страниц в Gemma vision-LLM может
идти минутами (до 4 попыток по 90-180с на страницу, см.
docsorter_common/ocr_cascade.py::gemma_vision_ocr). Держать HTTP-соединение
с Paperless открытым на всё это время рискованно — вместо этого сервис
сразу кладёт document_id в очередь и отвечает 202, а обработка идёт уже вне
рамок этого запроса. Никакой внешней инфраструктуры (Redis/Celery) — только
queue.Queue() и один фоновый поток в этом же процессе, по образцу
DocSorter.py::QueueWorker (см. его докстринг — та же причина: RapidOCR и
вызовы к LiteLLM-прокси не рассчитаны на конкурентные вызовы, поэтому
обработка идёт строго последовательно, один документ за раз).

ЗАПУСК (локально, без Docker, из корня DocSorter — docsorter_common должен
быть импортируемым как соседний пакет):
    WEBHOOK_SHARED_SECRET=test uvicorn ocr_webhook_service.main:app --reload

ТЕЛО ЗАПРОСА: {"document_id": <int>, "force": <bool, опционально>}. Paperless
шлёт только document_id (см. шаблон вебхука в workflow); force — только для
ручных/скриптовых вызовов (trigger_missing_cascade.py --force) — заставляет
обработать документ заново, даже если уже помечен флагом каскада.

ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ (специфичные для сервиса; PAPERLESS_URL,
WEBHOOK_SHARED_SECRET и остальные "общие" — в docsorter_common/config.py):
    WEBHOOK_SHARED_SECRET  — обязателен, сервис не стартует без него.
    WEBHOOK_PORT           — порт, по умолчанию 8000 (используется только
                             в CMD Dockerfile — uvicorn.run() ниже нужен
                             только для локального запуска "python -m").
    LOG_LEVEL              — INFO (по умолчанию) | DEBUG.
    OCR_CASCADE_APPLY      — true (по умолчанию) | false — при false делает
                             всё, кроме реальной записи в Paperless
                             (единственная точка dry-run во всём проекте
                             теперь — раньше был ещё --dry-run у
                             apply_ocr_cascade.py, но тот скрипт больше не
                             пишет в Paperless сам).
    CASCADE_PROGRESS_FILE  — путь к jsonl-журналу, по умолчанию
                             /data/cascade_progress.jsonl.
"""

import json
import logging
import os
import queue
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Логи сервиса на русском (как и весь остальной проект) — на Windows
# консоль по умолчанию использует cp1252/cp866, который не может закодировать
# кириллицу, и logging тогда падает ("UnicodeEncodeError: 'charmap' codec
# can't encode..."). В Docker-контейнере (Linux, UTF-8 по умолчанию) этой
# проблемы нет вообще, но локальный запуск на Windows (см. README, разработка
# ведётся на Windows-машине) должен работать тоже.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, Header, HTTPException, Request

from docsorter_common import config, frist_extractor, google_tasks
from docsorter_common.classification import Classifier
from docsorter_common.http_utils import _startup_step_with_retry
from docsorter_common.ocr_cascade import CASCADE_FLAG_FIELD_NAME, OcrCascade
from docsorter_common.ocr_engine import DualRapidOcrEngine
from docsorter_common.paperless_client import PaperlessClient

# ── конфигурация из окружения ────────────────────────────────────────────

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ocr_webhook_service")

# Тот же WEBHOOK_SHARED_SECRET, что читает docsorter_common/config.py —
# trigger_missing_cascade.py шлёт им же авторизованные запросы этому
# сервису, одно имя переменной окружения на оба конца.
WEBHOOK_SHARED_SECRET = config.WEBHOOK_SHARED_SECRET
if not WEBHOOK_SHARED_SECRET:
    raise RuntimeError(
        "WEBHOOK_SHARED_SECRET не задан — сервис не должен стартовать без "
        "секрета вебхука (иначе кто угодно, дотянувшийся до эндпоинта, "
        "смог бы поставить в очередь произвольный document_id)."
    )

WEBHOOK_PORT = int(os.environ.get("WEBHOOK_PORT", "8000"))
OCR_CASCADE_APPLY = os.environ.get("OCR_CASCADE_APPLY", "true").strip().lower() not in ("false", "0", "no")
PROGRESS_PATH = Path(os.environ.get("CASCADE_PROGRESS_FILE", "/data/cascade_progress.jsonl"))
OCR_RENDER_DPI = 300


# ── общее состояние сервиса (создаётся один раз при старте) ─────────────

class ServiceState:
    def __init__(self, *, client: PaperlessClient, cascade: OcrCascade, classifier: Classifier,
                 fields: dict, cascade_flag_field_id, frist_field_id: int):
        self.client = client
        self.cascade = cascade
        self.classifier = classifier
        self.fields = fields
        self.cascade_flag_field_id = cascade_flag_field_id
        self.frist_field_id = frist_field_id
        self.doc_queue: "queue.Queue[tuple[int, bool]]" = queue.Queue()
        self.worker_thread: threading.Thread | None = None


def append_progress(record: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_patch_payload(content: str, classification: dict) -> dict:
    """PATCH-payload БЕЗ custom_fields — те пишутся отдельно через
    PaperlessClient.patch_custom_fields() (см. его докстринг: на этом
    Paperless PATCH custom_fields ЗАМЕНЯЕТ весь список, поэтому обычный
    dict-литерал здесь их больше не содержит — раздельная запись custom
    fields в разное время иначе стирала бы друг друга, реальный случай
    5 августа 2026)."""
    payload = {
        "content": content,
        "title": classification["title"],
        "created": classification["date"],
        "tags": classification["tag_ids"],
    }
    if classification["correspondent_id"]:
        payload["correspondent"] = classification["correspondent_id"]
    if classification["document_type_id"]:
        payload["document_type"] = classification["document_type_id"]
    return payload


def process_document(doc_id: int, force: bool, state: ServiceState) -> None:
    """Обработка одного документа — 1-в-1 повторяет цикл apply_ocr_cascade.py
    (скачать оригинал -> каскад -> классификация -> PATCH), только на один
    doc_id и без dry-run (вместо него OCR_CASCADE_APPLY). Исключения
    пробрасываются наверх — вызывающий (worker_loop) решает, что с ними
    делать (лог + запись ошибки в cascade_progress.jsonl).

    force=True — переобработать, даже если уже помечен флагом каскада (это
    единственный оставшийся способ форсировать пересчёт — старой локальной
    резюмируемости в trigger_missing_cascade.py больше нет, "обработан"
    целиком решает этот флаг)."""
    worker_log = logging.getLogger("ocr_webhook_service.worker")
    t0 = time.monotonic()
    worker_log.info(f"[{doc_id}] начинаю обработку (force={force})")

    doc_meta = state.client.get_document(doc_id)
    if OcrCascade.document_has_cascade_flag(doc_meta, state.cascade_flag_field_id) and not force:
        worker_log.info(f"[{doc_id}] уже помечен полем '{CASCADE_FLAG_FIELD_NAME}' — пропускаю "
                         f"(force=true в теле вебхука, чтобы обработать заново)")
        append_progress({"doc_id": doc_id, "applied": False, "skipped": True, "error": None})
        return

    with tempfile.TemporaryDirectory(prefix=f"ocr_cascade_{doc_id}_") as tmp_dir:
        pdf_path = Path(tmp_dir) / f"doc_{doc_id}.pdf"
        state.client.download_original(doc_id, pdf_path)
        worker_log.info(f"[{doc_id}] оригинал скачан, запускаю OCR-каскад")

        result = state.cascade.process_document(pdf_path)
        worker_log.info(f"[{doc_id}] каскад готов: content_by={result['content_by']} "
                         f"confidence={result['avg_confidence_pct']}% accuracy={result['avg_accuracy_pct']}%")

        custom_field_values = OcrCascade.build_custom_field_values(state.client, state.fields, result)

        worker_log.info(f"[{doc_id}] классифицирую")
        classification = state.classifier.classify(result["content"])
        worker_log.info(f"[{doc_id}] классификация готова: title={classification['title']!r} "
                         f"tags={len(classification['tag_ids'])} "
                         f"correspondent_id={classification['correspondent_id']} "
                         f"document_type_id={classification['document_type_id']}")

        if OCR_CASCADE_APPLY:
            # Поворот страниц (docsorter_common/orientation.py, определён
            # внутри cascade.process_document() выше, ДО этого места мы
            # только знаем результат) — правим СОХРАНЁННЫЙ файл в Paperless
            # ДО записи content/классификации: edit_pdf может заставить
            # Paperless пересобрать документ, и наш финальный content должен
            # лечь ПОСЛЕ этого пересбора, а не быть им перезаписанным.
            if result["page_rotations"]:
                worker_log.info(f"[{doc_id}] поворачиваю страницы: {result['page_rotations']}")
                try:
                    state.client.edit_pdf_rotate_pages(doc_id, result["page_rotations"])
                except Exception as e:
                    worker_log.error(f"[{doc_id}] не удалось повернуть страницы в Paperless: {e} "
                                      f"(текст всё равно распознан верно — из уже исправленной ориентации)")

            payload = build_patch_payload(result["content"], classification)
            state.client.patch_custom_fields(doc_id, custom_field_values, extra_payload=payload)
            worker_log.info(f"[{doc_id}] записано в Paperless")

            # Frist — отдельным шагом ПОСЛЕ основного PATCH, читает content
            # уже из Paperless через API — см.
            # docsorter_common/frist_extractor.py. Ошибка здесь не должна
            # ронять уже успешно записанный документ — свой try/except.
            # patch_custom_fields (не голый patch_document!) — иначе эта
            # запись стёрла бы 4 поля каскада, записанные строкой выше (см.
            # докстринг patch_custom_fields).
            try:
                doc_content = state.client.get_document(doc_id)["content"]
                frist = frist_extractor.extract_frist(doc_content, classification["date"], log_fn=worker_log.info)
                if frist:
                    state.client.patch_custom_fields(doc_id, [{"field": state.frist_field_id, "value": frist}])
                    worker_log.info(f"[{doc_id}] Frist: {frist}")
                    if config.FRIST_CREATE_TASKS:
                        try:
                            google_tasks.create_task(classification["title"], frist,
                                                      notes=state.client.document_url(doc_id))
                        except Exception as e:
                            worker_log.error(f"[{doc_id}] Google Tasks: не удалось создать напоминание: {e}")
            except Exception as e:
                worker_log.error(f"[{doc_id}] Frist-экстракция не удалась: {e}")
        else:
            worker_log.info(f"[{doc_id}] OCR_CASCADE_APPLY=false — ничего не пишу (dry-run)")

    append_progress({
        "doc_id": doc_id,
        "content_by": result["content_by"],
        "avg_confidence_pct": result["avg_confidence_pct"],
        "avg_accuracy_pct": result["avg_accuracy_pct"],
        "title": classification["title"],
        "applied": OCR_CASCADE_APPLY,
        "error": None,
    })
    worker_log.info(f"[{doc_id}] готово за {time.monotonic() - t0:.1f}с")


def worker_loop(state: ServiceState) -> None:
    """Крутится в отдельном потоке всё время жизни процесса, разбирает
    очередь строго последовательно — см. докстринг модуля насчёт того,
    почему не параллельно."""
    worker_log = logging.getLogger("ocr_webhook_service.worker")
    worker_log.info("воркер запущен, жду документы в очереди")
    while True:
        doc_id, force = state.doc_queue.get()
        try:
            process_document(doc_id, force, state)
        except Exception as e:
            worker_log.exception(f"[{doc_id}] ошибка при обработке")
            append_progress({"doc_id": doc_id, "error": str(e)})
        finally:
            state.doc_queue.task_done()


# ── FastAPI приложение ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("старт сервиса ocr-cascade-webhook (LOG_LEVEL=%s, OCR_CASCADE_APPLY=%s)",
                LOG_LEVEL, OCR_CASCADE_APPLY)

    # Paperless (Docker) может ещё не быть поднят в момент старта этого
    # контейнера — `docker compose up` не гарантирует порядок готовности.
    # Раньше первая же неудачная попытка здесь роняла lifespan
    # необработанным исключением (контейнер падал в crash-loop, гоняя
    # RestartOnFailure Docker'а вместо того, чтобы просто подождать) — тот
    # же паттерн, что уже был у DocSorter.py, вынесенный в
    # http_utils._startup_step_with_retry.
    client = PaperlessClient(log_fn=logger.debug)
    _startup_step_with_retry("подключение к Paperless (detect_api_version)",
                              client.detect_api_version, log_fn=logger.info)

    logger.info("проверяю/создаю кастомные поля каскада")
    fields = _startup_step_with_retry("кастомные поля каскада",
                                       lambda: OcrCascade.ensure_fields(client), log_fn=logger.info)
    cascade_flag_field_id = fields[CASCADE_FLAG_FIELD_NAME]["id"]

    # Frist — отдельное кастомное поле, создано вручную в Paperless (не
    # через ensure_fields — не часть набора OCR-каскада). Сетевой запрос
    # ретраим, но отсутствие/неверный тип самого поля — конфигурационная
    # ошибка, ждать её исчезновения бессмысленно, падаем сразу.
    frist_field = _startup_step_with_retry("кастомные поля (для Frist)",
                                            client.fetch_all_custom_fields, log_fn=logger.info).get("Frist")
    if frist_field is None or frist_field["data_type"] != "date":
        raise RuntimeError(
            "Кастомное поле 'Frist' (тип date) не найдено в Paperless — "
            "создай его вручную перед запуском (Settings → Custom fields)."
        )
    frist_field_id = frist_field["id"]

    logger.info("загружаю модели RapidOCR (latin + eslav) — может занять время при первом запуске")
    ocr_engine = DualRapidOcrEngine(render_dpi=OCR_RENDER_DPI, log_fn=logger.debug)
    cascade = OcrCascade(ocr_engine, render_dpi=OCR_RENDER_DPI, log_fn=logger.debug)
    classifier = Classifier(client, log_fn=logger.debug)

    state = ServiceState(
        client=client, cascade=cascade, classifier=classifier,
        fields=fields, cascade_flag_field_id=cascade_flag_field_id,
        frist_field_id=frist_field_id,
    )
    app.state.service = state
    app.state.ready = False

    worker = threading.Thread(target=worker_loop, args=(state,), daemon=True, name="ocr-cascade-worker")
    worker.start()
    state.worker_thread = worker
    app.state.ready = True

    logger.info(f"готов принимать вебхуки (порт {WEBHOOK_PORT} в контейнере, /webhook/ocr-cascade)")
    yield
    logger.info("остановка сервиса")


app = FastAPI(title="ocr-cascade-webhook", lifespan=lifespan)


def _parse_webhook_body(raw_body: bytes) -> tuple[int, bool]:
    """Разбирает тело вебхука в (document_id, force), устойчиво к тому, как
    именно настроено действие Webhook в Paperless.

    Реальный случай (3 августа 2026): при действии Webhook с телом-шаблоном
    ("Body", не "Params") и включённым "As JSON" Paperless берёт уже
    отрендеренную Jinja-строку (валидный JSON-текст вида
    '{"document_id": 307}') и передаёт её как значение в httpx-параметр
    json=... — httpx сериализует эту строку ЕЩЁ РАЗ (см. исходники
    paperless-ngx src/documents/workflows/webhooks.py: `if as_json:
    post_args["json"] = data`, где data — уже готовая строка, а не dict).
    В итоге тело запроса — JSON-строка, СОДЕРЖАЩАЯ JSON, а не сам объект
    (json.loads даёт str, а не dict). Вместо того чтобы зависеть от точной
    комбинации чекбоксов "As JSON"/"Params" в конкретной версии Paperless,
    разворачиваем такую двойную обёртку сами — если после первого
    json.loads() получилась строка, а не dict, парсим её ещё раз.

    force — необязательный, по умолчанию false; используется
    trigger_missing_cascade.py --force и любым ручным вызовом, Paperless
    его никогда не присылает (в его шаблоне вебхука только document_id)."""
    parsed = json.loads(raw_body)
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, dict):
        raise ValueError(f"ожидал JSON-объект, получил {type(parsed).__name__}")
    return int(parsed["document_id"]), bool(parsed.get("force", False))


@app.post("/webhook/ocr-cascade", status_code=202)
async def receive_webhook(request: Request,
                           x_webhook_token: str | None = Header(default=None, alias="X-Webhook-Token")):
    if x_webhook_token != WEBHOOK_SHARED_SECRET:
        logger.info("отклонён вебхук: неверный или отсутствующий X-Webhook-Token")
        raise HTTPException(status_code=401, detail="invalid or missing X-Webhook-Token")

    raw_body = await request.body()
    try:
        document_id, force = _parse_webhook_body(raw_body)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.info(f"отклонён вебхук: невалидное тело запроса ({e}); тело: {raw_body[:300]!r}")
        raise HTTPException(status_code=400, detail="missing or invalid 'document_id'")

    state: ServiceState = app.state.service
    state.doc_queue.put((document_id, force))
    logger.info(f"вебхук: doc_id={document_id} force={force} поставлен в очередь "
                f"(длина очереди: {state.doc_queue.qsize()})")
    return {"status": "queued", "document_id": document_id}


@app.get("/healthz")
def healthz():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ocr_webhook_service.main:app", host="0.0.0.0", port=WEBHOOK_PORT)
