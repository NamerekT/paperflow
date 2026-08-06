
"""
DocSorter  v7
───────────────────────────────────────────────────────────────────────────────
Тупой загрузчик: queue → upload в Paperless-ngx (без метаданных) → локальный
бэкап → Windows-тост.

Что изменилось в v7 (единый триггер обработки):
  • DocSorter БОЛЬШЕ НЕ гоняет OCR-каскад/классификацию/Frist сам — раньше
    это было третьей копией той же логики (кроме ocr_webhook_service и
    бывшего apply_ocr_cascade.py). Теперь единственная оставшаяся
    реализация — docsorter_common + ocr_webhook_service/main.py.
  • В Paperless настроен универсальный workflow-триггер "Document Added"
    (БЕЗ фильтра по источнику) → действие Webhook → ocr_webhook_service.
    Он срабатывает на ЛЮБое добавление документа — скан через DocSorter,
    email, веб-загрузку — одинаково. DocSorter про существование вебхука
    вообще не знает и никогда его не вызывает (это дело Paperless, уже
    ПОСЛЕ того, как DocSorter закончил загрузку и забыл про файл).
  • Тост теперь означает "файл принят Paperless", а не "полностью
    обработан" — реальная классификация/OCR-каскад/срок (Frist) приходят
    позже, асинхронно, через вебхук.
  • Побочный эффект: DocSorter больше не тянет rapidocr/onnxruntime/
    pymupdf/numpy вообще — вся тяжёлая OCR/LLM-зависимость осталась только
    в Docker-контейнере ocr_webhook_service. Требования этого файла теперь
    — только requests + watchdog (+ powershell для тостов, Windows-only).
  • Если вебхук в момент загрузки окажется недоступен (Docker не поднят) —
    документ всё равно загрузится, просто останется необработанным до
    следующего ночного прогона trigger_missing_cascade.py (плановая
    страховка, см. README) — DocSorter специально НЕ проверяет доступность
    вебхука: это не его забота, у него нет с вебхуком прямой связи вообще.

v1 → v6 (устарело в v7 — просто история):
  • v6: OCR через docsorter_common/ocr_cascade.py::OcrCascade, PATCH
    content+кастомных полей каскада после загрузки — всё это переехало в
    ocr_webhook_service.
  • v5: Paperless API/RapidOCR/LLM-classification logic moved into the
    shared docsorter_common/ package.
  • v4: RapidOCR вместо удалённого Docling; метаданные через LLM вместо
    папок; upload через REST API.
  • v3: Toast click opens the folder the file was sorted into.

Всё это (устаревшая логика v1-v6) по-прежнему актуально ИСТОРИЧЕСКИ для
apply_ocr_cascade.py/ocr_webhook_service — см. их докстринги/README.

  QUEUE (не менялось с v1)
  • NAPS2 always saves the same Scan.pdf — the script immediately copies it
    to QUEUE_FOLDER/scan_0001.pdf, scan_0002.pdf … with a unique sequential number.
  • The original (Scan.pdf) remains untouched; the next scan can safely overwrite it.
  • QueueWorker thread processes files from the queue one at a time.
  • At startup, files remaining in the queue from the previous run are automatically selected.

  DEBOUNCE WATCHDOG (не менялось с v1)
  • Both events are processed: on_created and on_modified (NAPS2 may send either).
  • Duplicate events for the same file are ignored until enqueue is completed.
───────────────────────────────────────────────────────────────────────────────
Requirements: pip install requests watchdog
Launch: python DocSorter.py
"""

import os
import sys
import time
import shutil
import base64
import logging
import threading
import subprocess
from pathlib import Path

# На Windows-консоли по умолчанию cp1252/cp866 — не может закодировать
# кириллицу, и лог (весь на русском) тогда падает с UnicodeEncodeError. Под
# Task Scheduler/pythonw.exe (без консоли) это не всплывает, но при ручном
# запуске "python DocSorter.py" в обычном терминале — всплывает. Тот же
# фикс, что уже применён в trigger_missing_cascade.py/ocr_webhook_service.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from docsorter_common import config
from docsorter_common.http_utils import _startup_step_with_retry
from docsorter_common.paperless_client import PaperlessClient


# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS — change to your liking
# ══════════════════════════════════════════════════════════════════════════════

# Единая точка правды — docsorter_common/config.py::DOCSORTER_WATCH_FOLDER —
# та же папка, которую trigger_missing_cascade.py --scan-folder явно
# ИСКЛЮЧАЕТ из обхода архива (чтобы не заливать повторно уже загруженные
# отсюда файлы).
WATCH_FOLDER            = str(config.DOCSORTER_WATCH_FOLDER)                              # where NAPS2 saves
QUEUE_FOLDER            = str(config.DOCSORTER_WATCH_FOLDER / "queue")                     # queue
UPLOADED_ARCHIVE_FOLDER = str(config.DOCSORTER_WATCH_FOLDER / "Uploaded to Paperless")     # local backup after upload
LOG_FILE                = r"D:\Projects\Automations\DocSorter\watchdog.log"

PAPERLESS_WAIT_TIMEOUT = 60   # сек — сколько ждать обработки, чтобы дать в уведомлении прямую ссылку на документ

# How many seconds to wait after file appears (NAPS2 may not write file instantly)
WRITE_SETTLE_DELAY = 4


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ],
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  НЕОБРАБОТАННЫЕ ИСКЛЮЧЕНИЯ — обязательно в лог, не молча в никуда
# ══════════════════════════════════════════════════════════════════════════════
#
# Запуск через Task Scheduler (LogonTrigger) использует pythonw.exe — без
# консоли, stdout/stderr никуда не подключены. Раньше необработанное
# исключение (например, сетевой сбой на старте) означало ПОЛНОСТЬЮ молчаливую
# смерть процесса: ни строчки в watchdog.log, только код возврата в истории
# Task Scheduler. Реальный случай: 2026-08-01 07:32 — Task Scheduler запустил
# процесс (запись в его собственном логе есть), а в watchdog.log за это время
# нет вообще ничего. sys.excepthook покрывает главный поток,
# threading.excepthook — QueueWorker и любые другие фоновые потоки (у них
# исключения по умолчанию тоже уходят в stderr, а не в logging, и поток
# тихо умирает, оставляя процесс формально живым, но бездействующим).

def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    log.critical("Необработанное исключение в главном потоке — процесс завершается",
                 exc_info=(exc_type, exc_value, exc_traceback))


def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
    log.critical(f"Необработанное исключение в потоке '{args.thread.name}' — поток завершается",
                 exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


sys.excepthook = _log_uncaught_exception
threading.excepthook = _log_thread_exception


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED CLIENT — Paperless API
#  (docsorter_common/ — общий пакет, но DocSorter теперь берёт из него
#   только PaperlessClient/config/http_utils — никакой OCR/LLM-логики)
# ══════════════════════════════════════════════════════════════════════════════

paperless = PaperlessClient(log_fn=log.info)
_startup_step_with_retry("подключение к Paperless (detect_api_version)",
                          paperless.detect_api_version, log_fn=log.info)


# ══════════════════════════════════════════════════════════════════════════════
#  QUEUE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

_queue_lock = threading.Lock()


def _next_queue_number() -> int:
    """Finds the next sequential number. Call only under _queue_lock."""
    max_num = 0
    for f in Path(QUEUE_FOLDER).glob("scan_*.pdf"):
        try:
            num = int(f.stem.split("_", 1)[1])
            if num > max_num:
                max_num = num
        except (IndexError, ValueError):
            pass
    return max_num + 1


def enqueue_pdf(src_path: str) -> str | None:
    """
    Moves PDF from src_path to QUEUE_FOLDER/scan_NNNN.pdf with unique number.
    Thread-safe. Returns queue file path or None on error.
    """
    with _queue_lock:
        Path(QUEUE_FOLDER).mkdir(parents=True, exist_ok=True)
        num = _next_queue_number()
        dest_name = f"scan_{num:04d}.pdf"
        dest_path = os.path.join(QUEUE_FOLDER, dest_name)
        try:
            shutil.move(src_path, dest_path)
            log.info(f"  +  Queue [{num:04d}]: {dest_name}  ←  {Path(src_path).name}")
            return dest_path
        except Exception as e:
            log.error(f"  ✗  Failed to move to queue: {e}")
            return None


def get_queue_files() -> list[str]:
    """Returns a sorted list of files in the queue."""
    q = Path(QUEUE_FOLDER)
    if not q.exists():
        return []
    return sorted(str(f) for f in q.glob("scan_*.pdf"))


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _ps_b64(s: str) -> str:
    """UTF-16LE + base64 — safe way to hand an arbitrary Python string
    (quotes, Cyrillic, umlauts, newlines) to PowerShell without escaping bugs."""
    return base64.b64encode(s.encode("utf-16-le")).decode("ascii")


def send_toast(title: str, body: str, open_target: str | None = None) -> None:
    """
    Native Windows 11 notification via PowerShell. No extra libraries.

    If open_target is given (a folder path OR a full http(s) URL), clicking
    the notification (protocol activation) opens it — Explorer for a folder,
    the default browser for a URL. No app/AUMID registration required.
    """
    title_b64  = _ps_b64(title)
    body_b64   = _ps_b64(body)
    target_b64 = _ps_b64(open_target) if open_target else ""

    ps = f"""
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType = WindowsRuntime] | Out-Null
[Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null

$title  = [System.Text.Encoding]::Unicode.GetString([System.Convert]::FromBase64String('{title_b64}'))
$body   = [System.Text.Encoding]::Unicode.GetString([System.Convert]::FromBase64String('{body_b64}'))
$titleX = [System.Security.SecurityElement]::Escape($title)
$bodyX  = [System.Security.SecurityElement]::Escape($body)

$launchAttr = ''
$targetB64  = '{target_b64}'
if ($targetB64 -ne '') {{
    $target = [System.Text.Encoding]::Unicode.GetString([System.Convert]::FromBase64String($targetB64))
    $uriX   = [System.Security.SecurityElement]::Escape(([Uri]$target).AbsoluteUri)
    $launchAttr = " activationType=`"protocol`" launch=`"$uriX`""
}}

$toastXmlString = "<toast$launchAttr><visual><binding template=`"ToastGeneric`"><text>$titleX</text><text>$bodyX</text></binding></visual></toast>"

$xmlDoc = New-Object Windows.Data.Xml.Dom.XmlDocument
$xmlDoc.LoadXml($toastXmlString)

$toast = [Windows.UI.Notifications.ToastNotification]::new($xmlDoc)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('DocSorter').Show($toast)
"""
    try:
        subprocess.Popen(
            ['powershell', '-WindowStyle', 'Hidden', '-NonInteractive', '-Command', ps],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception as e:
        log.debug(f'Toast error: {e}')


def safe_filename(name: str) -> str:
    """Removes characters forbidden in Windows file names."""
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name


def archive_local_copy(src: str, filename: str) -> str:
    """
    Once Paperless has accepted the upload, the local copy isn't the canonical
    file anymore — Paperless owns it now. We keep a dumb local backup instead
    of deleting outright, in case something goes wrong on the Paperless side.
    """
    filename = safe_filename(filename)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    os.makedirs(UPLOADED_ARCHIVE_FOLDER, exist_ok=True)
    dest = os.path.join(UPLOADED_ARCHIVE_FOLDER, filename)
    if os.path.exists(dest):
        dest = os.path.join(UPLOADED_ARCHIVE_FOLDER, f"{filename[:-4]}_{int(time.time())}.pdf")

    shutil.move(src, dest)
    return dest


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESSING A SINGLE FILE
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(path: str) -> bool:
    """
    Тупая загрузка: upload в Paperless → локальный архив → тост. Никакого
    OCR/классификации/Frist здесь больше нет — см. докстринг модуля.
    Returns True on success (file moved from queue to the local archive folder).
    False — file stays in queue for a retry.
    """
    name = Path(path).name
    try:
        # Без метаданных (title/tags/correspondent) — Paperless проставит
        # дефолт (обычно имя файла), реальные метаданные придут позже через
        # ocr_webhook_service (universal workflow-триггер "Document Added").
        task_id = paperless.upload_document(path)
        log.info(f"  ✓  Uploaded to Paperless (task {task_id})")

        doc_id      = paperless.wait_for_task(task_id, timeout=PAPERLESS_WAIT_TIMEOUT)
        open_target = paperless.document_url(doc_id) if doc_id else None

        # Имя для локального бэкапа — то, что Paperless реально присвоил
        # (обычно имя файла, раз метаданные не передавались), а не
        # результат классификации — той здесь больше нет.
        archive_name = name
        if doc_id:
            try:
                archive_name = paperless.get_document(doc_id).get("title") or name
            except Exception as e:
                log.warning(f"  ..  не удалось получить title документа {doc_id} для имени бэкапа: {e}")
        else:
            log.warning("  ✗  Paperless не подтвердил обработку вовремя — тост без прямой ссылки "
                        "(документ всё равно загружен, просто doc_id пока неизвестен)")

        archived = archive_local_copy(path, archive_name)
        log.info(f"  ✓  Local backup: {archived}")

        send_toast(
            "📄 Загружено, обрабатывается...",
            f"{archive_name}",
            open_target=open_target,
        )
        return True

    except requests.exceptions.ConnectionError as e:
        log.error(f"  ✗  No connection to Paperless: {e}")
        log.warning("       File stays in queue.")
        time.sleep(60)   # pause to prevent log spam when Paperless turned off
        return False
    except requests.HTTPError as e:
        log.error(f"  ✗  HTTP error: {e} | {e.response.text[:300]}")
        return False
    except Exception as e:
        log.error(f"  ✗  Unexpected error while processing {name}: {e}", exc_info=True)
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  QUEUE WORKER — background processing thread
# ══════════════════════════════════════════════════════════════════════════════

class QueueWorker(threading.Thread):
    """
    Processes files from QUEUE_FOLDER one at a time (strictly sequentially).
    Wakes on notify() signal or every IDLE_CHECK_INTERVAL seconds
    (useful if a queued file could not be processed — it will be retried).
    """
    IDLE_CHECK_INTERVAL = 20   # sec

    def __init__(self):
        super().__init__(daemon=True, name="QueueWorker")
        self._wake = threading.Event()

    def notify(self):
        """Wakes the worker — a new file appeared in the queue."""
        self._wake.set()

    def run(self):
        log.info("  QueueWorker started")
        self._wake.set()   # check queue immediately (files from previous run)

        while True:
            self._wake.wait(timeout=self.IDLE_CHECK_INTERVAL)
            self._wake.clear()
            self._drain_queue()

    def _drain_queue(self):
        """Processes all files in the queue in order."""
        files = get_queue_files()
        if not files:
            return

        log.info(f"  Queue: {len(files)} file(s) waiting")
        for path in files:
            if not os.path.exists(path):
                continue   # already processed while we walked the list

            name = Path(path).name
            log.info(f"  ── Starting: {name}")
            success = process_pdf(path)

            if not success:
                log.warning(f"  ── Skipping {name}, will retry later")
                time.sleep(2)   # brief pause before the next file


# ══════════════════════════════════════════════════════════════════════════════
#  WATCHDOG HANDLER — only enqueues, does not process directly
# ══════════════════════════════════════════════════════════════════════════════

_in_progress: set[str] = set()   # files for which a settle thread is already running
_in_progress_lock = threading.Lock()


class PDFHandler(FileSystemEventHandler):
    """
    Handles on_created and on_modified for .pdf files.
    Debounce: while settle+enqueue is running for a file, duplicates are ignored.
    """

    def __init__(self, worker: QueueWorker):
        self._worker = worker

    def on_created(self, event):
        self._handle(event)

    def on_modified(self, event):
        # NAPS2 may send modified instead of created when overwriting Scan.pdf
        self._handle(event)

    def _handle(self, event):
        if event.is_directory:
            return
        path = event.src_path
        if not path.lower().endswith(".pdf"):
            return

        with _in_progress_lock:
            if path in _in_progress:
                return   # already waiting — ignore duplicate event
            _in_progress.add(path)

        threading.Thread(
            target=self._settle_and_enqueue,
            args=(path,),
            daemon=True,
            name=f"Settle-{Path(path).name}",
        ).start()

    def _settle_and_enqueue(self, path: str):
        try:
            log.info(f"↓  Detected: {Path(path).name}  (waiting {WRITE_SETTLE_DELAY} sec)")
            time.sleep(WRITE_SETTLE_DELAY)   # wait for NAPS2 to finish writing the file

            if not os.path.exists(path):
                log.warning(f"   File disappeared before enqueue: {path}")
                return

            queued = enqueue_pdf(path)
            if queued:
                self._worker.notify()
        finally:
            with _in_progress_lock:
                _in_progress.discard(path)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(WATCH_FOLDER, exist_ok=True)
    os.makedirs(QUEUE_FOLDER, exist_ok=True)
    os.makedirs(UPLOADED_ARCHIVE_FOLDER, exist_ok=True)

    log.info("═" * 60)
    log.info("  DocSorter v7 (тупой загрузчик — каскад/классификация/Frist через ocr_webhook_service)")
    log.info(f"  Watch:     {WATCH_FOLDER}")
    log.info(f"  Queue:     {QUEUE_FOLDER}")
    log.info(f"  Archive:   {UPLOADED_ARCHIVE_FOLDER}")
    log.info(f"  Paperless: {paperless.base_url}")
    log.info("═" * 60)

    worker = QueueWorker()
    worker.start()

    handler  = PDFHandler(worker)
    observer = Observer()
    observer.schedule(handler, WATCH_FOLDER, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        log.info("Stopped via Ctrl+C")
        observer.stop()

    observer.join()
    log.info("DocSorter stopped.")
