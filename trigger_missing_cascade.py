r"""
trigger_missing_cascade.py
────────────────────────────────────────────────────────────────────────────
Плановый (ночной, Task Scheduler) "подборщик" — находит документы в
Paperless и просит ocr_webhook_service их обработать. Сам НИКАКОГО каскада
не гоняет — тонкий вызывающий, единственная оставшаяся реализация каскада/
классификации/Frist живёт в ocr_webhook_service/main.py (см. его докстринг).

Зачем вообще нужен, если есть universal workflow-триггер "Document Added" в
Paperless: тот покрывает документы, добавленные ПОСЛЕ его настройки. Этот
скрипт — страховка на случай, если вебхук был недоступен в момент
добавления документа (Docker не поднят, контейнер перезапускался и т.п.) —
раз в сутки подбирает всё, что реально осталось без флага каскада.

РЕЗЮМИРУЕМОСТЬ: раньше здесь был свой локальный cascade_progress.jsonl —
теперь его нет. "Обработан или нет" решает только живой флаг Paperless
'Updated by OCR Cascade' — эту проверку делает сам ocr_webhook_service при
получении вебхука (см. process_document() там). Этот скрипт просто
шлёт id, ничего не проверяет и не запоминает сам.

ЗАПУСК:
    python trigger_missing_cascade.py                        # все документы -> вебхуку (сам решит, что пропустить)
    python trigger_missing_cascade.py --only 1,4,10           # только эти id
    python trigger_missing_cascade.py --only 1,4,10 --force   # force=true в теле вебхука — переобработать,
                                                                # даже если уже помечен флагом каскада
    python trigger_missing_cascade.py --scan-folder "D:\Доки доки\Docs Germany" --limit 20
                                                                # + залить новые файлы с диска (дедуп по SHA-256
                                                                # содержимого — тот же механизм, что был раньше),
                                                                # БЕЗ явного вызова вебхука для них — универсальный
                                                                # workflow-триггер Paperless сам вызовет вебхук на
                                                                # их "Document Added"
"""

import argparse
import hashlib
import sys
import json
from pathlib import Path

# На Windows-консоли по умолчанию cp1252/cp866 — не может закодировать
# кириллицу, и print() тогда падает ("UnicodeEncodeError: 'charmap' codec
# can't encode..."). Тот же фикс, что уже применён в DocSorter.py/
# ocr_webhook_service/main.py.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

from docsorter_common import config
from docsorter_common.paperless_client import PaperlessClient

SCRIPT_DIR      = Path(__file__).resolve().parent
UPLOADED_JSONL  = SCRIPT_DIR / "cascade_uploaded_from_folder.jsonl"  # локальный путь -> id, чтобы не заливать повторно

PAPERLESS_WAIT_TIMEOUT = 120
WEBHOOK_TIMEOUT = 10   # секундный, не ретраим — пропущенный документ подхватится на следующем плановом прогоне


def log(msg: str) -> None:
    print(msg, flush=True)


def trigger_webhook(doc_id: int, force: bool) -> None:
    if not config.WEBHOOK_SHARED_SECRET:
        raise RuntimeError("WEBHOOK_SHARED_SECRET не задан (переменная окружения) — "
                            "без него сервис отклонит запрос 401")
    body = {"document_id": doc_id}
    if force:
        body["force"] = True
    resp = requests.post(
        config.WEBHOOK_URL,
        headers={"X-Webhook-Token": config.WEBHOOK_SHARED_SECRET, "Content-Type": "application/json"},
        json=body,
        timeout=WEBHOOK_TIMEOUT,
    )
    resp.raise_for_status()
    log(f"  ..  doc_id={doc_id}: {resp.json()}")


# ── дедуп заливки по SHA-256 (не менялось со старого apply_ocr_cascade.py) ──

def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Хэш содержимого файла — используется вместо пути для дедупликации
    заливки. В архиве одни и те же сканы (например, Ausweis) лежат в
    нескольких папках сразу — один и тот же файл под разными путями."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_uploaded_state() -> tuple[dict[str, int], dict[str, int]]:
    """Возвращает (path -> doc_id, content_hash -> doc_id)."""
    by_path: dict[str, int] = {}
    by_hash: dict[str, int] = {}
    if UPLOADED_JSONL.exists():
        for line in UPLOADED_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            by_path[r["path"]] = r["doc_id"]
            file_hash = r.get("hash")
            if file_hash is None:
                p = Path(r["path"])
                if p.exists():
                    try:
                        file_hash = file_sha256(p)
                    except OSError:
                        file_hash = None
            if file_hash is not None:
                by_hash[file_hash] = r["doc_id"]
    return by_path, by_hash


def append_uploaded(path: str, doc_id: int, file_hash: str) -> None:
    with open(UPLOADED_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({"path": path, "doc_id": doc_id, "hash": file_hash}, ensure_ascii=False) + "\n")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="список id через запятую — обработать только их")
    ap.add_argument("--scan-folder", default=None, help="залить все новые PDF из папки (без метаданных)")
    ap.add_argument("--limit", type=int, default=None, help="максимум НОВЫХ файлов из --scan-folder за прогон")
    ap.add_argument("--force", action="store_true",
                     help="force=true в теле вебхука — обработать заново, даже если уже помечен флагом каскада "
                          "(имеет смысл почти всегда только вместе с --only)")
    args = ap.parse_args()

    client = PaperlessClient(log_fn=log)
    client.detect_api_version()

    if args.only:
        doc_ids = [int(x) for x in args.only.split(",")]
        log(f"  ..  режим --only: {doc_ids}")
    else:
        doc_ids = client.fetch_all_document_ids()
        log(f"  ..  документов в Paperless: {len(doc_ids)} (вебхук сам решит, какие пропустить)")

    if args.scan_folder:
        uploaded_by_path, uploaded_by_hash = load_uploaded_state()
        folder = Path(args.scan_folder)
        all_pdfs = sorted(set(folder.rglob("*.pdf")) | set(folder.rglob("*.PDF")))
        # Папка DocSorter.py (queue/ ещё не обработана; "Uploaded to
        # Paperless/" — уже документы в Paperless) — заливать её содержимое
        # отсюда снова значило бы либо подхватить недообработанный файл,
        # либо создать документ-дубликат уже существующего.
        pdfs = [p for p in all_pdfs if not p.is_relative_to(config.DOCSORTER_WATCH_FOLDER)]
        skipped = len(all_pdfs) - len(pdfs)
        if skipped:
            log(f"  ..  пропускаю {skipped} файл(ов) из {config.DOCSORTER_WATCH_FOLDER} (папка DocSorter.py)")

        new_uploads = 0
        for pdf_path in pdfs:
            key = str(pdf_path)
            if key in uploaded_by_path:
                continue

            try:
                file_hash = file_sha256(pdf_path)
            except OSError as e:
                log(f"  !!  не смог прочитать {pdf_path}: {e}")
                continue

            if file_hash in uploaded_by_hash:
                existing_id = uploaded_by_hash[file_hash]
                log(f"  ..  {pdf_path} — уже заливали как id={existing_id} под другим путём, пропускаю")
                uploaded_by_path[key] = existing_id
                append_uploaded(key, existing_id, file_hash)
                continue

            if args.limit is not None and new_uploads >= args.limit:
                continue

            log(f"  ..  заливаю: {pdf_path}")
            # Без метаданных — универсальный workflow-триггер Paperless
            # ("Document Added", без фильтра) сам вызовет вебхук после
            # того, как документ реально появится в базе.
            new_id = client.upload_and_wait(pdf_path, timeout=PAPERLESS_WAIT_TIMEOUT)
            if new_id is not None:
                append_uploaded(key, new_id, file_hash)
                uploaded_by_path[key] = new_id
                uploaded_by_hash[file_hash] = new_id
                new_uploads += 1
            else:
                log(f"  !!  не удалось залить: {pdf_path}")

    log(f"  ..  отправляю {len(doc_ids)} id вебхуку ({config.WEBHOOK_URL})")
    ok, failed = 0, 0
    for doc_id in doc_ids:
        try:
            trigger_webhook(doc_id, force=args.force)
            ok += 1
        except Exception as e:
            log(f"  !!  doc_id={doc_id}: {e}")
            failed += 1

    log(f"  ..  готово: {ok} принято, {failed} ошибок")


if __name__ == "__main__":
    main()
