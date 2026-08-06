r"""
rapidocr_confidence_calibration.py
────────────────────────────────────────────────────────────────────────────
Калибровка RapidOCR confidence на боевой паре движков (latin + eslav — та же
логика, что в DocSorter.py и rapidocr_compare_test.py), чтобы подобрать
порог эскалации в Gemma.

ПРАВИЛО ЭСКАЛАЦИИ (см. ESCALATE_MEAN / ESCALATE_P10 ниже) считается НЕ по
всему документу целиком, а ПО КАЖДОЙ СТРАНИЦЕ ОТДЕЛЬНО — "не средняя
температура по палате, а сколько в морге". Документ на 8 страниц, где
7 страниц идеальны, а 1 страница — рукописный ответ, при усреднении по
всему документу выглядит нормально (плохая страница тонет в общем среднем),
а при постраничной оценке — ловится сразу. Если хотя бы одна страница
документа проваливает mean < ESCALATE_MEAN ИЛИ p10 < ESCALATE_P10 — весь
документ помечается в сводке как ESCALATE.

КАК ЭТО УСТРОЕНО (важно, раз это крутится на большом числе файлов):

  1. Список файлов — в отдельном текстовом файле calibration_files.txt
     (по одному пути/id на строку, # — комментарии). Можно дописывать туда
     новые строки и просто перезапускать скрипт заново.

  2. Результаты копятся в calibration_results.jsonl — по одной JSON-строке
     на документ (путь, категория, разбивка по страницам, все строки текста
     и их score). Построчный формат специально для того, чтобы можно было
     прервать прогон на середине и запустить заново: уже посчитанные
     документы (по пути) скрипт находит в этом файле и пропускает.

  3. Сводка — в calibration_summary.txt при каждом запуске: список документов
     с пометкой ESCALATE/ok, сводка по категориям, худшие/лучшие строки.
     Это единственный файл, который нужно присылать/показывать после
     прогона.

  4. СКОРОСТЬ: документы обрабатываются параллельно (несколько процессов
     одновременно — по умолчанию все ядра минус два). Это единственный
     значимый рычаг ускорения, который тут есть: OCR — это CPU-bound работа
     (никакого скачивания моделей на лету, они уже на диске), и по одному
     документу за раз использует только одно ядро процессора, оставляя
     остальные простаивать. Ryzen 5600 — 6 ядер/12 потоков, так что можно
     получить в разы более быстрый прогон почти бесплатно.

ЗАПУСК:
    python rapidocr_confidence_calibration.py
    python rapidocr_confidence_calibration.py --scan-folder "D:\Доки доки\Docs Germany"
    python rapidocr_confidence_calibration.py --workers 8

    (--workers по умолчанию = число ядер минус 2, но не меньше 1)
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ограничиваем внутренний многопоточный параллелизм onnxruntime до 1 потока
# на процесс-воркер — иначе каждый из N параллельных процессов сам попытается
# расхватать все ядра под капотом (onnxruntime/OpenMP), и вместо ускорения
# получится противоположное — процессы будут толкаться за одни и те же ядра.
# ВАЖНО: должно быть выставлено ДО импорта onnxruntime/rapidocr, поэтому
# делается на уровне модуля, ещё до соответствующих import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import fitz  # PyMuPDF
import numpy as np
import requests

try:
    from rapidocr import RapidOCR, LangRec, OCRVersion, ModelType
except ImportError:
    from rapidocr_onnxruntime import RapidOCR
    LangRec = OCRVersion = ModelType = None

SCRIPT_DIR    = Path(__file__).resolve().parent
RESULTS_DIR   = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FILES_LIST    = RESULTS_DIR / "calibration_files.txt"
RESULTS_JSONL = RESULTS_DIR / "calibration_results.jsonl"
SUMMARY_TXT   = RESULTS_DIR / "calibration_summary.txt"

# Секреты только из окружения — см. .env.example в корне проекта.
PAPERLESS_URL       = os.environ.get("PAPERLESS_URL", "https://paperless.boreray-balance.ts.net")
PAPERLESS_API_TOKEN = os.environ["PAPERLESS_API_TOKEN"]
OCR_RENDER_DPI      = 300

# Порог эскалации в Gemma — считается ПОСТРАНИЧНО (см. докстринг выше).
ESCALATE_MEAN = 0.95
ESCALATE_P10  = 0.95


def _headers() -> dict:
    return {"Authorization": f"Token {PAPERLESS_API_TOKEN}"}


def _log(msg: str) -> None:
    print(msg, flush=True)


# ── чтение списка файлов ────────────────────────────────────────────────

def read_file_list(path: Path) -> list[str]:
    if not path.exists():
        _log(f"  !!  Не найден {path} — создай его (см. пример в репозитории)")
        sys.exit(1)
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def download_paperless_doc(doc_id: int) -> str:
    path = str(RESULTS_DIR / f"doc_{doc_id}_original.pdf")
    if os.path.exists(path):
        return path
    _log(f"  ..  скачиваю doc_id={doc_id} из Paperless")
    resp = requests.get(
        f"{PAPERLESS_URL}/api/documents/{doc_id}/download/",
        headers=_headers(), params={"original": "true"}, timeout=60,
    )
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_folder_into_files_list(folder: str) -> int:
    """Рекурсивно находит все *.pdf в folder, дедуплицирует по sha256
    содержимого, дописывает новые пути в calibration_files.txt."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        _log(f"  !!  Папка не найдена: {folder}")
        sys.exit(1)

    existing_entries = set(read_file_list(FILES_LIST)) if FILES_LIST.exists() else set()
    existing_hashes = set()
    for entry in existing_entries:
        if entry.isdigit():
            continue
        p = Path(entry)
        if p.exists():
            try:
                existing_hashes.add(file_sha256(p))
            except OSError:
                pass

    # На Windows/NTFS сопоставление регистронезависимое, поэтому rglob("*.pdf")
    # сам по себе уже находит и .PDF — оба вызова возвращают один и тот же
    # список путей. Раньше они просто СКЛАДЫВАЛИСЬ (list + list), из-за чего
    # каждый файл считался дважды (542 вместо 271 в выводе — это оно) и дважды
    # хэшировался. Через set() дубли схлопываются на уровне путей — на Linux,
    # где регистр значим, оба варианта имени всё ещё будут найдены корректно.
    all_pdfs = sorted(set(folder_path.rglob("*.pdf")) | set(folder_path.rglob("*.PDF")))
    seen_hashes = set(existing_hashes)
    new_paths = []
    for pdf_path in all_pdfs:
        try:
            h = file_sha256(pdf_path)
        except OSError as e:
            _log(f"  !!  не читается {pdf_path}: {e}")
            continue
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        new_paths.append(str(pdf_path))

    if new_paths:
        with open(FILES_LIST, "a", encoding="utf-8") as f:
            f.write(f"\n# автоматически добавлено сканом папки: {folder}\n")
            for p in new_paths:
                f.write(p + "\n")

    _log(f"  ..  найдено PDF всего: {len(all_pdfs)}, уникальных новых (по содержимому): {len(new_paths)}")
    return len(new_paths)


def resolve_entry(entry: str) -> tuple[str, str]:
    """Возвращает (путь_к_pdf, категория). Для чисел (id в Paperless) уже
    скачивает файл — вызывается в главном процессе ДО раздачи задач
    воркерам, чтобы не скачивать один и тот же документ параллельно из
    нескольких процессов одновременно."""
    if entry.isdigit():
        path = download_paperless_doc(int(entry))
        return path, "Paperless"
    path = entry
    category = Path(path).parent.name or "?"
    return path, category


# ── движки RapidOCR (боевая пара latin + eslav) ─────────────────────────
# Глобальные переменные внутри КАЖДОГО процесса-воркера — инициализируются
# один раз через initializer при старте пула, дальше переиспользуются для
# всех документов, доставшихся этому воркеру (не грузим модели заново на
# каждый файл).
_ENGINE_LATIN = None
_ENGINE_ESLAV = None


def build_ocr_engine(lang_rec, label: str):
    if LangRec is None:
        return RapidOCR()
    try:
        return RapidOCR(params={
            "Rec.lang_type": lang_rec,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
        })
    except Exception as e:
        _log(f"  !!  Не удалось загрузить модель '{label}' ({e}) — используем дефолтную")
        return RapidOCR()


def _worker_init() -> None:
    global _ENGINE_LATIN, _ENGINE_ESLAV
    _ENGINE_LATIN = build_ocr_engine(LangRec.LATIN if LangRec else None, "latin")
    _ENGINE_ESLAV = build_ocr_engine(LangRec.ESLAV if LangRec else None, "eslav")


def ocr_lines(result):
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


def merge_best_per_line(latin_lines, eslav_lines):
    if latin_lines and eslav_lines and len(latin_lines) == len(eslav_lines):
        return [
            lt_ls if lt_ls[1] >= et_es[1] else et_es
            for lt_ls, et_es in zip(latin_lines, eslav_lines)
        ]
    return latin_lines + eslav_lines


def analyze_pdf_per_page(path: str, engine_latin, engine_eslav) -> list[list]:
    """Возвращает список страниц; каждая страница — список (text, score)."""
    pages_out = []
    doc = fitz.open(path)
    try:
        zoom = OCR_RENDER_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = img_rgb[:, :, ::-1]

            latin_lines = ocr_lines(engine_latin(img_bgr))
            eslav_lines = ocr_lines(engine_eslav(img_bgr))
            merged = merge_best_per_line(latin_lines, eslav_lines)
            pages_out.append(list(merged))
    finally:
        doc.close()
    return pages_out


def _process_one(task: tuple[str, str, str]) -> dict:
    """Выполняется в процессе-воркере. task = (entry, path, category)."""
    entry, path, category = task
    t0 = time.monotonic()
    try:
        pages = analyze_pdf_per_page(path, _ENGINE_LATIN, _ENGINE_ESLAV)
        return {
            "path": path, "category": category,
            "pages": pages, "error": None,
            "elapsed": round(time.monotonic() - t0, 1),
        }
    except Exception as e:
        return {
            "path": path, "category": category,
            "pages": [], "error": str(e),
            "elapsed": round(time.monotonic() - t0, 1),
        }


# ── накопление результатов ──────────────────────────────────────────────

def load_already_done() -> set[str]:
    done = set()
    if RESULTS_JSONL.exists():
        for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["path"])
            except Exception:
                pass
    return done


def append_result(record: dict) -> None:
    with open(RESULTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_all_results() -> list[dict]:
    """Читает calibration_results.jsonl. Записи со старой (до постраничной
    разбивки) схемой — {"scores": [...], "lines": [...]} без "pages" —
    оборачиваются как один "виртуальный" документ-страница, чтобы не терять
    уже посчитанные (дорогие) результаты. Для таких документов эскалация
    оценивается по всему документу целиком (как раньше) — то есть ровно
    так же неточно, как до этого изменения, не хуже. Чтобы получить точную
    постраничную оценку и для них — удали их строку из calibration_results.jsonl
    и путь останется в calibration_files.txt, скрипт пересчитает при следующем
    запуске."""
    if not RESULTS_JSONL.exists():
        return []
    records = []
    for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "pages" not in r and "lines" in r:
            r["pages"] = [r["lines"]] if r["lines"] else []
        records.append(r)
    return records


# ── сводка ───────────────────────────────────────────────────────────────

def page_stats(page_lines: list) -> tuple[float, float, float, int]:
    scores = np.array([s for _, s in page_lines], dtype=float)
    if len(scores) == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    return (float(scores.mean()), float(np.percentile(scores, 10)), float(scores.min()), len(scores))


def doc_needs_escalation(record: dict) -> list[int]:
    """Возвращает номера страниц (1-based), которые проваливают порог."""
    bad_pages = []
    for i, page_lines in enumerate(record.get("pages", []), 1):
        mean, p10, _min, n = page_stats(page_lines)
        if n == 0:
            continue
        if mean < ESCALATE_MEAN or p10 < ESCALATE_P10:
            bad_pages.append(i)
    return bad_pages


def write_summary(records: list[dict]) -> None:
    lines_out = []
    w = lines_out.append

    w("Калибровка RapidOCR confidence — сводка")
    w(f"Всего документов: {len(records)}")
    w(f"Правило эскалации (постранично): mean < {ESCALATE_MEAN} ИЛИ p10 < {ESCALATE_P10}")
    w("")

    escalate_list = []
    w(f"{'':9s} {'категория':28s} {'файл':45s} {'стр':>4s} {'mean':>6s} {'p10':>6s} {'min':>6s}")
    w("-" * 110)
    for r in records:
        all_lines = [ln for page in r.get("pages", []) for ln in page]
        arr = np.array([s for _, s in all_lines], dtype=float)
        name = Path(r["path"]).name
        if len(arr) == 0:
            w(f"{'ERR':9s} {r['category']:28s} {name:45s}  ошибка: {r.get('error')}")
            continue
        bad_pages = doc_needs_escalation(r)
        tag = f"ESCALATE" if bad_pages else "ok"
        w(f"{tag:9s} {r['category']:28s} {name:45s} {len(r['pages']):4d} {arr.mean():6.3f} "
          f"{np.percentile(arr,10):6.3f} {arr.min():6.3f}")
        if bad_pages:
            escalate_list.append((r, bad_pages))

    w("")
    w(f"== ДОКУМЕНТЫ НА ЭСКАЛАЦИЮ В GEMMA ({len(escalate_list)} шт.) ==")
    w("(хотя бы одна страница проваливает порог — показаны именно проблемные страницы)")
    for r, bad_pages in escalate_list:
        name = Path(r["path"]).name
        w(f"\n  [{r['category']}] {name}")
        for pno in bad_pages:
            mean, p10, mn, n = page_stats(r["pages"][pno - 1])
            w(f"      стр.{pno}: n={n} mean={mean:.3f} p10={p10:.3f} min={mn:.3f}")

    w("")
    w("== СВОДКА ПО КАТЕГОРИЯМ (по всем строкам, без учёта постраничности) ==")
    by_cat = defaultdict(list)
    for r in records:
        for page in r.get("pages", []):
            by_cat[r["category"]].extend(s for _, s in page)
    for cat, scores in sorted(by_cat.items(), key=lambda kv: np.mean(kv[1]) if kv[1] else 0):
        if not scores:
            continue
        a = np.array(scores)
        w(f"{cat:32s} n={len(a):5d}  mean={a.mean():.3f}  p10={np.percentile(a,10):.3f}  min={a.min():.3f}")

    w("")
    w("== ХУДШИЕ 40 СТРОК ПО ВСЕЙ ВЫБОРКЕ ==")
    flat = []
    for r in records:
        for page in r.get("pages", []):
            for text, score in page:
                flat.append((score, r["category"], Path(r["path"]).name, text))
    for score, cat, name, text in sorted(flat, key=lambda x: x[0])[:40]:
        w(f"{score:.3f}  [{cat}] {name}: {text!r}")

    w("")
    w("== ЛУЧШИЕ 15 СТРОК (для контраста) ==")
    for score, cat, name, text in sorted(flat, key=lambda x: -x[0])[:15]:
        w(f"{score:.3f}  [{cat}] {name}: {text!r}")

    SUMMARY_TXT.write_text("\n".join(lines_out), encoding="utf-8")
    _log(f"\n  OK  сводка записана: {SUMMARY_TXT}  (документов на эскалацию: {len(escalate_list)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan-folder", default=None,
                     help="рекурсивно найти все PDF в этой папке и дописать в calibration_files.txt")
    ap.add_argument("--workers", type=int, default=None,
                     help="сколько процессов гонять параллельно (по умолчанию — ядра минус 2)")
    args = ap.parse_args()

    if args.scan_folder:
        scan_folder_into_files_list(args.scan_folder)

    entries = read_file_list(FILES_LIST)
    already_done = load_already_done()

    to_process = []
    for entry in entries:
        path, category = resolve_entry(entry)
        if path in already_done:
            continue
        to_process.append((entry, path, category))

    if not to_process:
        _log("  i   Все файлы из списка уже посчитаны — пересобираю только сводку.")
    else:
        n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
        _log(f"  ..  новых файлов к обработке: {len(to_process)} (уже готово: {len(already_done)})")
        _log(f"  ..  параллельных процессов: {n_workers} (ядер в системе: {os.cpu_count()})")

        started = time.monotonic()
        done_count = 0
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as pool:
            futures = {pool.submit(_process_one, t): t for t in to_process}
            for fut in as_completed(futures):
                entry, path, category = futures[fut]
                done_count += 1
                try:
                    record = fut.result()
                except Exception as e:
                    record = {"path": path, "category": category, "pages": [], "error": str(e)}
                append_result(record)

                elapsed_total = time.monotonic() - started
                rate = done_count / elapsed_total if elapsed_total > 0 else 0
                remaining = (len(to_process) - done_count) / rate if rate > 0 else float("nan")
                status = "OK" if not record.get("error") else "ERR"
                _log(f"  {status}  [{done_count}/{len(to_process)}]  "
                     f"({record.get('elapsed', '?')}с)  осталось ~{remaining/60:.1f} мин  "
                     f"{Path(path).name}")

    write_summary(load_all_results())


if __name__ == "__main__":
    main()
