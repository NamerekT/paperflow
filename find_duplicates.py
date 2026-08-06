r"""
find_duplicates.py
────────────────────────────────────────────────────────────────────────────
Paperless-ngx не показывает дубли списком нигде в интерфейсе — только
вкладка "Duplicates" на странице ОТДЕЛЬНОГО документа. Общего дашборда
"покажи все дубли архива разом" в самом Paperless нет (см. GitHub discussion
#345 "How to find duplicates?"). Этот скрипт — такой batch job.

ИСТОРИЯ ТРЁХ ЗАХОДОВ (для контекста, если снова не будет находить то, что
нужно, — не наступай на те же грабли ещё раз):

  1. checksum из API — DocumentSerializer вообще не объявляет это поле
     в Meta.fields ни для одной версии API. Не приходит никогда.
  2. SHA-256 по скачанным файлам — даже если бы checksum приходил, не
     помогло бы: в documents/models.py у Document поле checksum объявлено
     unique=True (UNIQUE-констрейнт в Postgres) — Paperless физически не
     даёт сохранить два байт-в-байт идентичных файла, так что дублей по
     точному хэшу в архиве в принципе быть не может. Все "живые" дубли —
     это ПЕРЕСКАН того же документа (другие байты, тот же смысл).
  3. perceptual hash (phash) целой страницы — реальный тест на архиве
     показал два вида провала СРАЗУ:
       - false negative: id=55/56 (буквально то же письмо VHS, отличие в
         одном слове "Valentyn"/"Valentyna") — НЕ поймал;
       - false positive: 25 совершенно разных писем от Jobcenter/
         Familienkasse (разные суммы, даты, адресаты) склеились в одну
         группу, потому что у немецких официальных писем одного ведомства
         почти одинаковая вёрстка/шапка — phash целой страницы видит
         в первую очередь ЭТО (общую структуру), а не то, что реально
         написано в тексте. Плюс кластеризация была через Union-Find
         (транзитивная): если A похож на B, а B похож на C, то A и C
         попадают в одну группу, даже если A и C друг на друга не похожи
         вообще — отсюда и группа из 25.

ЧТО ДЕЛАЕТ ЭТОТ (четвёртый) ЗАХОД:
  - Основной сигнал — сам OCR-текст документа (поле "content" — он, в
    отличие от checksum, ЕСТЬ в DocumentSerializer.Meta.fields и приходит
    через обычный REST API, скачивать файлы для текстовых документов не
    нужно вообще). Текст нормализуется (нижний регистр, схлопнутые
    пробелы) и режется на "шинглы" — скользящие окна по 8 слов подряд;
    похожесть = коэффициент Жаккара (|пересечение| / |объединение|)
    множеств шинглов двух документов. Это ловит именно "тот же текст,
    поменяли имя/дату/сумму" — ровно то, что не поймал phash, и НЕ ловит
    "тот же шаблон ведомства, другое содержание", потому что содержание
    (даты, суммы, ФИО, номера дел) не совпадёт почти нигде.
  - phash страниц остаётся ЗАПАСНЫМ вариантом — только для документов, где
    OCR-текста почти нет (< TEXT_MIN_CHARS символов после нормализации):
    сканы паспортов, фото, документы без текстового слоя.
  - Кластеризация транзитивным Union-Find УБРАНА. Вместо "групп" скрипт
    печатает ПАРЫ документов с реальным числом схожести — так видно, из-за
    чего именно сработало совпадение, и невозможна склейка несвязанных
    документов через цепочку промежуточных.

Только ПОКАЗЫВАЕТ дубли — ничего не удаляет и не трогает.

ЗАПУСК:
    pip install imagehash --break-system-packages   # нужен только для phash-запасного пути; Pillow/PyMuPDF уже используются в проекте
    python find_duplicates.py
    python find_duplicates.py --text-threshold 0.5 --image-threshold 10
"""

import argparse
import re
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from docsorter_common.paperless_client import PaperlessClient

try:
    import fitz
    import imagehash
    from PIL import Image
    _HAS_IMAGE_DEPS = True
except ImportError:
    _HAS_IMAGE_DEPS = False

# Документов с текстом короче этого (после нормализации) считаем
# "не текстовыми" — почти наверняка это скан без OCR-слоя или чисто
# графический документ (фото, паспорт) — используем phash-запасной путь.
TEXT_MIN_CHARS = 100
SHINGLE_WORDS = 8
DEFAULT_TEXT_THRESHOLD = 0.55    # коэффициент Жаккара, 0-1
DEFAULT_IMAGE_THRESHOLD = 8      # расстояние Хэмминга между phash, 0-64 (для запасного пути)
HASH_SIZE = 8                    # imagehash.phash(hash_size=8) -> 64-битный хэш
IMAGE_PAGES = 3                  # сколько первых страниц хэшировать в запасном пути


def log(msg: str) -> None:
    print(msg, flush=True)


# ── текстовый путь (основной) ────────────────────────────────────────────

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _normalize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _shingles(words: list[str], n: int = SHINGLE_WORDS) -> set[tuple]:
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ── графический запасной путь (для документов почти без текста) ─────────

def _pdf_page_hashes(path: Path, max_pages: int) -> list:
    hashes = []
    try:
        with fitz.open(path) as doc:
            for i in range(min(max_pages, doc.page_count)):
                pix = doc[i].get_pixmap()
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                hashes.append(imagehash.phash(img, hash_size=HASH_SIZE))
    except Exception as e:
        log(f"  !!  {path.name}: не удалось отрендерить PDF — {e}")
    return hashes


def _image_hash(path: Path) -> list:
    try:
        with Image.open(path) as img:
            return [imagehash.phash(img.convert("RGB"), hash_size=HASH_SIZE)]
    except Exception as e:
        log(f"  !!  {path.name}: не удалось открыть как изображение — {e}")
        return []


def _hash_document(client: PaperlessClient, doc: dict, tmp_dir: Path, max_pages: int) -> tuple[dict, list]:
    doc_id = doc["id"]
    mime = doc.get("mime_type") or ""
    suffix = ".pdf" if "pdf" in mime else (Path(doc.get("original_file_name") or "").suffix or ".bin")
    dest = tmp_dir / f"{doc_id}{suffix}"
    try:
        client.download_original(doc_id, dest)
        if "pdf" in mime:
            hashes = _pdf_page_hashes(dest, max_pages)
        elif mime.startswith("image/"):
            hashes = _image_hash(dest)
        else:
            hashes = []
        return doc, hashes
    except Exception as e:
        log(f"  !!  id={doc_id} ({doc.get('original_file_name') or '?'}): {e}")
        return doc, []
    finally:
        dest.unlink(missing_ok=True)


def _doc_line(d: dict) -> str:
    created = (d.get("created") or "?")[:10]
    title = d.get("title") or "(без названия)"
    fname = d.get("original_file_name") or "?"
    return f"     id={d['id']:<6} {created}  {title!r:<60} [{fname}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6,
                         help="параллельность скачивания для запасного (графического) пути")
    parser.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD,
                         help=f"мин. коэффициент Жаккара по тексту, 0-1 (по умолчанию {DEFAULT_TEXT_THRESHOLD})")
    parser.add_argument("--image-threshold", type=int, default=DEFAULT_IMAGE_THRESHOLD,
                         help=f"макс. расстояние Хэмминга phash для документов без текста (по умолчанию {DEFAULT_IMAGE_THRESHOLD})")
    parser.add_argument("--pages", type=int, default=IMAGE_PAGES,
                         help=f"сколько первых страниц хэшировать в запасном пути (по умолчанию {IMAGE_PAGES})")
    args = parser.parse_args()

    client = PaperlessClient(log_fn=log)
    client.detect_api_version()

    log("  ..  забираю документы из Paperless (вместе с OCR-текстом, content уже есть в API)...")
    docs = client.fetch_documents_fields(["id", "title", "original_file_name", "created", "mime_type", "content"])
    log(f"       всего документов: {len(docs)}")

    text_docs: list[tuple[dict, set]] = []
    image_docs: list[dict] = []
    for d in docs:
        words = _normalize_words(d.get("content") or "")
        if len(" ".join(words)) >= TEXT_MIN_CHARS:
            text_docs.append((d, _shingles(words)))
        else:
            image_docs.append(d)

    log(f"       текстовых (сравниваю по OCR-содержимому): {len(text_docs)}")
    log(f"       почти без текста (сравниваю по phash страниц): {len(image_docs)}")

    text_pairs = []
    for i in range(len(text_docs)):
        d_i, sh_i = text_docs[i]
        for j in range(i + 1, len(text_docs)):
            d_j, sh_j = text_docs[j]
            score = _jaccard(sh_i, sh_j)
            if score >= args.text_threshold:
                text_pairs.append((score, d_i, d_j))
    text_pairs.sort(key=lambda x: -x[0])

    image_pairs = []
    if image_docs:
        if not _HAS_IMAGE_DEPS:
            log("  !!  imagehash/PyMuPDF/Pillow не установлены — пропускаю "
                f"{len(image_docs)} документ(ов) почти без текста (pip install imagehash --break-system-packages)")
        else:
            log(f"  ..  скачиваю и хэширую {len(image_docs)} документ(ов) почти без текста "
                f"(параллельно, {args.workers} потоков)...")
            doc_hashes: dict[int, tuple[dict, list]] = {}
            with tempfile.TemporaryDirectory(prefix="docsorter_dupcheck_") as tmp:
                tmp_dir = Path(tmp)
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = [pool.submit(_hash_document, client, d, tmp_dir, args.pages) for d in image_docs]
                    for fut in as_completed(futures):
                        doc, hashes = fut.result()
                        doc_hashes[doc["id"]] = (doc, hashes)
            usable = [(d, h) for d, h in doc_hashes.values() if h]
            for i in range(len(usable)):
                d_i, h_i = usable[i]
                for j in range(i + 1, len(usable)):
                    d_j, h_j = usable[j]
                    best = min(x - y for x in h_i for y in h_j)
                    if best <= args.image_threshold:
                        image_pairs.append((best, d_i, d_j))
            image_pairs.sort(key=lambda x: x[0])

    if not text_pairs and not image_pairs:
        log("\n  OK  дублей не найдено (текстовый порог "
            f"{args.text_threshold}, графический {args.image_threshold} — попробуй ослабить, если ожидал больше).")
        return

    if text_pairs:
        log(f"\n  !!  текстовые совпадения (по OCR-содержимому): {len(text_pairs)}\n")
        for score, d_i, d_j in text_pairs:
            log(f"── схожесть текста: {score:.0%}")
            log(_doc_line(d_i))
            log(f"            -> {client.base_url}/documents/{d_i['id']}/details")
            log(_doc_line(d_j))
            log(f"            -> {client.base_url}/documents/{d_j['id']}/details")
            log("")

    if image_pairs:
        log(f"\n  !!  графические совпадения (документы почти без текста, phash): {len(image_pairs)}\n")
        for dist, d_i, d_j in image_pairs:
            log(f"── расстояние phash: {dist} (0 = идентично, {args.image_threshold} = порог)")
            log(_doc_line(d_i))
            log(f"            -> {client.base_url}/documents/{d_i['id']}/details")
            log(_doc_line(d_j))
            log(f"            -> {client.base_url}/documents/{d_j['id']}/details")
            log("")


if __name__ == "__main__":
    main()
