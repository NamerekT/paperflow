"""
docling_ocr_test.py
────────────────────────────────────────────────────────────────────────────
Разовый тестовый скрипт (НЕ встроен в DocSorter.py): скачивает оригинал
документа из Paperless, гоняет его через Docling (layout-aware OCR) и либо
печатает получившийся Markdown, либо (с --patch) записывает его прямо в
поле content документа через PATCH /api/documents/{id}/.

Зачем отдельно от DocSorter: сначала нужно своими глазами увидеть, действительно
ли Docling исправляет то, что сейчас ломает NAPS2/Tesseract (пропущенный текст,
путаница кириллица/латиница), прежде чем встраивать это в постоянный пайплайн.

ПОЧЕМУ ЭТОТ СКРИПТ НАДО ЗАПУСКАТЬ У СЕБЯ, А НЕ Я ЕГО ЗАПУСТИЛ:
Я пытался прогнать тест в своей облачной песочнице — не получилось физически:
у неё всего ~9.8 ГБ диска, а `pip install docling` по умолчанию тащит
CUDA-версию torch (это стандартный wheel с PyPI на Linux, ~2-3 ГБ библиотек
для видеокарты). Установка упала с "No space left on device". Официальный
CPU-only индекс PyTorch (download.pytorch.org) тоже недоступен — прокси
песочницы его блокирует. На твоём ПК с нормальным интернетом это не проблема,
если поставить torch правильно (см. ниже).

УСТАНОВКА (один раз; сам Docling потом не держит НИЧЕГО в памяти между
запусками — грузится и работает только пока выполняется этот скрипт,
как ты и просил изначально):

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install docling

Первая строка ВАЖНА: без неё pip поставит тебе CUDA-версию torch (лишние
~2 ГБ библиотек для видеокарты, которой на этом ПК нет). Команда выше ставит
CPU-only сборку — она меньше и не тянет CUDA рантайм. Ставь именно в этом
порядке (torch первым), тогда Docling увидит, что torch уже есть, и не
попытается заменить его на GPU-версию.

ВАЖНО ПРО ЯЗЫКИ (проверено по исходникам Docling/EasyOCR, а не на глаз):
EasyOCR (дефолтный OCR-движок Docling) НЕ даёт смешать немецкий и украинский
в одном вызове — в его коде есть проверка, которая кидает ValueError при
смешении латинской и кириллической языковых групп. Документы 29 и 4 — чисто
украинские, так что здесь это не проблема (lang=["uk"]). Но если в будущем
захочешь прогнать через Docling смешанный документ (немецкий + украинский на
одной странице) — потребуется два прохода с разными lang и merge результата,
как уже сделано для RapidOCR в DocSorter.py (_merge_best_per_line).

ЗАПУСК:
    python docling_ocr_test.py 29                # только показать результат
    python docling_ocr_test.py 29 --patch         # показать и записать в content
    python docling_ocr_test.py 4 --patch --lang uk
    python docling_ocr_test.py <id> --lang de     # для немецкого документа
"""

import argparse
import os
import sys
from pathlib import Path

import requests

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Секреты только из окружения — см. .env.example в корне проекта.
PAPERLESS_URL       = os.environ.get("PAPERLESS_URL", "https://paperless.boreray-balance.ts.net")
PAPERLESS_API_TOKEN = os.environ["PAPERLESS_API_TOKEN"]


def _headers() -> dict:
    return {"Authorization": f"Token {PAPERLESS_API_TOKEN}"}


def download_original(doc_id: int) -> str:
    """Скачивает ОРИГИНАЛ (не архивную копию с уже кривым текстовым слоем)."""
    resp = requests.get(
        f"{PAPERLESS_URL}/api/documents/{doc_id}/download/",
        headers=_headers(),
        params={"original": "true"},
        timeout=60,
    )
    resp.raise_for_status()
    path = str(RESULTS_DIR / f"doc_{doc_id}_original.pdf")
    with open(path, "wb") as f:
        f.write(resp.content)
    print(f"  OK  скачан оригинал: {path}  ({len(resp.content)} байт)")
    return path


def run_docling(pdf_path: str, lang: str) -> str:
    """
    force_full_page_ocr=True — обязательно для наших сканов: страницы уже
    содержат кривой текстовый слой от NAPS2, и без этого флага Docling (как и
    Paperless по умолчанию) может решить, что текст уже есть, и не станет
    перераспознавать страницу заново.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        ocr_options=EasyOcrOptions(lang=[lang], force_full_page_ocr=True),
    )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


def patch_content(doc_id: int, markdown: str) -> None:
    """
    Пишет Markdown прямо в поле content документа. Проверено через браузер
    (OPTIONS /api/documents/{id}/): поле НЕ read-only, PATCH разрешён
    (Allow: GET, PUT, PATCH, DELETE, HEAD, OPTIONS). Сам PDF-файл при этом не
    трогается — меняется только то, что Paperless использует для поиска.
    """
    resp = requests.patch(
        f"{PAPERLESS_URL}/api/documents/{doc_id}/",
        headers=_headers(),
        json={"content": markdown},
        timeout=30,
    )
    resp.raise_for_status()
    print(f"  OK  content документа {doc_id} обновлён ({len(markdown)} символов)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc_id", type=int)
    ap.add_argument("--lang", default="uk", help="код языка для EasyOCR (по умолчанию uk)")
    ap.add_argument("--patch", action="store_true", help="записать результат в content документа")
    args = ap.parse_args()

    pdf_path = download_original(args.doc_id)
    print(f"  ..  гоняем Docling (lang={args.lang}); первый запуск дольше — грузит модели")
    markdown = run_docling(pdf_path, args.lang)

    print("\n" + "=" * 70)
    print(markdown)
    print("=" * 70 + "\n")

    out_path = str(RESULTS_DIR / f"doc_{args.doc_id}_docling.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"  OK  сохранено локально: {out_path}")

    if args.patch:
        patch_content(args.doc_id, markdown)
    else:
        print("  i   content НЕ записан — перезапусти с --patch, если результат устраивает")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        sys.exit(0)
    main()
