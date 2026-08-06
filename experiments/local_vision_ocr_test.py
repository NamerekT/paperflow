"""
local_vision_ocr_test.py
────────────────────────────────────────────────────────────────────────────
То же самое, что vision_ocr_test.py, но без OpenRouter/облака — гоняет
vision-модель ЛОКАЛЬНО через Ollama на твоей RTX 5060. Ни лимитов, ни 429,
ни зависаний на чужом бесплатном бэкенде — модель либо тянет твою видеокарту,
либо нет, и это сразу видно.

УСТАНОВКА (один раз):
    1. Скачай и поставь Ollama: https://ollama.com/download/windows
       (после установки в трее появится иконка — сервер слушает на
       http://localhost:11434 сам по себе, отдельно запускать не нужно)
    2. Скачай модель (несколько ГБ, один раз):
           ollama pull qwen3-vl:8b
       Это основной вариант. Проверил по независимым сравнениям OCR-точности
       на кириллице: Qwen в них оказался лучшим по точности именно на
       кириллическом тексте среди открытых моделей, Gemma заметно слабее
       (~70% точности на кириллице в том же сравнении), а MiniCPM-V в целом
       показал слабые результаты по OCR (я его раньше рекомендовал в первую
       очередь — ошибся, понижаю приоритет). Для немецкого/латиницы Qwen
       тоже без проблем — это одна из основных языков в его обучающих данных.
       8B поместится в 8 ГБ VRAM RTX 5060 в квантованном виде.

    Альтернативы, если qwen3-vl не устроит по качеству или скорости:
           ollama pull gemma4:12b            — сильнее Qwen по немецкому
                                                (родной язык Google — one of
                                                the best-covered), слабее по
                                                украинскому (см. выше), но
                                                всё ещё рабочий вариант
           ollama pull glm-ocr                — 0.9B, 2.2 ГБ, ОЧЕНЬ быстрая,
                                                #1 на бенчмарке OmniDocBench
                                                (таблицы/печати/сложная
                                                вёрстка), но тренирована с
                                                акцентом на китайский/
                                                английский — украинский под
                                                вопросом
           ollama pull maternion/lightonocr-2 — 1B, 1.5 ГБ, очень быстрая,
                                                но заточена авторами (LightOn,
                                                Франция) под французский и
                                                научные статьи/LaTeX —
                                                украинский тоже под вопросом
           ollama pull deepseek-ocr           — 3B, 6.7 ГБ, общего назначения
                                                OCR, охват кириллицы не
                                                подтверждён
           ollama pull minicpm-v4.5:8b        — держи про запас, но по
                                                данным сравнений — не лучший
                                                выбор для этой задачи

Ollama выгружает модель из VRAM сама через ~5 минут простоя — не висит в
памяти постоянно, пока ты явно не гоняешь скрипт.

ЕСЛИ МОДЕЛЬ ВСЁ РАВНО "УМИРАЕТ" (обычно значит — не влезла в VRAM):
    1. Уже сделано в этом скрипте: контекст обрезан до 4096 токенов
       (--num-ctx) и рендер снижен до 150 DPI — оба параметра напрямую
       уменьшают требуемую VRAM, без потери в качестве весов модели.
    2. Если всё равно не лезет — возьми модель поменьше:
           ollama pull qwen3-vl:4b   (3.3 ГБ вместо 6.1 ГБ у 8b)
           ollama pull qwen3-vl:2b   (1.9 ГБ — почти наверняка влезет)
       Qwen3-VL — это одна архитектура на разных размерах, OCR-улучшения
       (32 языка) есть во всей линейке, не только в 8B.
    3. Ещё один рычаг — квантовать не веса (они уже Q4 по умолчанию), а
       KV-cache контекста: переменная окружения OLLAMA_KV_CACHE_TYPE=q8_0
       (или q4_0 для ещё большей экономии) перед запуском Ollama. На Windows
       задать через "Изменение переменных среды для вашей учётной записи",
       затем перезапустить Ollama (иконка в трее -> Quit -> запустить снова).
    4. Проверь, что VRAM не делят с чем-то ещё (игры, браузер с аппаратным
       ускорением, второй монитор с тяжёлым десктопом) — `ollama ps` и
       диспетчер задач -> вкладка "Производительность" -> GPU покажут,
       сколько памяти реально свободно перед запуском.

ЗАПУСК:
    python local_vision_ocr_test.py 4
    python local_vision_ocr_test.py 4 --model minicpm-v4.5:8b
    python local_vision_ocr_test.py 4 --patch
"""

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Секреты только из окружения — см. .env.example в корне проекта.
PAPERLESS_URL       = os.environ.get("PAPERLESS_URL", "https://paperless.boreray-balance.ts.net")
PAPERLESS_API_TOKEN = os.environ["PAPERLESS_API_TOKEN"]

OLLAMA_URL  = "http://localhost:11434"
RENDER_DPI  = 150   # снижено с 200: меньше картинка -> меньше vision-токенов -> меньше VRAM
NUM_CTX     = 4096  # Qwen3-VL "родное" окно 256K — Ollama по умолчанию резервирует
                     # под него VRAM даже на одну страницу. Нам за глаза хватит
                     # 4096 токенов (картинка + промпт + разумный ответ), это резко
                     # снижает требования к памяти без потери качества распознавания.

PROMPT = (
    "Распознай ВЕСЬ текст на этом изображении документа — печатный и "
    "рукописный (документ может быть на украинском или немецком). Выведи "
    "результат как plain text, сохраняя порядок блоков на странице. "
    "Ничего не придумывай и не пропускай — если что-то неразборчиво, "
    "отметь это как [нерозбірливо] или [unleserlich], но не оставляй пропуски молча. "
    "Не добавляй собственных комментариев или пояснений — только "
    "распознанный текст."
)


def log(msg: str) -> None:
    print(msg, flush=True)


def _paperless_headers() -> dict:
    return {"Authorization": f"Token {PAPERLESS_API_TOKEN}"}


def check_ollama_running() -> None:
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Не могу достучаться до Ollama на {OLLAMA_URL} ({e}). "
            f"Установлена ли Ollama и запущена ли она (иконка в трее)? "
            f"См. https://ollama.com/download/windows"
        )


def get_or_download(doc_id: int) -> str:
    path = str(RESULTS_DIR / f"doc_{doc_id}_original.pdf")
    if os.path.exists(path):
        log(f"  OK  уже скачан ранее: {path}")
        return path
    resp = requests.get(
        f"{PAPERLESS_URL}/api/documents/{doc_id}/download/",
        headers=_paperless_headers(), params={"original": "true"}, timeout=60,
    )
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    log(f"  OK  скачан оригинал: {path}  ({len(resp.content)} байт)")
    return path


def render_pages_base64(pdf_path: str) -> list[str]:
    images = []
    doc = fitz.open(pdf_path)
    try:
        zoom = RENDER_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            jpeg_bytes = pix.tobytes("jpeg", jpg_quality=85)
            images.append(base64.b64encode(jpeg_bytes).decode("ascii"))
    finally:
        doc.close()
    return images


def ocr_page_with_ollama(image_b64: str, model: str, timeout: int, num_ctx: int) -> str:
    started = time.monotonic()
    log(f"       -> отправляю страницу модели {model} (может грузиться в VRAM "
        f"первый раз — не пугайся, если первая страница дольше остальных)...")
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": PROMPT, "images": [image_b64]}
            ],
            "stream": False,
            "options": {"num_ctx": num_ctx},
        },
        timeout=timeout,
    )
    elapsed = time.monotonic() - started
    log(f"       <- ответ за {elapsed:.1f}с, статус {resp.status_code}")

    if resp.status_code == 404:
        raise RuntimeError(
            f"Модель '{model}' не найдена. Скачай её сначала: ollama pull {model}"
        )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"].strip()


def patch_content(doc_id: int, text: str) -> None:
    resp = requests.patch(
        f"{PAPERLESS_URL}/api/documents/{doc_id}/",
        headers=_paperless_headers(), json={"content": text}, timeout=30,
    )
    resp.raise_for_status()
    log(f"  OK  content документа {doc_id} обновлён ({len(text)} символов)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc_id", type=int)
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--timeout", type=int, default=180,
                     help="сек ожидания на страницу (первый запуск модели дольше — грузит веса в VRAM)")
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX,
                     help="размер контекста для Ollama (меньше -> меньше VRAM, по умолчанию 4096)")
    ap.add_argument("--patch", action="store_true")
    args = ap.parse_args()

    check_ollama_running()

    pdf_path = get_or_download(args.doc_id)

    log("  ..  рендерим страницы PDF в картинки")
    images = render_pages_base64(pdf_path)
    log(f"       {len(images)} стр., модель: {args.model}")

    page_texts = []
    for i, img in enumerate(images):
        log(f"  ..  OCR страницы {i + 1}/{len(images)}")
        page_texts.append(ocr_page_with_ollama(img, args.model, args.timeout, args.num_ctx))

    full_text = "\n\n---\n\n".join(page_texts)

    log("\n" + "=" * 70)
    log(full_text)
    log("=" * 70 + "\n")

    out_path = str(RESULTS_DIR / f"doc_{args.doc_id}_local_vision.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    log(f"  OK  сохранено локально: {out_path}")

    if args.patch:
        patch_content(args.doc_id, full_text)
    else:
        log("  i   content НЕ записан — перезапусти с --patch, если результат устраивает")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        log(__doc__)
        sys.exit(0)
    main()
