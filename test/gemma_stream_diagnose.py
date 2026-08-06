r"""
gemma_stream_diagnose.py
────────────────────────────────────────────────────────────────────────────
Диагностика теории "зависает не сеть/прокси, а сама Gemma" — зацикливается
на генерации и никогда не доходит до стоп-токена.

Стримит ответ от LiteLLM-прокси (docsorter_common/config.py::VISION_PROXY_URL)
постранично и печатает каждый кусок текста по мере поступления, с
таймштампом и паузой от предыдущего куска. Так видно РАЗНИЦУ между:
  - "тишина совсем, ни одного байта" — сетевая/прокси проблема (то, что мы
    чинили http_utils.py::post_with_deadline() раньше);
  - "текст идёт, но модель залипла и повторяет одно и то же" — проблема
    самой генерации, известная болячка vision-LLM на плотном визуально
    повторяющемся контенте (ряды чекбоксов, повторяющаяся структура) при
    жадном (temperature=0) decoding — сбежать из локального повторяющегося
    цикла нечем без капли случайности.

Скрипт сам ловит зацикливание: если один и тот же короткий "хвост" текста
повторяется REPEAT_THRESHOLD раз подряд, ожидание прерывается и громко
печатается диагноз — не нужно ждать реального зависания/лимита токенов
вручную, чтобы понять, что происходит.

ЗАПУСК:
    python gemma_stream_diagnose.py "D:\Доки доки\Docs Germany\Unsorted scans\queue\scan_0001.pdf" 2
    (путь к PDF, номер страницы, с 1)

    Необязательно:
    python gemma_stream_diagnose.py <pdf> <page> --no-mitigations
        — шлёт БЕЗ max_tokens/frequency_penalty (как было раньше), чтобы
          сравнить: если с ними зацикливание не ловится, а без них —
          ловится, это прямое подтверждение теории и того, что фикс в
          ocr_cascade.py реально помогает, а не просто "на всякий случай".
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docsorter_common import config
from docsorter_common.ocr_cascade import PROMPT_VISION

RENDER_DPI = 300
REPEAT_THRESHOLD   = 6    # сколько подряд идентичных "хвостов" текста считать зацикливанием
REPEAT_CHUNK_CHARS = 20   # длина "хвоста", который сверяем на повтор
HARD_CUTOFF_SEC    = 180  # если за это время само не закончилось — считаем зависшим/зацикленным без явного повтора


def log(msg: str) -> None:
    print(msg, flush=True)


def render_page_b64(pdf_path: str, page_num: int, dpi: int = RENDER_DPI) -> str:
    doc = fitz.open(pdf_path)
    try:
        if not (1 <= page_num <= doc.page_count):
            raise ValueError(f"В файле {doc.page_count} стр., запрошена {page_num}")
        page = doc[page_num - 1]
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB)
        return base64.b64encode(pix.tobytes("jpeg", jpg_quality=85)).decode("ascii")
    finally:
        doc.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf_path")
    ap.add_argument("page_num", type=int)
    ap.add_argument("--no-mitigations", action="store_true",
                     help="не слать max_tokens/frequency_penalty — как было ДО фикса, для сравнения")
    args = ap.parse_args()

    log(f"  ..  рендерю страницу {args.page_num} из {args.pdf_path} (dpi={RENDER_DPI})")
    img_b64 = render_page_b64(args.pdf_path, args.page_num)
    log(f"       {len(img_b64) // 1024} КБ base64")

    payload = {
        "model": config.VISION_PROXY_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_VISION},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }],
        "temperature": 0,
        "stream": True,
    }
    if not args.no_mitigations:
        payload["max_tokens"] = 8192
        payload["frequency_penalty"] = 0.3
        log("  ..  с митигациями (max_tokens=8192, frequency_penalty=0.3)")
    else:
        log("  ..  БЕЗ митигаций — как было до фикса")

    log("  ..  отправляю стриминговый запрос...")
    start = time.monotonic()
    resp = requests.post(
        f"{config.VISION_PROXY_URL}/chat/completions",
        headers={"Authorization": f"Bearer {config.VISION_PROXY_KEY}", "Content-Type": "application/json"},
        json=payload, stream=True, timeout=HARD_CUTOFF_SEC,
    )
    resp.raise_for_status()

    full_text = ""
    last_chunk_time = start
    repeat_window: list[str] = []
    stopped_reason = None
    first_byte_time = None

    for raw_line in resp.iter_lines(decode_unicode=True):
        now = time.monotonic()
        if first_byte_time is None:
            first_byte_time = now
            log(f"  ..  первый байт пришёл через {now - start:.1f}с")

        if now - start > HARD_CUTOFF_SEC:
            stopped_reason = f"жёсткий предел {HARD_CUTOFF_SEC}с — обрываю вручную (естественно НЕ закончилось)"
            break
        if not raw_line or not raw_line.startswith("data: "):
            continue
        data_str = raw_line[len("data: "):]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices") or [{}]
        delta = (choices[0].get("delta") or {}).get("content") or ""
        finish_reason = choices[0].get("finish_reason")
        if finish_reason:
            log(f"  ..  finish_reason: {finish_reason}")
        if not delta:
            continue

        gap = now - last_chunk_time
        last_chunk_time = now
        full_text += delta
        log(f"[{now - start:6.1f}s +{gap:4.1f}s] {delta!r}")

        # Детектор зацикливания: смотрим на последний "хвост" накопленного текста.
        tail = full_text[-REPEAT_CHUNK_CHARS:]
        repeat_window.append(tail)
        if len(repeat_window) > REPEAT_THRESHOLD:
            repeat_window.pop(0)
        if len(repeat_window) == REPEAT_THRESHOLD and len(set(repeat_window)) == 1:
            stopped_reason = (f"ЗАЦИКЛИВАНИЕ: последние {REPEAT_THRESHOLD} кусков текста подряд идентичны "
                               f"({tail!r}) — модель застряла в повторе, это НЕ сеть/прокси молчит")
            break

    elapsed = time.monotonic() - start
    log("\n" + "=" * 70)
    if first_byte_time is None:
        log(f"ЗАВИСЛО НА СЕТЕВОМ УРОВНЕ: за {elapsed:.1f}с не пришло ни одного байта — "
            f"это сетевая/прокси проблема, не генерация модели")
    elif stopped_reason:
        log(f"ОСТАНОВЛЕНО ВРУЧНУЮ через {elapsed:.1f}с: {stopped_reason}")
    else:
        log(f"Закончилось само (естественный stop) через {elapsed:.1f}с, {len(full_text)} символов")
    log("=" * 70)
    log(f"\nПолный текст на момент остановки ({len(full_text)} симв.):\n")
    log(full_text)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        log(__doc__)
        sys.exit(0)
    main()
