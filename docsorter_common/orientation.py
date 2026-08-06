"""
orientation.py — определение ориентации отсканированной страницы
(0/90/180/270°) ДО OCR, отдельным узким промтом.

Реальная проблема: сканер (у пользователя — принтер-сканер, NAPS2) не
переворачивает автоматически неправильно вложенные листы (типичный случай
— чеки). NAPS2 умеет только Auto Deskew (лёгкий перекос на пару градусов),
автоопределения поворота 90/180/270° у него нет (открытый issue в их
GitHub с 2023 года, так и не реализован).

Почему ОТДЕЛЬНЫЙ узкий промт, а не одна задача вместе с
gemma_vision_ocr()/PROMPT_VISION: если дать модели 2 разные задачи в одном
запросе (распознать И определить поворот), выше риск, что она хуже
справится с обеими сразу. Раздельно — надёжнее для каждой по отдельности.

Почему ДО OCR, а не как реакция на низкий confidence RapidOCR: чистое
"попробовать все 4 поворота и сравнить confidence" ненадёжно само по
себе — короткий/скудный текст (типичный чек) может дать обманчиво похожие
скоры на разных поворотах. Vision-модель, которая просто СМОТРИТ на
картинку, справляется с этим гораздо надёжнее, чем косвенный вывод из
чисел уверенности OCR.

Побочный эффект, а не просто дополнительная стоимость: если страница была
перевёрнута и её исправили ДО OCR, RapidOCR на уже правильной ориентации,
скорее всего, справится сам — до полноценной (дорогой) vision-транскрипции
в gemma_vision_ocr() дело часто просто не дойдёт. Один дешёвый узкий вызов
экономит один дорогой полный.
"""

import json

from . import config
from .http_utils import post_with_deadline

PROMPT_ORIENTATION = (
    "На этом изображении страницы документа текст может быть перевёрнут "
    "(лист был неправильно вложен при сканировании — боком или вверх "
    "ногами). Определи, на сколько градусов ПО ЧАСОВОЙ СТРЕЛКЕ нужно "
    "повернуть изображение, чтобы текст читался нормально — слева "
    "направо, сверху вниз, как обычная страница. Если текст уже читается "
    "нормально — верни 0. Ответь СТРОГО JSON, без markdown и пояснений: "
    '{"rotate": 0} или {"rotate": 90} или {"rotate": 180} или '
    '{"rotate": 270}.'
)


def detect_page_rotation(image_b64: str, log_fn=print, timeout: int = 30, max_retries: int = 3) -> int:
    """Возвращает 0/90/180/270 — на сколько градусов по часовой стрелке
    нужно повернуть страницу. Значение прямо совместимо с
    fitz.Page.set_rotation() (та же система отсчёта — по часовой стрелке,
    как /Rotate в самом PDF).

    При любой неудаче (сеть/парсинг/неожиданный ответ) возвращает 0 —
    эта проверка не должна ронять обработку документа; в худшем случае
    просто не поймает перевёрнутую страницу, как было бы и без неё."""
    payload = {
        "model": config.ORIENTATION_LLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_ORIENTATION},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
        "temperature": 0,
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
        except Exception as e:
            log_fn(f"       .. определение ориентации: {e} — попытка {attempt}/{max_retries}")
            continue

        if resp.status_code >= 400:
            log_fn(f"       .. определение ориентации: HTTP {resp.status_code} — попытка {attempt}/{max_retries}")
            continue

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            rotate = json.loads(content)["rotate"]
            if rotate in (0, 90, 180, 270):
                return rotate
            log_fn(f"       .. определение ориентации: неожиданное значение {rotate!r} — считаю 0")
            return 0
        except Exception as e:
            log_fn(f"       .. определение ориентации: не разобрал ответ ({e}) — попытка {attempt}/{max_retries}")
            continue

    log_fn(f"       .. определение ориентации: не удалось за {max_retries} попыток(и) — считаю 0")
    return 0
