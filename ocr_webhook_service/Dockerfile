# Собирать из КОРНЯ DocSorter как build context (нужен доступ к
# docsorter_common/ и family_members.json, которые лежат рядом, а не внутри
# ocr_webhook_service/):
#
#   docker build -f ocr_webhook_service/Dockerfile -t ocr-cascade-webhook .
#
# (запускать из D:\Projects\Automations\DocSorter)

FROM python:3.11-slim

WORKDIR /app

# rapidocr зависит от opencv-python (не headless), а тот при импорте (import
# cv2) требует нескольких системных shared-библиотек, которых нет в
# минимальном python:3.11-slim (Debian slim собран без графического стека).
# Без них "from rapidocr import RapidOCR, ..." падает с ImportError
# ("libxcb.so.1: cannot open shared object file") ещё ДО того, как rapidocr
# успевает что-либо сделать — а ocr_engine.py ловит именно ImportError и
# тихо откатывается на старый пакет rapidocr_onnxruntime, которого в
# requirements.txt нет вовсе, из-за чего реальная ошибка (нехватка системных
# библиотек) маскируется под "ModuleNotFoundError: rapidocr_onnxruntime".
# Подтверждено воспроизведением в чистом python:3.11-slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Зависимости отдельным слоем — не переустанавливаются при правке кода.
COPY ocr_webhook_service/requirements.txt ./ocr_webhook_service/requirements.txt
RUN pip install --no-cache-dir -r ocr_webhook_service/requirements.txt

# docsorter_common — общий пакет (тот же, что используют DocSorter.py и
# apply_ocr_cascade.py на Windows-хосте), не дублируется, копируется как есть.
COPY docsorter_common/ ./docsorter_common/
COPY family_members.json ./family_members.json
COPY ocr_webhook_service/ ./ocr_webhook_service/

ENV PYTHONUNBUFFERED=1
# WEBHOOK_SHARED_SECRET обязателен — передаётся через env_file/docker secrets
# при запуске контейнера, не встраивается в образ.

# credentials/ (client_secret*.json + token.json для Google Tasks,
# docsorter_common/google_tasks.py) НЕ копируется в образ — монтируется
# read-only томом при запуске контейнера (docker-compose volumes:
# D:\Projects\Automations\DocSorter\credentials:/app/credentials:ro).
# token.json получить внутри контейнера нельзя (нет браузера для
# интерактивного consent) — используется тот, что уже сгенерирован на
# Windows-хосте через "python -m docsorter_common.google_tasks". Путь
# /app/credentials обязателен: docsorter_common/config.py вычисляет
# GOOGLE_CREDENTIALS_DIR как PROJECT_ROOT/"credentials", а PROJECT_ROOT в
# контейнере — это /app (WORKDIR выше).

EXPOSE 8000

# ${WEBHOOK_PORT:-8000} — чтобы WEBHOOK_PORT реально управлял портом, на
# котором слушает uvicorn, а не только EXPOSE (который сам по себе ничего
# не открывает, это просто документация порта в образе).
CMD ["sh", "-c", "uvicorn ocr_webhook_service.main:app --host 0.0.0.0 --port ${WEBHOOK_PORT:-8000}"]
