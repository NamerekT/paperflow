"""
config.py — общие константы и загрузка family_members.json.

Держим здесь именно КОНФИГУРАЦИЮ (URL-ы, ключи, модели) — не логику. Логика
живёт в paperless_client.py / classification.py / ocr_engine.py.
"""

import json
import os
from pathlib import Path

# Корень проекта — на уровень выше пакета docsorter_common (т.е. рядом с
# DocSorter.py и apply_ocr_cascade.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _require(name: str) -> str:
    """Секрет обязателен и НЕ имеет дефолта в коде.

    Раньше здесь стояли реальные значения (токен Paperless, ключ
    LiteLLM-прокси) — удобно, но означало, что секреты лежат в исходниках
    и попали бы в git при первом же коммите. Теперь единственный источник —
    окружение: файл .env рядом с этим пакетом (не в git, см. .gitignore) на
    Windows-хосте, env_file/environment у docker-compose в контейнере.

    Отсутствие значения — явная ошибка при импорте, а не 401 посреди уже
    начатой обработки документа."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in "
            f"(expected at {PROJECT_ROOT / '.env'}), or pass the variable "
            f"through the container environment."
        )
    return value


# .env читается только если он есть — в контейнере его нет, там переменные
# приходят из docker-compose. python-dotenv не перетирает уже выставленные
# переменные окружения, так что явный export всегда побеждает файл.
try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv(PROJECT_ROOT / ".env")

# Все константы ниже читаются из переменных окружения с фолбэком на
# значения, которые сегодня реально используются на Windows-машине —
# существующие скрипты (DocSorter.py/apply_ocr_cascade.py/...) продолжают
# работать без единой переменной окружения. Нужно для ocr_webhook_service/
# (Docker-контейнер, секреты приходят через env_file, не через код) — см.
# README.md, раздел "Что нужно знать перед докеризацией".
PAPERLESS_URL       = os.environ.get("PAPERLESS_URL", "https://paperless.boreray-balance.ts.net")
PAPERLESS_API_TOKEN = _require("PAPERLESS_API_TOKEN")

# Эскалация плохих страниц (vision OCR) — через собственный self-hosted
# LiteLLM-прокси (docker-compose в D:\Projects\YAML\ollama-openwebui-compose,
# litellm-config/config.yaml), а не напрямую в OpenRouter. Раньше при сбое
# OpenRouter (429/ошибка/зависание без ответа — см. http_utils.py) фолбэк на
# прямой Google AI Studio был написан руками в ocr_cascade.py
# (_gemma_via_openrouter/_gemma_via_google_direct) — теперь этим занимается
# сам роутер LiteLLM: модель "openrouter-gemma-4-31b-it" при сбое сама
# переключается на "google-gemma-4-31b-it" (router_settings.fallbacks в
# litellm-config/config.yaml). Проверено вручную 2 августа 2026 — рабочий
# запрос вернул "provider":"Google AI Studio", то есть фолбэк реально
# сработал. HTTP, не HTTPS — так же доступен и без TLS-оверхеда.
VISION_PROXY_URL   = os.environ.get("VISION_PROXY_URL", "http://lllm.namerek.com")
VISION_PROXY_KEY   = _require("VISION_PROXY_KEY")
VISION_PROXY_MODEL = os.environ.get("VISION_PROXY_MODEL", "openrouter-gemma-4-31b-it")

# Определение ориентации страницы (docsorter_common/orientation.py) — ДО
# OCR, отдельным узким промтом (не смешиваем с задачей распознавания
# текста самой Gemma — риск, что модель хуже справится с обеими задачами
# сразу в одном запросе). Экономит вызов Gemma-эскалации в большинстве
# случаев: если страница была перевёрнута, RapidOCR на уже исправленной
# ориентации, скорее всего, справится сам. google-gemini-3.5-flash-lite —
# та же модель, что Frist, отдельный дешёвый узкий вызов на каждую
# СКАНИРОВАННУЮ страницу (не на born-digital — той поворот не нужен).
ORIENTATION_LLM_MODEL = os.environ.get("ORIENTATION_LLM_MODEL", "google-gemini-3.5-flash-lite")

# Извлечение срока (Frist) из уже готового текста документа
# (docsorter_common/frist_extractor.py) — через тот же self-hosted
# LiteLLM-прокси, что и vision-OCR выше, но ДРУГАЯ модель: прямой Google AI
# Studio (не OpenRouter), отдельный бюджет квоты — не пересекается с
# openrouter-gemma-4-31b-it (тот занят OCR-эскалацией, ограничен ~16k TPM
# на бесплатном OpenRouter). gemini-3.5-flash-lite на бесплатном тарифе
# Google AI Studio даёт 250k TPM (проверено 5 августа 2026).
FRIST_LLM_MODEL = os.environ.get("FRIST_LLM_MODEL", "google-gemini-3.5-flash-lite")

# Классификация документов (title/correspondent/document_type/tags) — через
# тот же self-hosted LiteLLM-прокси, прямой Google AI Studio (раньше шла
# напрямую в OpenRouter на inclusionai/ling-3.0-flash:free — та модель
# регулярно промахивалась с тегами/корреспондентом, примерно 1 документ из
# 5-7). google-gemini-3.6-flash — вершина фолбэк-цепочки в
# litellm-config/config.yaml (router_settings.fallbacks): при 429
# (дневная квота ~20 запросов на бесплатном тарифе исчерпана) роутер сам
# спускается по цепочке 3.6 -> 3.5 -> 3 (preview) -> 2.5 -> 3.5-flash-lite.
#
# CLASSIFIER_ENV=dev (например, во время тестов/отладки на Windows-машине,
# где документы гоняются через классификатор многократно за один присест)
# — обходит всю цепочку и шлёт СРАЗУ на google-gemini-3.5-flash-lite,
# чтобы не жечь скудную дневную квоту умных моделей впустую. В проде
# (ocr_webhook_service, Docker) эту переменную не выставлять — там
# документы редкие (несколько в день), полная цепочка не проблема.
CLASSIFIER_ENV = os.environ.get("CLASSIFIER_ENV", "prod").strip().lower()
CLASSIFIER_LLM_MODEL = (
    "google-gemini-3.5-flash-lite" if CLASSIFIER_ENV == "dev"
    else os.environ.get("CLASSIFIER_LLM_MODEL", "google-gemini-3.6-flash")
)

# Google Tasks — напоминания о сроках (Frist). credentials/ — папка рядом с
# этим пакетом (PROJECT_ROOT), содержит client_secret*.json (OAuth-клиент
# Desktop app, создан вручную в Google Cloud Console) и, после одноразового
# consent (python -m docsorter_common.google_tasks), token.json. НЕ читает
# переменную окружения — как и FAMILY_MEMBERS_FILE, путь всегда рядом с
# пакетом, включая контейнер ocr_webhook_service (папка монтируется
# read-only томом, см. Dockerfile).
GOOGLE_CREDENTIALS_DIR  = PROJECT_ROOT / "credentials"
GOOGLE_TASKS_LIST_NAME  = os.environ.get("GOOGLE_TASKS_LIST_NAME", "DocSorter Fristen")

# Пока false — на старте пишем только поле Frist в Paperless и проверяем
# точность извлечения, не заваливая Google Tasks мусором. Включить (true)
# после ручной проверки на реальных документах (см. README/план).
FRIST_CREATE_TASKS = os.environ.get("FRIST_CREATE_TASKS", "false").strip().lower() not in ("false", "0", "no")

# ocr_webhook_service — единственная оставшаяся реализация OCR-каскада/
# классификации/Frist. WEBHOOK_URL — куда trigger_missing_cascade.py (и
# больше никто — DocSorter.py вообще не знает про вебхук, см. README) шлёт
# POST-запросы; сервис слушает на docker-сети shared-network, наружу не
# торчит, кроме этого одного проброшенного порта на localhost (см.
# docker-compose.yml в D:\Projects\YAML\paperless-ngx). WEBHOOK_SHARED_SECRET
# — тот же секрет, что задан сервису в его environment — без дефолта:
# отсутствие значения должно быть явной ошибкой, а не тихой отправкой
# запроса без токена.
WEBHOOK_URL           = os.environ.get("WEBHOOK_URL", "http://localhost:8002/webhook/ocr-cascade")
WEBHOOK_SHARED_SECRET = os.environ.get("WEBHOOK_SHARED_SECRET")

# FAMILY_MEMBERS_FILE намеренно НЕ читает переменную окружения — путь
# всегда рядом с этим пакетом (PROJECT_ROOT), включая контейнер
# ocr_webhook_service (Dockerfile копирует family_members.json в тот же
# относительный путь). Переопределять его нет практического смысла.
FAMILY_MEMBERS_FILE = PROJECT_ROOT / "family_members.json"

# Папка, куда DocSorter.py сохраняет свежие сканы (queue/ — ещё не
# обработаны) и уже загруженные в Paperless копии (Uploaded to Paperless/ —
# локальный бэкап ПОСЛЕ успешной загрузки). Единая точка правды: DocSorter.py
# строит из неё WATCH_FOLDER/QUEUE_FOLDER/UPLOADED_ARCHIVE_FOLDER, а
# trigger_missing_cascade.py --scan-folder использует её, чтобы ЦЕЛИКОМ
# пропускать эту ветку архива при обходе — файлы в queue/ ещё не готовы, а
# файлы в "Uploaded to Paperless/" уже являются
# документами в Paperless (загружены через DocSorter), повторная заливка их
# с диска создала бы документ-дубликат с тем же содержимым под новым id.
DOCSORTER_WATCH_FOLDER = Path(r"D:\Доки доки\Docs Germany\Unsorted scans")

# Запасной список на случай, если family_members.json отсутствует/битый —
# скрипт не должен падать целиком из-за опечатки в конфиге, просто
# продолжит работать с этим списком и громко предупредит.
_FAMILY_MEMBERS_FALLBACK = ["Mykolai", "Valentyn", "Valentyna", "Dmytro", "Matvii", "Vira", "Familie"]


def load_family_members() -> list[str]:
    """Читает family_members.json (правится руками, без изменений кода).
    "Familie" должен быть в списке — используется как запасной вариант,
    когда LLM не может определить, к кому из семьи относится документ."""
    try:
        data = json.loads(FAMILY_MEMBERS_FILE.read_text(encoding="utf-8"))
        members = data["members"]
        if not isinstance(members, list) or not members:
            raise ValueError("family_members.json: 'members' пуст или не список")
        if "Familie" not in members:
            members = [*members, "Familie"]
        return members
    except Exception as e:
        print(f"  !!  Не удалось прочитать {FAMILY_MEMBERS_FILE} ({e}) — "
              f"использую запасной список: {_FAMILY_MEMBERS_FALLBACK}")
        return list(_FAMILY_MEMBERS_FALLBACK)
