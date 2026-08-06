"""
google_tasks.py — напоминания о сроках (Frist) в Google Tasks.

Один Google-аккаунт (подтверждено с пользователем — этого достаточно).
OAuth-клиент (Desktop app) уже создан вручную в Google Cloud Console,
client_secret*.json лежит в config.GOOGLE_CREDENTIALS_DIR. Одноразовый
интерактивный consent (браузер) — см. блок `if __name__ == "__main__"`
внизу файла, запускать вручную ОДИН раз:

    python -m docsorter_common.google_tasks

После этого в той же папке появляется token.json (access+refresh token) —
дальше сервис работает без браузера, тихо рефрешит access token через
refresh_token по мере истечения. Если token.json нет или сломан —
get_tasks_service() падает с понятным сообщением, а НЕ пытается сам
открыть браузер посреди фоновой обработки документа (DocSorter.py/
ocr_webhook_service работают без интерактивной сессии в её обычном смысле —
открывать браузер посреди этого было бы неожиданным и бесполезным).
"""

import glob

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from . import config

SCOPES = ["https://www.googleapis.com/auth/tasks"]

_TOKEN_FILE = config.GOOGLE_CREDENTIALS_DIR / "token.json"


def _find_client_secret_file():
    matches = glob.glob(str(config.GOOGLE_CREDENTIALS_DIR / "client_secret*.json"))
    if not matches:
        raise FileNotFoundError(
            f"Не найден client_secret*.json в {config.GOOGLE_CREDENTIALS_DIR} — "
            "нужен OAuth-клиент (Desktop app) из Google Cloud Console."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"В {config.GOOGLE_CREDENTIALS_DIR} несколько client_secret*.json — "
            f"оставь один: {matches}"
        )
    return matches[0]


def get_tasks_service():
    """Возвращает готовый Google Tasks API client (googleapiclient
    Resource). Тихо рефрешит access token через refresh_token, если он
    истёк, — без браузера."""
    if not _TOKEN_FILE.exists():
        raise RuntimeError(
            f"{_TOKEN_FILE} не найден — запусти один раз вручную "
            "'python -m docsorter_common.google_tasks' (откроется браузер "
            "для одноразового Google-логина), после этого сервис будет "
            "работать без браузера."
        )
    creds = Credentials.from_authorized_user_file(str(_TOKEN_FILE), SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            _TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        else:
            raise RuntimeError(
                f"{_TOKEN_FILE} недействителен и не может быть тихо обновлён "
                "(нет refresh_token) — запусти заново "
                "'python -m docsorter_common.google_tasks'."
            )
    return build("tasks", "v1", credentials=creds)


def _get_or_create_tasklist_id(service) -> str:
    """Ищет список задач config.GOOGLE_TASKS_LIST_NAME (регистронезависимо),
    создаёт при отсутствии — отдельный список, чтобы не мешаться с личными
    задачами в '@default'."""
    resp = service.tasklists().list(maxResults=100).execute()
    for tl in resp.get("items", []):
        if tl["title"].casefold() == config.GOOGLE_TASKS_LIST_NAME.casefold():
            return tl["id"]
    created = service.tasklists().insert(body={"title": config.GOOGLE_TASKS_LIST_NAME}).execute()
    return created["id"]


def create_task(title: str, due_date: str, notes: str | None = None) -> str:
    """due_date — 'YYYY-MM-DD'. Google Tasks API хочет RFC3339-таймстамп в
    поле due, но реально учитывает только дату (время игнорируется
    интерфейсом) — поэтому просто дописываем полночь UTC.
    Возвращает id созданной задачи."""
    service = get_tasks_service()
    tasklist_id = _get_or_create_tasklist_id(service)
    body = {"title": title, "due": f"{due_date}T00:00:00.000Z"}
    if notes:
        body["notes"] = notes
    created = service.tasks().insert(tasklist=tasklist_id, body=body).execute()
    return created["id"]


if __name__ == "__main__":
    config.GOOGLE_CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    client_secret_file = _find_client_secret_file()
    print(f"  ..  использую OAuth-клиент: {client_secret_file}")
    flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
    # access_type='offline' — обязательно, иначе не будет refresh_token и
    # придётся заново логиниться в браузере при каждом истечении access
    # token (обычно ~1 час) — неприемлемо для фонового сервиса.
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    _TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    print(f"  OK  сохранено: {_TOKEN_FILE}")

    print("  ..  проверяю: создаю тестовую задачу...")
    task_id = create_task(
        "DocSorter — тестовая задача",
        __import__("datetime").date.today().isoformat(),
        notes="Эту задачу можно удалить — проверка настройки Google Tasks.",
    )
    print(f"  OK  создана тестовая задача (id={task_id}) в списке '{config.GOOGLE_TASKS_LIST_NAME}' — "
          f"проверь в Google Tasks и удали её вручную.")
