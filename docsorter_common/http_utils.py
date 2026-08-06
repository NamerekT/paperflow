"""
http_utils.py — POST с ЖЁСТКИМ дедлайном по wall-clock времени.

Реальный случай, из-за которого это появилось: POST к бесплатной Gemma
через OpenRouter иногда просто зависает — не ошибка, не
requests.exceptions.ReadTimeout, буквально ничего, даже спустя 10+ минут
при выставленном timeout=180. При этом со стороны OpenRouter в "Upstream
Requests" виден 200 к Google AI Studio, а сама Google AI Studio показывает
потраченные токены — то есть генерация где-то состоялась, но HTTP-ответ до
клиента так и не доходит. Похоже, что шлюз держит соединение открытым на
уровне прокси/keep-alive, из-за чего urllib3 не видит "тишины" достаточной
длины, чтобы сработал его собственный read timeout — таймаут requests в
такой ситуации не защита. Тот же риск актуален для ЛЮБОГО HTTP-шлюза перед
LLM (включая наш собственный self-hosted LiteLLM-прокси в ocr_cascade.py),
не только для OpenRouter — отсюда и то, что используется отдельной общей
функцией, а не завязано на конкретный эндпоинт.

post_with_deadline() оборачивает requests.post в отдельный поток и обрывает
ОЖИДАНИЕ результата по wall-clock дедлайну независимо от того, что
происходит на уровне сокета внутри потока. Сам поток Python принудительно
убить не может (GIL/threading этого не позволяют) — он останется висеть в
фоне до естественного разрыва соединения провайдером, но выполнение
скрипта дальше это не блокирует, а зависший запрос перестаёт значить
"скрипт встал намертво".
"""

import concurrent.futures
import time

import requests

# Общий пул на все LLM-вызовы (классификация + vision-OCR) — специально НЕ
# shutdown()'им его: shutdown(wait=True) заблокировал бы ожиданием как раз
# тех зависших воркеров, от которых мы и защищаемся.
_HTTP_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm-http")


def post_with_deadline(url: str, *, headers: dict | None = None, json: dict | None = None,
                        timeout: int = 90, deadline_margin: int = 10):
    """requests.post(...), но с жёстким потолком ожидания = timeout +
    deadline_margin секунд. Бросает concurrent.futures.TimeoutError при
    превышении дедлайна — ловить его вызывающему коду нужно НАРЯДУ с
    requests.exceptions.ReadTimeout (обычный таймаут requests тоже может
    сработать первым и это нормально)."""
    future = _HTTP_POOL.submit(requests.post, url, headers=headers, json=json, timeout=timeout)
    return future.result(timeout=timeout + deadline_margin)


def _startup_step_with_retry(step_name: str, fn, *, interval: int = 5, log_every: int = 1200, log_fn=print):
    """Сетевые шаги старта, зависящие от Paperless (детект версии API,
    кастомные поля) — общая для DocSorter.py и ocr_webhook_service/main.py.

    Paperless крутится в Docker, который поднимается не всегда сразу вместе
    с потребителем (Windows-хост с DocSorter.py стартует по логону; сам
    Docker-стек с вебхуком — по `docker compose up`, без гарантии порядка
    готовности контейнеров) — значит недоступность Paperless на старте это
    ОЖИДАЕМОЕ состояние, а не авария. Поэтому ждём БЕЗ дедлайна (никогда не
    сдаёмся и не роняем процесс сами) — необработанное исключение на
    старте раньше означало ПОЛНОСТЬЮ молчаливую смерть процесса (см.
    историю в DocSorter.py про 2026-08-01 07:32 — pythonw.exe/без консоли
    вообще не оставляет traceback).

    Долбить лог предупреждением на каждую попытку (interval=5с) было бы
    спамом, если Paperless поднимут только через час — поэтому
    предупреждение пишем сразу при первой неудаче, а дальше не чаще раза в
    log_every секунд (по умолчанию 20 минут), пока ждём. Сами попытки
    подключения остаются частыми (interval), чтобы среагировать быстро,
    как только Paperless станет доступен — троттлится только ЛОГ, не сама
    проверка.

    НЕ использовать в одноразовых скриптах (например
    trigger_missing_cascade.py) — там разумнее один раз упасть с понятной
    ошибкой и дать следующему плановому запуску попробовать снова, а не
    ждать бесконечно внутри самого прогона."""
    attempt = 0
    last_logged = None
    while True:
        attempt += 1
        try:
            result = fn()
            if attempt > 1:
                log_fn(f"  ..  {step_name}: доступен (после {attempt} попыток)")
            return result
        except Exception as e:
            now = time.time()
            if last_logged is None or now - last_logged >= log_every:
                log_fn(f"  ..  {step_name}: недоступен ({e}) — жду "
                       f"(Paperless, похоже, ещё не запущен в Docker); попытка {attempt}")
                last_logged = now
            time.sleep(interval)
