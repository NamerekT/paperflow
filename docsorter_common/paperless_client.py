"""
paperless_client.py — вся работа с Paperless-ngx REST API в одном месте.

Раньше эти же ~15 функций были продублированы (с мелкими расхождениями!) в
DocSorter.py и apply_ocr_cascade.py — например, фикс "/api/tasks/ теперь
пагинирован" сначала попал только в один файл. PaperlessClient — общий
источник правды для обоих скриптов.
"""

import hashlib
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

from . import config

# Дефолтный цвет новых тегов в Paperless — блёклый серый, все теги в архиве
# выглядят одинаково и не отличить на глаз друг от друга. Вместо этого —
# яркая, хорошо различимая палитра без пастельных/приглушённых тонов. Цвет
# выбирается ДЕТЕРМИНИРОВАННО по имени тега (хэш, не рандом), чтобы:
#  - один и тот же тег всегда получал один и тот же цвет между запусками;
#  - разовая перекраска существующих тегов (см. set_tag_colors.py) не
#    расходилась с тем, что достаётся новым тегам "на лету" при
#    классификации (get_or_create_id).
TAG_COLOR_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#469990", "#9a6324", "#800000", "#808000",
    "#000075", "#ff4500", "#daa520", "#2e8b57",
]


def tag_color_for_name(name: str) -> str:
    idx = int(hashlib.md5(name.strip().lower().encode("utf-8")).hexdigest(), 16) % len(TAG_COLOR_PALETTE)
    return TAG_COLOR_PALETTE[idx]


class PaperlessClient:
    def __init__(self, base_url: str | None = None, api_token: str | None = None,
                 log_fn=print):
        self.base_url = (base_url or config.PAPERLESS_URL).rstrip("/")
        self.api_token = api_token or config.PAPERLESS_API_TOKEN
        self.log = log_fn
        # Реальная версия API определяется динамически (detect_api_version) —
        # разные инстансы Paperless-ngx поддерживают разные диапазоны, а
        # запрос несуществующей версии даёт 406 Not Acceptable. Начиная с
        # API v7 формат select-полей и значение select у документа — это id
        # опции (строка); до v7 — индекс (число), а сами опции — просто
        # список строк, а не список объектов {id,label}.
        self.api_version = 1

    # ── low-level ────────────────────────────────────────────────────────

    def _fix_next_url(self, next_url: str | None) -> str | None:
        """Paperless сам строит абсолютные ссылки пагинации ("next") — на
        основе того, что Django думает о своей схеме/хосте. Если он сидит
        за TLS-терминирующим прокси (Tailscale Serve/nginx/Caddy) и не
        настроен доверять заголовку X-Forwarded-Proto, Django видит только
        голый HTTP-запрос от прокси и строит "next" как http://...:80 —
        даже если САМ запрос, на который пришёл этот ответ, шёл по HTTPS.
        requests потом честно пытается достучаться на 80-й порт, который
        снаружи никто не слушает — ConnectionRefusedError, который на
        первый взгляд похож на обрыв сети/Tailscale, хотя сеть тут ни при
        чём (реальный случай — Paperless 3.0.4 за Tailscale Serve, 2
        августа 2026). Подменяем схему и хост на те, что мы сами
        используем в self.base_url — доверяем от ответа сервера только
        путь и query, не схему/хост."""
        if not next_url:
            return None
        base = urlsplit(self.base_url)
        parts = urlsplit(next_url)
        return urlunsplit((base.scheme, base.netloc, parts.path, parts.query, parts.fragment))

    def headers(self, versioned: bool = False) -> dict:
        h = {"Authorization": f"Token {self.api_token}"}
        if versioned:
            h["Accept"] = f"application/json; version={self.api_version}"
        return h

    def detect_api_version(self) -> int:
        """Дёргает любой authenticated-эндпоинт и читает заголовок ответа
        X-Api-Version — так сервер сообщает максимальную версию API, которую
        он поддерживает. Если заголовка нет (очень старый Paperless) — 1.
        Обновляет self.api_version и возвращает его."""
        resp = requests.get(f"{self.base_url}/api/", headers=self.headers(), timeout=30)
        resp.raise_for_status()
        version = int(resp.headers.get("X-Api-Version", "1"))
        server_version = resp.headers.get("X-Version", "?")
        self.log(f"  ..  сервер Paperless {server_version}, поддерживаемая версия API: {version}")
        self.api_version = version
        return version

    # ── теги / корреспонденты / типы документов ─────────────────────────

    def list_names(self, resource: str, limit: int = 200) -> list[str]:
        """resource: 'tags' | 'correspondents' | 'document_types'."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/{resource}/",
                headers=self.headers(), params={"page_size": limit}, timeout=20,
            )
            resp.raise_for_status()
            return [r["name"] for r in resp.json().get("results", [])]
        except Exception as e:
            self.log(f"  ..  не удалось получить список {resource} из Paperless: {e}")
            return []

    def fetch_existing_metadata(self) -> dict:
        """Существующие теги/корреспонденты/типы — показываем LLM, чтобы
        переиспользовал вместо создания почти-дублей ('Jobcenter' vs 'jobcenter')."""
        return {
            "tags": self.list_names("tags"),
            "correspondents": self.list_names("correspondents"),
            "document_types": self.list_names("document_types"),
        }

    def get_or_create_id(self, resource: str, name: str) -> int | None:
        """Ищет тег/корреспондента/тип по имени (регистронезависимо),
        создаёт при отсутствии. Возвращает id, или None при ошибке/пустом имени."""
        if not name:
            return None
        try:
            resp = requests.get(
                f"{self.base_url}/api/{resource}/",
                headers=self.headers(), params={"name__iexact": name}, timeout=20,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                return results[0]["id"]

            payload = {"name": name, "matching_algorithm": 0}
            if resource == "tags":
                # Иначе новый тег создаётся дефолтным серым — не отличить
                # на глаз от остальных. См. TAG_COLOR_PALETTE выше.
                payload["color"] = tag_color_for_name(name)
            resp = requests.post(
                f"{self.base_url}/api/{resource}/",
                headers=self.headers(),
                # matching_algorithm: 0 = "Automatic" — ручные загрузки через
                # UI Paperless продолжают получать авто-подсказки метаданных
                # от его собственного классификатора, как и остальной архив.
                #
                # correspondent/document_type, переданные явно при загрузке,
                # полностью обходят классификатор (set_correspondent /
                # set_document_type возвращаются сразу, если уже выставлено).
                # tags этой защиты не имеют — set_tags() всегда прогоняет
                # matching(), так что к нашим тегам могут добавиться ещё и
                # от встроенного классификатора Paperless — это ожидаемо, не
                # баг, и шум будет уменьшаться по мере роста размеченного
                # архива.
                json=payload,
                timeout=20,
            )
            resp.raise_for_status()
            new_id = resp.json()["id"]
            self.log(f"       создан новый {resource[:-1]}: {name}")
            return new_id
        except Exception as e:
            self.log(f"  !!  Paperless {resource} '{name}': {e}")
            return None

    # ── кастомные поля ───────────────────────────────────────────────────

    def fetch_all_custom_fields(self) -> dict:
        """Возвращает {name: {"id":.., "data_type":.., "extra_data":..}}."""
        fields = {}
        url = f"{self.base_url}/api/custom_fields/"
        while url:
            resp = requests.get(url, headers=self.headers(versioned=True), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for f in data["results"]:
                fields[f["name"]] = f
            url = self._fix_next_url(data.get("next"))
        return fields

    def ensure_custom_fields(self, defs: dict) -> dict:
        """defs: {name: {"data_type":.., "extra_data":.. | None}}.
        Создаёт недостающие поля, возвращает {name: field_dict} для всех."""
        existing = self.fetch_all_custom_fields()
        result = {}
        for name, spec in defs.items():
            if name in existing:
                result[name] = existing[name]
                self.log(f"  ..  поле уже есть: '{name}' (id={existing[name]['id']}, {existing[name]['data_type']})")
                continue
            payload = {"name": name, "data_type": spec["data_type"]}
            if spec.get("extra_data") is not None:
                payload["extra_data"] = spec["extra_data"]
            resp = requests.post(
                f"{self.base_url}/api/custom_fields/",
                headers=self.headers(versioned=True), json=payload, timeout=30,
            )
            resp.raise_for_status()
            created = resp.json()
            result[name] = created
            self.log(f"  OK  создано поле: '{name}' (id={created['id']})")
        return result

    def select_option_value(self, field: dict, label: str):
        """Значение для custom_fields[].value у select-поля.
        С API v7+ сервер отдаёт select_options как список {id,label} и хочет
        в качестве значения id (строку). До v7 — список просто строк-лейблов,
        и значение — это ИНДЕКС нужного лейбла в списке (число)."""
        options = field["extra_data"]["select_options"]
        if self.api_version >= 7:
            for opt in options:
                if opt["label"] == label:
                    return opt["id"]
            raise ValueError(f"Опция '{label}' не найдена в поле '{field['name']}': {options}")
        else:
            for idx, opt_label in enumerate(options):
                if opt_label == label:
                    return idx
            raise ValueError(f"Опция '{label}' не найдена в поле '{field['name']}': {options}")

    @staticmethod
    def custom_field_value(doc_meta: dict, field_id):
        """Значение конкретного кастомного поля в ответе get_document(), или
        None, если поле не проставлено на этом документе."""
        for cf in doc_meta.get("custom_fields") or []:
            if cf.get("field") == field_id:
                return cf.get("value")
        return None

    # ── документы ─────────────────────────────────────────────────────────

    def get_document(self, doc_id: int) -> dict:
        resp = requests.get(
            f"{self.base_url}/api/documents/{doc_id}/",
            headers=self.headers(versioned=True), timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def fetch_all_document_ids(self) -> list[int]:
        ids = []
        url = f"{self.base_url}/api/documents/?page_size=200&fields=id"
        while url:
            resp = requests.get(url, headers=self.headers(versioned=True), timeout=60)
            resp.raise_for_status()
            data = resp.json()
            ids.extend(d["id"] for d in data["results"])
            url = self._fix_next_url(data.get("next"))
        return ids

    def fetch_documents_fields(self, fields: list[str]) -> list[dict]:
        """Пагинированный GET /api/documents/ с выбором конкретных полей
        (?fields=id,title,...) — тот же трюк, что fetch_all_document_ids()
        уже использует для одного поля "id", только на произвольный набор.
        Экономит трафик по сравнению с get_document() на каждый документ по
        отдельности.

        ВАЖНО: "checksum" сюда передавать бесполезно, и дело не только в
        API. Во-первых, DocumentSerializer Paperless-ngx
        (src/documents/serialisers.py) вообще не объявляет это поле в
        Meta.fields, ни для одной версии API — запрос ?fields=checksum
        молча игнорируется. Во-вторых, даже если бы отдавало — не помогло
        бы искать дубли: в documents/models.py поле checksum объявлено с
        unique=True (UNIQUE-констрейнт в Postgres), так что два документа
        с одинаковым checksum структурно не могут существовать
        одновременно (Paperless сам блокирует повторную заливку байт-в-
        байт идентичного файла). find_duplicates.py поэтому ищет
        визуально похожие документы через perceptual hash (phash)
        страниц — см. докстринг в начале того файла."""
        docs: list[dict] = []
        fields_param = ",".join(fields)
        url = f"{self.base_url}/api/documents/?page_size=200&fields={fields_param}"
        while url:
            resp = requests.get(url, headers=self.headers(versioned=True), timeout=60)
            resp.raise_for_status()
            data = resp.json()
            docs.extend(data["results"])
            url = self._fix_next_url(data.get("next"))
        return docs

    # Реальный случай (3 файла-выписки ПриватБанка, 2 августа 2026):
    # "Unsupported mime type application/octet-stream" от Paperless — файлы
    # физически НЕ начинаются с "%PDF-". Вместо этого первые байты —
    # ASN.1 DER с OID 1.2.840.113549.1.7.2 (PKCS#7 SignedData): это PDF,
    # обёрнутый в контейнер электронной подписи (украинский КЕП), просто
    # сохранённый с расширением .pdf без снятия обёртки. Сам PDF внутри
    # цел и полностью читается — "%PDF-" был на 71-м байте во всех трёх,
    # с нормальным "%%EOF" дальше в файле (после которого ещё идут байты
    # сертификата/подписи). И python-magic у Paperless, и обычные ридеры
    # смотрят на самые первые байты файла — видят ASN.1, а не "%PDF-", и
    # честно репортят application/octet-stream, хотя контент валиден.
    _PDF_MAGIC = b"%PDF-"
    _PDF_EOF_MARKER = b"%%EOF"

    def _unwrap_pdf_if_needed(self, file_path: Path) -> Path:
        """Если file_path уже начинается с "%PDF-" — возвращает как есть.
        Если "%PDF-" найден ГДЕ-ТО внутри файла (не с байта 0) — вырезает
        настоящий PDF (от "%PDF-" до последнего "%%EOF" в файле, если он
        есть и идёт после начала PDF, иначе до конца файла) во временный
        файл рядом с оригиналом и возвращает путь к нему; вызывающий код
        отвечает за удаление временного файла после отправки. Если
        "%PDF-" не встречается вообще — файл действительно не PDF/битый,
        возвращает path как есть, пусть падает с понятной ошибкой Paperless,
        как и раньше."""
        data = file_path.read_bytes()
        if data.startswith(self._PDF_MAGIC):
            return file_path

        pdf_start = data.find(self._PDF_MAGIC)
        if pdf_start == -1:
            return file_path

        pdf_end = data.rfind(self._PDF_EOF_MARKER)
        if pdf_end == -1 or pdf_end < pdf_start:
            pdf_bytes = data[pdf_start:]
        else:
            pdf_bytes = data[pdf_start:pdf_end + len(self._PDF_EOF_MARKER)]

        tmp_path = file_path.with_name(f".{file_path.stem}__unwrapped.pdf")
        tmp_path.write_bytes(pdf_bytes)
        self.log(f"       {file_path.name}: не начинается с %PDF- (похоже на PKCS#7-обёртку электронной "
                 f"подписи) — распаковал реальный PDF из середины файла ({len(pdf_bytes)} байт)")
        return tmp_path

    def download_original(self, doc_id: int, dest: Path) -> Path:
        if dest.exists():
            return dest
        resp = requests.get(
            f"{self.base_url}/api/documents/{doc_id}/download/",
            headers=self.headers(), params={"original": "true"}, timeout=60,
        )
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest

    def upload_document(self, file_path, *, title: str | None = None,
                         correspondent_id: int | None = None,
                         document_type_id: int | None = None,
                         tag_ids: list | None = None,
                         created: str | None = None) -> str:
        """POST /api/documents/post_document/. Все именованные параметры
        опциональны — вызывающий может залить голый файл (apply_ocr_cascade,
        классификация идёт отдельным PATCH позже) или сразу с метаданными
        (DocSorter, классификация уже готова до загрузки).
        Возвращает task_id (UUID) консьюминга."""
        file_path = Path(file_path)
        data = []
        if title:
            data.append(("title", title))
        if created:
            data.append(("created", created))
        if correspondent_id:
            data.append(("correspondent", str(correspondent_id)))
        if document_type_id:
            data.append(("document_type", str(document_type_id)))
        for tag_id in (tag_ids or []):
            if tag_id:
                data.append(("tags", str(tag_id)))   # repeated field — НЕ строка через запятую

        # см. _unwrap_pdf_if_needed() — некоторые файлы (замечено на
        # банковских выписках) физически не начинаются с "%PDF-", хотя
        # реальный PDF внутри цел; без распаковки Paperless отвечает
        # "Unsupported mime type application/octet-stream" ещё до того,
        # как вообще успевает открыть файл.
        upload_path = self._unwrap_pdf_if_needed(file_path)
        try:
            with open(upload_path, "rb") as f:
                resp = requests.post(
                    f"{self.base_url}/api/documents/post_document/",
                    headers=self.headers(),
                    data=data or None,
                    # Имя файла — оригинальное (file_path.name), даже если
                    # реально отправляем распакованную временную копию:
                    # Paperless должен увидеть человекочитаемое имя, а не
                    # служебное ".xxx__unwrapped.pdf".
                    files={"document": (file_path.name, f, "application/pdf")},
                    timeout=120,
                )
        finally:
            if upload_path != file_path:
                upload_path.unlink(missing_ok=True)
        resp.raise_for_status()
        return resp.json()

    def wait_for_task(self, task_id: str, timeout: int = 120) -> int | None:
        """Опрашивает /api/tasks/ до завершения консьюминга. Не критично для
        самой загрузки — она уже прошла успешно до вызова этого метода.

        NOTE: этот инстанс Paperless отдаёт статусы в нижнем регистре
        ("success"/"failure"/"pending"/"started"), не в верхнем Celery-стиле
        ("SUCCESS"/"FAILURE") — сравнение с верхним регистром тихо никогда
        не совпадало. Поле со списком id называется related_document_ids,
        не related_document.

        NOTE 2: /api/tasks/ может вернуть либо голый список [...] (старые
        версии), либо пагинированный объект {"count":.., "results":[...]}
        (Paperless 3.0.4 / API v10, подтверждено на этом инстансе). Прямое
        resp.json()[0] на пагинированном ответе кидало KeyError: 0 на каждой
        попытке — цикл крутился до таймаута, хотя файл уже был обработан."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"{self.base_url}/api/tasks/",
                    headers=self.headers(), params={"task_id": task_id}, timeout=20,
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results") if isinstance(data, dict) else data
                if results:
                    status = (results[0].get("status") or "").lower()
                    if status == "success":
                        related = results[0].get("related_document_ids") or []
                        return related[0] if related else None
                    if status == "failure":
                        self.log(f"  !!  Paperless не смог обработать файл: {results[0].get('result_data')}")
                        return None
            except Exception as e:
                self.log(f"  ..  жду обработку Paperless: {e}")
            time.sleep(3)
        self.log("  !!  таймаут ожидания обработки Paperless")
        return None

    def upload_and_wait(self, file_path, timeout: int = 120, **upload_kwargs) -> int | None:
        """upload_document() + wait_for_task() одним вызовом."""
        task_id = self.upload_document(file_path, **upload_kwargs)
        return self.wait_for_task(task_id, timeout=timeout)

    def edit_pdf_rotate_pages(self, doc_id: int, page_rotations: dict[int, int], timeout: int = 120) -> None:
        """Поворачивает КОНКРЕТНЫЕ страницы уже загруженного документа "на
        месте" — через /api/documents/edit_pdf/ (не /api/documents/rotate/,
        тот поворачивает только ВЕСЬ документ целиком одним углом).
        update_document=True — правит СУЩЕСТВУЮЩИЙ документ (тот же id, тот
        же PATCH-история), а не создаёт вторую копию/версию — так это
        подтверждено схемой API (см. README, грабли).

        page_rotations — {номер_страницы (с 1): градусы по часовой
        стрелке} — ОТНОСИТЕЛЬНАЯ поправка ("повернуть ещё на X°"), НЕ
        абсолютное значение /Rotate — подтверждено эмпирически 6 августа
        2026 (документ с исходным /Rotate=90, отправлен rotate=270, итог
        на сервере — /Rotate=0=90+270 mod 360, страница отрендерилась
        визуально ровно). То, что уже посчитал
        OcrCascade.process_document() (docsorter_common/orientation.py) —
        сырое значение из detect_page_rotation(), не пересчитанное.

        Асинхронно (как и upload) — ответ содержит id фоновой задачи,
        ждём её той же wait_for_task(), что уже используется для загрузки
        (возвращаемый related_document_ids нам тут не нужен, важно только
        дождаться завершения, прежде чем продолжать PATCH content/полей —
        иначе Paperless может ещё пересобирать файл параллельно с нашей
        записью метаданных)."""
        operations = [{"page": page, "rotate": degrees} for page, degrees in sorted(page_rotations.items())]
        resp = requests.post(
            f"{self.base_url}/api/documents/edit_pdf/",
            headers=self.headers(versioned=True),
            json={"documents": [doc_id], "operations": operations, "update_document": True},
            timeout=30,
        )
        resp.raise_for_status()
        task_id = resp.json().get("result")
        if task_id:
            self.wait_for_task(task_id, timeout=timeout)

    def patch_resource(self, resource: str, resource_id: int, payload: dict) -> dict:
        """Голый PATCH на произвольный ресурс (tags/correspondents/documents/…)."""
        resp = requests.patch(
            f"{self.base_url}/api/{resource}/{resource_id}/",
            headers=self.headers(versioned=True), json=payload, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def patch_document(self, doc_id: int, payload: dict) -> dict:
        """Голый PATCH — вызывающий сам решает, что положить в payload
        (content/custom_fields/title/correspondent/document_type/tags/…)."""
        return self.patch_resource("documents", doc_id, payload)

    def patch_custom_fields(self, doc_id: int, new_values: list[dict], extra_payload: dict | None = None) -> dict:
        """PATCH кастомных полей документа, СОХРАНЯЯ значения полей, не
        упомянутых в new_values.

        Реальный случай (5 августа 2026): на этом инстансе Paperless-ngx
        (3.0.4) `PATCH .../documents/{id}/` с `custom_fields=[...]`
        ЗАМЕНЯЕТ весь список custom_fields документа целиком — а не делает
        upsert по присланным полям. Раньше (см. историю в коде) считалось,
        что PATCH — это merge (проверено по исходникам paperless-ngx на
        ветке `main` на GitHub: `DocumentSerializer.update()` вызывает
        `CustomFieldInstance.objects.update_or_create()` для каждого поля
        из присланного списка, не трогая остальные) — но реальное поведение
        ИМЕННО ЭТОГО развёрнутого сервера оказалось другим (либо
        расхождение версий, либо код на `main` относится не к тому пути).
        Практическое следствие бага: раздельные PATCH разных кастомных
        полей одного документа (например, 4 поля каскада одним вызовом, а
        Frist — отдельным, более поздним) стирали друг друга. Документ 3
        (Zahlungserinnerung) реально потерял так все 4 поля каскада, когда
        Frist записался отдельным PATCH позже — обнаружено и почищено
        вручную в этот же день.

        Фикс — сам делаем merge на своей стороне: читаем текущий полный
        список custom_fields, убираем из него поля, которые сейчас
        перезаписываем (по field id), добавляем новые значения, шлём PATCH
        с ПОЛНЫМ получившимся списком. extra_payload — остальные поля PATCH
        (content/title/tags/…), которые ходят тем же запросом, но эту
        особенность не имеют (Paperless явно мержит tags/etc. штатно,
        баг именно в custom_fields)."""
        current = self.get_document(doc_id).get("custom_fields") or []
        new_field_ids = {v["field"] for v in new_values}
        merged = [cf for cf in current if cf["field"] not in new_field_ids] + new_values
        payload = dict(extra_payload or {})
        payload["custom_fields"] = merged
        return self.patch_document(doc_id, payload)

    def list_tags_full(self) -> list[dict]:
        """Все теги целиком (id, name, color, ...), с пагинацией — в отличие
        от list_names('tags'), который отдаёт только имена."""
        tags = []
        url = f"{self.base_url}/api/tags/?page_size=200"
        while url:
            resp = requests.get(url, headers=self.headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            tags.extend(data["results"])
            url = self._fix_next_url(data.get("next"))
        return tags

    def document_url(self, doc_id: int) -> str:
        return f"{self.base_url}/documents/{doc_id}/details"
