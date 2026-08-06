"""
classification.py — LLM-классификация документа (title/correspondent/
document_type/tags/family_member) через self-hosted LiteLLM-прокси (прямой
Google AI Studio, фолбэк-цепочка Gemini-моделей — см.
config.py::CLASSIFIER_LLM_MODEL и litellm-config/config.yaml).

Раньше (до 6 августа 2026) шло напрямую в OpenRouter на
inclusionai/ling-3.0-flash:free — та модель регулярно промахивалась с
тегами/корреспондентом (примерно 1 документ из 5-7).

Промт живёт здесь ОДИН раз — раньше он был продублирован в DocSorter.py и
apply_ocr_cascade.py, и правки (например, "не дублируй correspondent тегом",
"предпочитай существующий document_type") приходилось вносить в двух местах.
"""

import concurrent.futures
import json
import time
from datetime import datetime

import requests

from . import config
from .http_utils import post_with_deadline
from .paperless_client import PaperlessClient


class Classifier:
    def __init__(self, client: PaperlessClient, family_members: list[str] | None = None,
                 model: str | None = None, log_fn=print):
        self.client = client
        self.family_members = family_members or config.load_family_members()
        self.model = model or config.CLASSIFIER_LLM_MODEL
        self.log = log_fn

    # ── промт ────────────────────────────────────────────────────────────

    def build_prompt(self, document_text: str, existing: dict) -> list:
        system_content = f"""You are a document classifier for a Ukrainian migrant family in Germany
(the Tymchenko family). Documents are archived in Paperless-ngx (a document
management system) instead of folders — your job is to assign metadata, not
to pick a folder.

EXISTING TAGS (reuse a good match instead of creating a near-duplicate):
{", ".join(existing.get("tags", [])) or "(none yet)"}

EXISTING CORRESPONDENTS (senders):
{", ".join(existing.get("correspondents", [])) or "(none yet)"}

EXISTING DOCUMENT TYPES:
{", ".join(existing.get("document_types", [])) or "(none yet)"}

FAMILY MEMBERS (pick exactly one — see family_member below):
{", ".join(self.family_members)}

RESPONSE FORMAT (JSON only, no Markdown, no explanations):
{{
  "title": "Short descriptive title, e.g. Bedarfsberechnung Januar 2026",
  "correspondent": "Sender name — reuse an existing one if it matches",
  "document_type": "Document type — reuse an existing one if it matches",
  "tags": ["1 to 4 tags — reuse existing tags where sensible"],
  "family_member": "Exactly one name from FAMILY MEMBERS above",
  "date": "YYYY-MM-DD, the date on the document, or {datetime.now().strftime('%Y-%m-%d')} if unknown"
}}

RULES:
- correspondent: the organization/person who SENT the document (e.g.
  "Jobcenter Wesel", "Agentur für Arbeit (Inkasso-Service)", "Familienkasse").
  Be precise — German authorities that sound similar are often legally
  different institutions (Jobcenter ≠ Agentur für Arbeit ≠ Familienkasse ≠
  Sozialamt). Never guess a "parent" institution name if the letterhead
  names a specific one.
- document_type: the KIND of document (e.g. "Bescheid", "Rückzahlung",
  "Rechnung", "Vertrag", "Ausweis", "Zeugnis"). ALWAYS prefer an EXISTING
  document_type that specifically matches over inventing a new one AND over
  falling back to a more generic existing type. Example: if the letter is
  specifically demanding repayment of an overpayment and "Rückzahlung"
  already exists in EXISTING DOCUMENT TYPES, use "Rückzahlung" — do not use
  a more generic type like "Bescheid" or "Schreiben" just because it's also
  technically true; pick the MOST SPECIFIC existing type that fits.
- tags: describe the SUBJECT/TOPIC of the document (e.g. "Bürgergeld",
  "Miete", "Schule", "2026") — NOT who sent it (that's `correspondent`) and
  NOT the kind of document (that's `document_type`). Do NOT add a tag that
  just repeats or aliases the correspondent's organization name (e.g. don't
  tag "Jobcenter" when correspondent is "Agentur für Arbeit
  (Inkasso-Service)" — besides being redundant, those are different
  institutions, don't conflate them).
- family_member: who the document is addressed to, or who it's about —
  e.g. whose Bürgergeld application, whose diploma, whose passport. This is
  about the FAMILY MEMBER, not the sender. If the document is a shared/
  household matter (rent, car registration) or you genuinely can't tell,
  use "Familie".
- Prefer reusing an existing correspondent/document_type/tag over inventing
  a near-duplicate (e.g. if "Jobcenter Wesel" exists, don't create
  "Jobcenter-Wesel" or "jobcenter wesel").
- title: no file extension, no forbidden characters, max ~80 characters."""

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"DOCUMENT TEXT:\n{document_text[:10000]}"},
        ]

    def fallback_result(self) -> dict:
        """Дефолт, когда LLM недоступна или вернула невалидный JSON."""
        return {
            "title": f"{datetime.now():%Y-%m-%d_%H-%M-%S}_Unbekannt",
            "correspondent": "Unbekannt",
            "document_type": "Sonstiges",
            "tags": ["Unsortiert"],
            "family_member": "Familie",
            "date": datetime.now().strftime("%Y-%m-%d"),
        }

    # ── вызов LLM ────────────────────────────────────────────────────────

    def classify_with_llm(self, messages: list, max_retries: int = 4, timeout: int = 120) -> dict:
        """С ретраями на 429/5xx/таймаут — раньше их не было. Идёт через
        self-hosted LiteLLM-прокси (тот же, что vision-OCR эскалация и
        Frist-извлечение) — на google-gemini-3.6-flash с фолбэк-цепочкой
        вниз по моделям Gemini при 429 (дневная квота ~20 запросов на
        бесплатном тарифе, см. litellm-config/config.yaml). Раньше запрос
        шёл напрямую в OpenRouter, в обход прокси — если та модель словит
        429, resp.raise_for_status() кидал HTTPError прямо из classify(),
        необработанным, и весь документ в apply_ocr_cascade.py/DocSorter.py
        падал с ошибкой уже ПОСЛЕ дорогого OCR-каскада — то есть выглядело
        как "после OCR/vision классификатор не работает". Ретраи здесь —
        подстраховка НАД фолбэком роутера (сеть/таймаут ДО ответа роутера),
        сам 429 фолбэк-цепочки роутер разруливает молча, сюда долетает уже
        либо успех, либо реальный отказ всей цепочки."""
        for attempt in range(1, max_retries + 1):
            try:
                resp = post_with_deadline(
                    f"{config.VISION_PROXY_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {config.VISION_PROXY_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.5,
                        "reasoning": {"effort": "none", "enabled": False},
                    },
                    timeout=timeout,
                )
            except requests.exceptions.ReadTimeout:
                self.log(f"  !!  LLM не ответил за {timeout}с, попытка {attempt}/{max_retries}")
                continue
            except concurrent.futures.TimeoutError:
                # requests' timeout не сработал — тот же зависший-без-ответа
                # апстрим OpenRouter, что и у Gemma в ocr_cascade.py, только
                # теперь и на текстовой классификации. См. http_utils.py.
                self.log(f"  !!  LLM завис без ответа и без ошибки за {timeout}с "
                         f"(похоже на зависший апстрим OpenRouter) — попытка {attempt}/{max_retries}")
                continue
            except requests.exceptions.ConnectionError as e:
                self.log(f"  !!  LLM недоступен: {e} — попытка {attempt}/{max_retries}")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
                return self.fallback_result()

            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 10 * attempt))
                self.log(f"  ..  429 Too Many Requests от LLM, жду {wait:.0f}с (попытка {attempt}/{max_retries})")
                if attempt < max_retries:
                    time.sleep(wait)
                    continue
                self.log("  !!  LLM: rate limit после всех попыток — использую fallback")
                return self.fallback_result()

            if resp.status_code >= 500:
                self.log(f"  !!  LLM HTTP {resp.status_code}, попытка {attempt}/{max_retries}: {resp.text[:300]}")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
                return self.fallback_result()

            resp.raise_for_status()  # прочие 4xx — реальная ошибка запроса, ретраить бессмысленно

            data = resp.json()
            content = data["choices"][0]["message"].get("content")
            if not content:
                self.log(f"  !!  LLM вернул пустой ответ: {json.dumps(data, ensure_ascii=False)[:500]}")
                return self.fallback_result()

            raw = content.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                self.log(f"  !!  Невалидный JSON от LLM: {e} | raw: {raw[:300]}")
                return self.fallback_result()

        return self.fallback_result()

    def _resolve_family_member(self, raw_value: str) -> str:
        """Сопоставляет ответ LLM с канонической записью из family_members.json.
        LLM не всегда возвращает значение дословно как в списке (например,
        может дописать фамилию, поменять регистр или порядок слов, особенно
        если по промту она же видела теги с полным именем в EXISTING TAGS) —
        поэтому сначала точное совпадение (без учёта регистра/пробелов),
        затем совпадение по первому слову (имени), и только если ничего не
        подошло — откат на 'Familie' с явным логом, чтобы это было видно,
        а не тихо теряло правильный ответ модели."""
        normalized = raw_value.strip()

        # 1. точное совпадение без учёта регистра
        for member in self.family_members:
            if member.casefold() == normalized.casefold():
                return member

        # 2. совпадение по первому слову (имени) — например, LLM вернула
        # "Mykolai Tymchenko", а в списке "Mykolai Tymchenko" тоже есть, но
        # с иным порядком слов/пробелами, или просто "Mykolai"
        first_word = normalized.split()[0].casefold() if normalized.split() else ""
        for member in self.family_members:
            if member.split()[0].casefold() == first_word:
                if member != normalized:
                    self.log(f"       LLM вернула '{raw_value}' — сопоставил с '{member}' по имени")
                return member

        self.log(f"       LLM выбрала незнакомого члена семьи '{raw_value}' — использую 'Familie'")
        return "Familie"

    # ── полный цикл: промт → LLM → get-or-create id-шники ───────────────

    def classify(self, content_text: str) -> dict:
        """Возвращает {"title","correspondent_id","document_type_id",
        "tag_ids","date"}.

        "Personal" + ровно один тег с именем члена семьи добавляются
        ВСЕГДА, программно — не полагаемся на то, что LLM не забудет их
        сама (то же самое, из-за чего модель то путала Jobcenter с
        Agentur für Arbeit, то теряла существующий document_type — нельзя
        доверять LLM детерминированные гарантии)."""
        existing = self.client.fetch_existing_metadata()
        messages = self.build_prompt(content_text, existing)
        result = self.classify_with_llm(messages)

        fallback      = self.fallback_result()
        title         = result.get("title") or fallback["title"]
        correspondent = result.get("correspondent") or fallback["correspondent"]
        document_type = result.get("document_type") or fallback["document_type"]
        tags          = result.get("tags") or fallback["tags"]
        doc_date      = result.get("date") or fallback["date"]

        family_member = self._resolve_family_member(result.get("family_member") or fallback["family_member"])

        all_tag_names = list(dict.fromkeys([*tags, "Personal", family_member]))

        return {
            "title": title,
            "correspondent_id": self.client.get_or_create_id("correspondents", correspondent),
            "document_type_id": self.client.get_or_create_id("document_types", document_type),
            "tag_ids": [tid for tid in (self.client.get_or_create_id("tags", t) for t in all_tag_names) if tid],
            "date": doc_date,
        }
