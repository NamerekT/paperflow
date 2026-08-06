"""
docsorter_common
────────────────────────────────────────────────────────────────────────────
Общая логика, которую использует и DocSorter.py (вотчер живых сканов), и
apply_ocr_cascade.py (пакетный прогон каскада RapidOCR→Gemma по всему
архиву). Раньше эти два скрипта дублировали одни и те же функции (Paperless
API, LLM-классификация, RapidOCR-движок) — и с каждой правкой рисковали
разойтись (что уже случилось: тот же баг с /api/tasks/ был пофикшен в одном
файле и не в другом, промт классификации правился в двух местах). Теперь
это один источник правды.

Модули:
  - config.py           — константы (Paperless/OpenRouter/модели) и загрузка
                           family_members.json.
  - paperless_client.py — класс PaperlessClient: вся работа с Paperless REST API.
  - classification.py   — класс Classifier: LLM-промпт + классификация документа.
  - ocr_engine.py        — класс DualRapidOcrEngine: связка latin+eslav RapidOCR.
"""
