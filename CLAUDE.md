# addrmatch — контекст проекта

Проект под управлением **GRACE 4** (`.grace/`). Перед любой правкой кода — `grace-bootstrap` → нужный `grace-*` скилл. Правила и разметка — `AGENTS.md`.

## Окружение
- Разработка: Windows 11, PowerShell/bash, Python 3.11+, CPU. Кодировка файлов UTF-8 явно.
- Деплой-таргет фазы A: чистый venv проверяющих (`pip install -r requirements.txt && python run.py ...`). Серверов и SSH нет.
- Данные ТЗ: `test_task_adress_match/data/*.jsonl`. ГАР (`gar_xml.zip`, 53 ГБ) качается в `F:\Downloads`; выборочная распаковка в фазе B.
- Ponytail: `lite` (поднимать до `full` только на `grace-execute`).

## Документы
- `docs/concept.md` — концепция v0.2 (нормативная для spec).
- `docs/approaches.md` — варианты решений; `docs/triz-resolution.md` — разрешённые противоречия.
- ТЗ: `test_task_adress_match/README.md`.
