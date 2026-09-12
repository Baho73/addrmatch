# START_MODULE_CONTRACT
#   PURPOSE: Симулировать дозапрос (docs/concept.md §6, H12): после ask_house/ask_city подать
#            второй match() с верным слотом-id и проверить, что диалог разрешается.
#   SCOPE: CLI-скрипт над Matcher; не трогает addrmatch/*. Только позитивные строки labeled (у
#          негативов нет истинного etalon_id, сравнивать не с чем).
#   DEPENDS: M-MATCHER
#   LINKS: V-M-MATCHER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   load_jsonl - чтение JSONL (UTF-8), как в run.py
#   simulate - первый+второй match() по позитивным строкам -> счётчики H12
#   main - CLI: --adresses/--etalon/--N, печать JSON, sys.exit(0|1) по целям H12
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-007: simulate_dialog.py — H12: ask_house -> второй match(slots=
#   {"street": {"id": leader_street_id}, "city": {"id": leader_city_id}}, asked_slot="house",
#   raw=str(истинный дом)) считает долю answer/answer_soft с верным candidates[0]; ask_city ->
#   второй match(slots={"city": {"id": истинный city_id}}, тот же raw, asked_slot="city") считает
#   долю answer/answer_soft/confirm с верным candidates[0]. n<5 в любой группе - "мало данных" в
#   лог (stderr), не провал; цель обеих долей >= 0.95 - MustPassCommand плана (exit 0/1).
# END_CHANGE_SUMMARY

"""tools/simulate_dialog.py — симуляция дозапроса H12 (docs/concept.md §6, plan.xml T-007)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# tools/ не пакет — добавляем корень репозитория в sys.path (как train_ranker.py/errors_report.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from addrmatch import Matcher  # noqa: E402
from addrmatch.normalizer import normalize  # noqa: E402

logger = logging.getLogger("addrmatch.simulate_dialog")

GOAL = 0.95
MIN_N = 5  # ниже - "мало данных" (не статистика), не провал (T-007 п.4)


# START_CONTRACT: load_jsonl
#   PURPOSE: Прочитать JSONL-файл в список словарей (тот же контракт, что run.py.load_jsonl).
#   INPUTS: { path: str - путь к файлу в кодировке UTF-8 }
#   OUTPUTS: { list[dict] - строки файла как объекты, пустые строки пропущены }
#   SIDE_EFFECTS: чтение файла с диска
# END_CONTRACT: load_jsonl
def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# START_CONTRACT: simulate
#   PURPOSE: Прогнать первый+второй match() по позитивным строкам labeled (H12).
#   INPUTS: { rows: list[dict] - adresses_labeled (id/channel/city/raw_adress/etalon_id), etalon:
#             list[dict] - эталонный справочник, N: float - стоимость ложного answer (N4) }
#   OUTPUTS: { dict - n_ask_house/ask_house_resolved/n_ask_city/ask_city_resolved/decisions_first_turn }
#   SIDE_EFFECTS: none (Matcher строится только in-memory)
# END_CONTRACT: simulate
def simulate(rows: list[dict], etalon: list[dict], N: float = 10.0) -> dict[str, Any]:
    # START_BLOCK_SIMULATE
    matcher = Matcher(etalon, N=N)
    etalon_by_id = {row["etalon_id"]: row for row in etalon}

    decisions_first_turn: Counter = Counter()
    n_ask_house = ask_house_ok = 0
    n_ask_city = ask_city_ok = 0

    for row in rows:
        if row.get("etalon_id") is None:
            continue  # H12: только позитивные строки - у негативов нет истинного etalon_id
        true_etalon = etalon_by_id.get(row["etalon_id"])
        if true_etalon is None:
            continue

        raw = row.get("raw_adress", "")
        channel = row.get("channel", "voice")
        r1 = matcher.match(raw, channel=channel, slots={"city": row.get("city", "")})
        decisions_first_turn[r1.decision] += 1

        if r1.decision == "ask_house":
            n_ask_house += 1
            street_id = r1.explain.get("leader_street_id")
            city_id = r1.explain.get("leader_city_id")
            true_house = true_etalon.get("house")
            if street_id and true_house:
                slots2: dict[str, Any] = {"street": {"id": street_id}}
                if city_id:
                    slots2["city"] = {"id": city_id}
                r2 = matcher.match(str(true_house), channel=channel, slots=slots2, asked_slot="house")
                if r2.decision in ("answer", "answer_soft") and r2.candidates[:1] == [row["etalon_id"]]:
                    ask_house_ok += 1

        elif r1.decision == "ask_city":
            n_ask_city += 1
            true_city_id = normalize(true_etalon.get("city", ""), channel="webchat").text
            r2 = matcher.match(raw, channel=channel, slots={"city": {"id": true_city_id}}, asked_slot="city")
            if r2.decision in ("answer", "answer_soft", "confirm") and r2.candidates[:1] == [row["etalon_id"]]:
                ask_city_ok += 1

    def _rate(ok: int, n: int) -> float:
        return ok / n if n else 0.0

    return {
        "n_ask_house": n_ask_house,
        "ask_house_resolved": _rate(ask_house_ok, n_ask_house),
        "n_ask_city": n_ask_city,
        "ask_city_resolved": _rate(ask_city_ok, n_ask_city),
        "decisions_first_turn": dict(decisions_first_turn),
    }
    # END_BLOCK_SIMULATE


# START_CONTRACT: main
#   PURPOSE: CLI H12: прогнать simulate(), напечатать JSON, вернуть код по достижению целей.
#   INPUTS: { --adresses: путь к labeled JSONL, --etalon: путь к эталону JSONL, --N: стоимость
#             ложного answer (N4, default 10) }
#   OUTPUTS: none (печатает один JSON-объект в stdout)
#   SIDE_EFFECTS: чтение JSONL с диска; sys.exit(0) при выполнении целей H12, иначе sys.exit(1)
# END_CONTRACT: main
def main() -> None:
    # START_BLOCK_ARGS
    parser = argparse.ArgumentParser(description="addrmatch tools/simulate_dialog.py — H12")
    parser.add_argument("--adresses", required=True, help="JSONL с размеченными строками")
    parser.add_argument("--etalon", required=True, help="JSONL с эталонным справочником")
    parser.add_argument("--N", type=float, default=10.0, help="стоимость ложного answer в переспросах (N4)")
    args = parser.parse_args()
    # END_BLOCK_ARGS

    rows = load_jsonl(args.adresses)
    etalon = load_jsonl(args.etalon)
    result = simulate(rows, etalon, N=args.N)

    # START_BLOCK_GATE
    exit_code = 0
    for n_key, rate_key, label in (
        ("n_ask_house", "ask_house_resolved", "ask_house"),
        ("n_ask_city", "ask_city_resolved", "ask_city"),
    ):
        n = result[n_key]
        if n < MIN_N:
            logger.info("[SimulateDialog][main][LOW_DATA] %s: n=%d < %d - мало данных, не провал", label, n, MIN_N)
            continue
        if result[rate_key] < GOAL:
            logger.info(
                "[SimulateDialog][main][GATE_FAIL] %s: %.4f < %.2f", label, result[rate_key], GOAL
            )
            exit_code = 1
    # END_BLOCK_GATE

    print(json.dumps(result, ensure_ascii=False))
    sys.exit(exit_code)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
