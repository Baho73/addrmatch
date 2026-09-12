# START_MODULE_CONTRACT
#   PURPOSE: Симулировать дозапрос (docs/concept.md §6, H12): после ask_house/ask_city подать
#            второй match() с верным слотом-id и проверить, что диалог разрешается.
#   SCOPE: CLI-скрипт над Matcher; не трогает addrmatch/*. Только позитивные строки labeled (у
#          негативов нет истинного etalon_id, сравнивать не с чем). Естественные сценарии (T-007,
#          ask_house/ask_city как их даёт дефолтный match()) статистически бедны (n~12/n~4 на 500
#          строк) - T-007b добавляет форсированные сценарии с большим n: forced_house (дом срезан
#          из raw у КАЖДОГО позитива) и forced_city (город обнулён у КАЖДОГО позитива).
#   DEPENDS: M-MATCHER, M-PARSER (DEFAULT_STREET_TYPES, parse - только для среза дома из raw,
#            эквивалент того, что Matcher уже делает внутри - без нового связывания)
#   LINKS: V-M-MATCHER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   load_jsonl - чтение JSONL (UTF-8), как в run.py
#   strip_house - raw -> (raw без дома и хвоста после него, распознанный дом) по ParseResult
#   simulate - естественные + форсированные сценарии по позитивным строкам -> счётчики H12
#   main - CLI: --adresses/--etalon/--N, печать JSON, sys.exit(0|1) по целям H12/T-007b
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-007: simulate_dialog.py — H12: ask_house -> второй match(slots=
#   {"street": {"id": leader_street_id}, "city": {"id": leader_city_id}}, asked_slot="house",
#   raw=str(истинный дом)) считает долю answer/answer_soft с верным candidates[0]; ask_city ->
#   второй match(slots={"city": {"id": истинный city_id}}, тот же raw, asked_slot="city") считает
#   долю answer/answer_soft/confirm с верным candidates[0].
#   C-ADDRMATCH-PHASE-A T-007b: естественные ask_house/ask_city статистически пустые (n~12/n~4 из
#   500 строк - 1 ошибка = ~8 п.п.), обе естественные ошибки ask_house — аномалии данных (e_0844
#   house="10комната 513" не парсится как дом вообще; a_0156 — неверный город в labeled, коллизия
#   тёзок-улиц в разных городах). Добавлены: (1) excluded_bad_etalon — естественный ask_house
#   исключает строки, где etalon.house не парсится как дом (strip_house-парсер вернёт house=None
#   на str(true_house)) — из знаменателя, id в вывод. (2) forced_house — для КАЖДОГО позитива дом
#   срезается из raw (strip_house, ParseResult.city_hint+street_type+street = ровно tokens до
#   дома), первый match() без дома -> ожидание ask_house; строки без распознанного в исходном raw
#   дома пропускаются (n_skipped). (3) forced_city — для КАЖДОГО позитива город обнуляется
#   (slots={"city": ""}), первый match() -> ожидание ask_city. Оба сценария используют тот же
#   второй-вызов протокол, что и естественные, но на n на порядок больше (статистическая сила).
#   Гейт на exit-код: 0/1 по MIN_N=20 на каждую из 4 долей (естественные ask_house/ask_city,
#   forced_house/forced_city) - ниже MIN_N "мало данных", не провал.
# END_CHANGE_SUMMARY

"""tools/simulate_dialog.py — симуляция дозапроса H12 (docs/concept.md §6, plan.xml T-007/T-007b)."""

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
from addrmatch.index import norm_house  # noqa: E402
from addrmatch.normalizer import normalize  # noqa: E402
from addrmatch.parser import DEFAULT_STREET_TYPES, parse  # noqa: E402

logger = logging.getLogger("addrmatch.simulate_dialog")

GOAL = 0.95
MIN_N = 20  # ниже - "мало данных" (не статистика), не провал (T-007b п.4); одна доля на все 4 метрики


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


# START_CONTRACT: strip_house
#   PURPOSE: Срезать house-токен и всё после него из raw (T-007b forced_house), используя
#            ParseResult: city_hint+street_type+street — ровно tokens[:house_idx] по контракту
#            parser.py §5.2 (street_types — тождественная карта, канонический тип == токену;
#            city_hint/street — verbatim join исходных токенов, без трансформации, в отличие от
#            house-токена). has_name не передаём — не участвует в поиске дома, только в cues.
#   INPUTS: { raw: str, channel: str }
#   OUTPUTS: { tuple[str|None, str|None] - (raw без дома, распознанный house) | (None, None), если
#             в исходном raw дом не найден }
#   SIDE_EFFECTS: none
# END_CONTRACT: strip_house
def strip_house(raw: str, channel: str) -> tuple[str | None, str | None]:
    # START_BLOCK_STRIP_HOUSE
    norm = normalize(raw or "", channel=channel)
    parsed = parse(norm, DEFAULT_STREET_TYPES)
    if not parsed.house:
        return None, None
    prefix = [p for p in (parsed.city_hint, parsed.street_type, parsed.street) if p]
    return " ".join(prefix), parsed.house
    # END_BLOCK_STRIP_HOUSE


# START_CONTRACT: house_parses
#   PURPOSE: Проверить, что строка (обычно etalon.house) сама по себе разбирается как дом (T-007b
#            excluded_bad_etalon) — round-trip той же цепочкой, что видит второй вызов match():
#            normalize+parse должны вернуть ровно тот же дом, что Index кладёт в houses[] при
#            загрузке (norm_house(row["house"]) — verbatim lower/strip, без токенизации). Простое
#            "house is not None" недостаточно: "10комната 513" парсится в house="513" (не None,
#            но не совпадает с ключом "10комната 513" в словаре объекта) — round-trip ловит это.
#   INPUTS: { house_text: str }
#   OUTPUTS: { bool }
#   SIDE_EFFECTS: none
# END_CONTRACT: house_parses
def house_parses(house_text: str) -> bool:
    verbatim = norm_house(str(house_text or ""))
    if not verbatim:
        return False
    norm = normalize(str(house_text), channel="voice")
    parsed_house = parse(norm, DEFAULT_STREET_TYPES).house
    return bool(parsed_house) and norm_house(parsed_house) == verbatim


# START_CONTRACT: simulate
#   PURPOSE: Прогнать естественные и форсированные (T-007b) сценарии дозапроса по позитивным
#            строкам labeled (H12).
#   INPUTS: { rows: list[dict] - adresses_labeled (id/channel/city/raw_adress/etalon_id), etalon:
#             list[dict] - эталонный справочник, N: float - стоимость ложного answer (N4) }
#   OUTPUTS: { dict - n_ask_house/ask_house_resolved/n_ask_city/ask_city_resolved/
#             decisions_first_turn/excluded_bad_etalon/n_forced_house/n_skipped/
#             forced_house_ask_rate/forced_house_resolved/n_forced_city/forced_city_resolved }
#   SIDE_EFFECTS: none (Matcher строится только in-memory)
# END_CONTRACT: simulate
def simulate(rows: list[dict], etalon: list[dict], N: float = 10.0) -> dict[str, Any]:
    # START_BLOCK_NATURAL
    matcher = Matcher(etalon, N=N)
    etalon_by_id = {row["etalon_id"]: row for row in etalon}
    positives = [r for r in rows if r.get("etalon_id") is not None and r["etalon_id"] in etalon_by_id]

    decisions_first_turn: Counter = Counter()
    n_ask_house = ask_house_ok = 0
    excluded_bad_etalon: list[str] = []
    n_ask_city = ask_city_ok = 0

    for row in positives:
        true_etalon = etalon_by_id[row["etalon_id"]]
        raw = row.get("raw_adress", "")
        channel = row.get("channel", "voice")
        r1 = matcher.match(raw, channel=channel, slots={"city": row.get("city", "")})
        decisions_first_turn[r1.decision] += 1

        if r1.decision == "ask_house":
            true_house = true_etalon.get("house")
            if not true_house or not house_parses(true_house):
                # T-007b: etalon.house сам не разбирается как дом (напр. "10комната 513") -
                # строка выкидывается из знаменателя естественного ask_house, а не считается провалом.
                excluded_bad_etalon.append(row["id"])
            else:
                n_ask_house += 1
                street_id = r1.explain.get("leader_street_id")
                city_id = r1.explain.get("leader_city_id")
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
    # END_BLOCK_NATURAL

    # START_BLOCK_FORCED_HOUSE
    n_forced_house = n_skipped = n_forced_house_ask = forced_house_ok = 0
    for row in positives:
        raw = row.get("raw_adress", "")
        channel = row.get("channel", "voice")
        stripped, _ = strip_house(raw, channel)
        if stripped is None:
            n_skipped += 1
            continue
        n_forced_house += 1

        r1 = matcher.match(stripped, channel=channel, slots={"city": row.get("city", "")})
        if r1.decision != "ask_house":
            continue
        n_forced_house_ask += 1

        true_etalon = etalon_by_id[row["etalon_id"]]
        true_house = true_etalon.get("house")
        street_id = r1.explain.get("leader_street_id")
        city_id = r1.explain.get("leader_city_id")
        if not street_id or not true_house:
            continue
        slots2 = {"street": {"id": street_id}}
        if city_id:
            slots2["city"] = {"id": city_id}
        r2 = matcher.match(str(true_house), channel=channel, slots=slots2, asked_slot="house")
        if r2.decision in ("answer", "answer_soft") and r2.candidates[:1] == [row["etalon_id"]]:
            forced_house_ok += 1
    # END_BLOCK_FORCED_HOUSE

    # START_BLOCK_FORCED_CITY
    n_forced_city = forced_city_ok = 0
    for row in positives:
        raw = row.get("raw_adress", "")
        channel = row.get("channel", "voice")
        r1 = matcher.match(raw, channel=channel, slots={"city": ""})
        if r1.decision != "ask_city":
            continue
        n_forced_city += 1

        true_etalon = etalon_by_id[row["etalon_id"]]
        true_city_id = normalize(true_etalon.get("city", ""), channel="webchat").text
        r2 = matcher.match(raw, channel=channel, slots={"city": {"id": true_city_id}}, asked_slot="city")
        if r2.decision in ("answer", "answer_soft", "confirm") and r2.candidates[:1] == [row["etalon_id"]]:
            forced_city_ok += 1
    # END_BLOCK_FORCED_CITY

    def _rate(ok: int, n: int) -> float:
        return ok / n if n else 0.0

    return {
        "n_ask_house": n_ask_house,
        "ask_house_resolved": _rate(ask_house_ok, n_ask_house),
        "n_ask_city": n_ask_city,
        "ask_city_resolved": _rate(ask_city_ok, n_ask_city),
        "decisions_first_turn": dict(decisions_first_turn),
        "excluded_bad_etalon": excluded_bad_etalon,
        "n_forced_house": n_forced_house,
        "n_skipped": n_skipped,
        "n_forced_house_ask": n_forced_house_ask,
        "forced_house_ask_rate": _rate(n_forced_house_ask, n_forced_house),
        "forced_house_resolved": _rate(forced_house_ok, n_forced_house_ask),
        "n_forced_city": n_forced_city,
        "forced_city_resolved": _rate(forced_city_ok, n_forced_city),
    }


# START_CONTRACT: main
#   PURPOSE: CLI H12/T-007b: прогнать simulate(), напечатать JSON, вернуть код по достижению целей.
#   INPUTS: { --adresses: путь к labeled JSONL, --etalon: путь к эталону JSONL, --N: стоимость
#             ложного answer (N4, default 10) }
#   OUTPUTS: none (печатает один JSON-объект в stdout)
#   SIDE_EFFECTS: чтение JSONL с диска; sys.exit(0) при выполнении целей, иначе sys.exit(1)
# END_CONTRACT: main
def main() -> None:
    # START_BLOCK_ARGS
    parser = argparse.ArgumentParser(description="addrmatch tools/simulate_dialog.py — H12/T-007b")
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
        ("n_ask_house", "ask_house_resolved", "ask_house(natural)"),
        ("n_ask_city", "ask_city_resolved", "ask_city(natural)"),
        ("n_forced_house_ask", "forced_house_resolved", "forced_house"),
        ("n_forced_city", "forced_city_resolved", "forced_city"),
    ):
        n = result[n_key]
        if n < MIN_N:
            logger.info("[SimulateDialog][main][LOW_DATA] %s: n=%d < %d - мало данных, не провал", label, n, MIN_N)
            continue
        if result[rate_key] < GOAL:
            logger.info("[SimulateDialog][main][GATE_FAIL] %s: %.4f < %.2f (n=%d)", label, result[rate_key], GOAL, n)
            exit_code = 1
    # END_BLOCK_GATE

    print(json.dumps(result, ensure_ascii=False))
    sys.exit(exit_code)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
