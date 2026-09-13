# START_MODULE_CONTRACT
#   PURPOSE: Прогнать normalize+parse на всём adresses_labeled.jsonl и напечатать метрики
#            приёмки (house_found_rate, street_type_found_rate, cues_rate по негативам).
#   SCOPE: CLI-отчёт для A2 (docs/concept.md §5.2); ничего не пишет, кроме stdout.
#   DEPENDS: M-NORMALIZER, M-PARSER
#   LINKS: V-M-PARSER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   build_name_words - множество слов >=4 букв из etalon.street (нормализованных) для has_name
#   classify_negative - негатив тип-1 "не про адрес" / тип-2 "адрес не из эталона"
#   main - CLI: --adresses --etalon, печать метрик и худших примеров
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-003: check_parser.py — house_found_rate, street_type_found_rate,
#   разбор негативов на тип-1/тип-2 с cues_rate, топ-10 худших примеров каждого рода.
# END_CHANGE_SUMMARY

"""tools/check_parser.py — метрики приёмки Parser (см. docs/concept.md §5.2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# tools/ не пакет — добавляем корень репозитория в sys.path, чтобы addrmatch был виден
# при прямом запуске `python tools/check_parser.py` (без -m/PYTHONPATH).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from addrmatch.normalizer import normalize  # noqa: E402
from addrmatch.parser import DEFAULT_STREET_TYPES, parse  # noqa: E402


# START_CONTRACT: load_jsonl
#   PURPOSE: Прочитать JSONL-файл (UTF-8) в список словарей.
#   INPUTS: { path: str }
#   OUTPUTS: { list[dict] }
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


# START_CONTRACT: build_name_words
#   PURPOSE: Множество нормализованных слов длиной >=4 из etalon.street — заглушка has_name.
#   INPUTS: { etalon: list[dict] }
#   OUTPUTS: { set[str] }
#   SIDE_EFFECTS: none
# END_CONTRACT: build_name_words
def build_name_words(etalon: list[dict]) -> set[str]:
    words: set[str] = set()
    for row in etalon:
        norm = normalize(row.get("street", ""), channel="voice")
        for tok in norm.text.split():
            if len(tok) >= 4:
                words.add(tok)
    return words


# START_CONTRACT: classify_negative
#   PURPOSE: Поделить негатив на тип-1 "не про адрес" / тип-2 "адрес не из эталона" (H4/H5).
#   INPUTS: { norm_text: str - нормализованный текст, street_type: str|None - parse().street_type }
#   OUTPUTS: { str - "type1"|"type2" }
#   SIDE_EFFECTS: none
# END_CONTRACT: classify_negative
def classify_negative(norm_text: str, street_type: str | None) -> str:
    has_digit = any(ch.isdigit() for ch in norm_text)
    if not has_digit and street_type is None:
        return "type1"
    return "type2"


# START_CONTRACT: main
#   PURPOSE: Прогнать normalize+parse по --adresses, напечатать метрики приёмки.
#   INPUTS: { --adresses: путь к labeled JSONL, --etalon: путь к эталону JSONL }
#   OUTPUTS: none (печатает отчёт в stdout)
#   SIDE_EFFECTS: чтение JSONL с диска
# END_CONTRACT: main
def main() -> None:
    # START_BLOCK_ARGS
    argp = argparse.ArgumentParser(description="Метрики Parser")
    argp.add_argument("--adresses", required=True)
    argp.add_argument("--etalon", required=True)
    args = argp.parse_args()
    # END_BLOCK_ARGS

    # START_BLOCK_LOAD
    rows = load_jsonl(args.adresses)
    etalon = load_jsonl(args.etalon)
    etalon_by_id = {r["etalon_id"]: r for r in etalon}
    name_words = build_name_words(etalon)
    has_name = lambda tok: tok in name_words  # noqa: E731
    # END_BLOCK_LOAD

    # START_BLOCK_RUN
    parsed_rows = []
    for row in rows:
        norm = normalize(row.get("raw_adress", ""), channel=row.get("channel", "voice"))
        result = parse(norm, DEFAULT_STREET_TYPES, has_name=has_name)
        parsed_rows.append((row, norm, result))
    # END_BLOCK_RUN

    # START_BLOCK_HOUSE
    house_total = 0
    house_hits = 0
    house_miss_examples: list[str] = []
    for row, norm, result in parsed_rows:
        etalon_id = row.get("etalon_id")
        if etalon_id is None:
            continue
        exp_house = etalon_by_id.get(etalon_id, {}).get("house", "")
        if not exp_house:
            continue
        house_total += 1
        if result.house is not None and result.house.lower() == exp_house.lower():
            house_hits += 1
        elif len(house_miss_examples) < 10:
            house_miss_examples.append(
                f"{row['id']}: raw={row['raw_adress']!r} norm={norm.text!r} "
                f"parsed_house={result.house!r} expected={exp_house!r}"
            )
    house_found_rate = house_hits / house_total if house_total else 0.0
    # END_BLOCK_HOUSE

    # START_BLOCK_STREET_TYPE
    type_total = 0
    type_hits = 0
    type_miss_examples: list[str] = []
    for row, norm, result in parsed_rows:
        if row.get("etalon_id") is None:
            continue
        has_type_word = any(tok in DEFAULT_STREET_TYPES for tok in norm.text.split())
        if not has_type_word:
            continue
        type_total += 1
        if result.street_type is not None:
            type_hits += 1
        elif len(type_miss_examples) < 10:
            type_miss_examples.append(f"{row['id']}: raw={row['raw_adress']!r} norm={norm.text!r}")
    street_type_found_rate = type_hits / type_total if type_total else 0.0
    # END_BLOCK_STREET_TYPE

    # START_BLOCK_NEGATIVES
    type1_total = 0
    type1_cued = 0
    type1_examples: list[str] = []
    type2_total = 0
    type2_cued = 0
    for row, norm, result in parsed_rows:
        if row.get("etalon_id") is not None:
            continue
        kind = classify_negative(norm.text, result.street_type)
        if kind == "type1":
            type1_total += 1
            if result.has_address_cues:
                type1_cued += 1
                if len(type1_examples) < 10:
                    type1_examples.append(f"{row['id']}: raw={row['raw_adress']!r} norm={norm.text!r}")
        else:
            type2_total += 1
            if result.has_address_cues:
                type2_cued += 1
    type1_cues_rate = type1_cued / type1_total if type1_total else 0.0
    type2_cues_rate = type2_cued / type2_total if type2_total else 0.0
    # END_BLOCK_NEGATIVES

    # START_BLOCK_POSITIVE_CUES
    positive_total = 0
    positive_no_cues = 0
    positive_no_cues_examples: list[str] = []
    for row, norm, result in parsed_rows:
        if row.get("etalon_id") is None:
            continue
        positive_total += 1
        if not result.has_address_cues:
            positive_no_cues += 1
            if len(positive_no_cues_examples) < 10:
                positive_no_cues_examples.append(f"{row['id']}: raw={row['raw_adress']!r} norm={norm.text!r}")
    positive_no_cues_rate = positive_no_cues / positive_total if positive_total else 0.0
    # END_BLOCK_POSITIVE_CUES

    # START_BLOCK_REPORT
    print(f"house_found_rate: {house_hits}/{house_total} = {house_found_rate:.4f} (цель >= 0.95)")
    print(f"street_type_found_rate: {type_hits}/{type_total} = {street_type_found_rate:.4f}")
    print(
        f"negatives type1 (не про адрес): {type1_total}, cues_rate={type1_cues_rate:.4f} "
        f"(цель <= 0.10, т.е. >= 90% пойманы)"
    )
    print(f"negatives type2 (адрес не из эталона): {type2_total}, cues_rate={type2_cues_rate:.4f}")
    print(f"positives cues=False rate: {positive_no_cues}/{positive_total} = {positive_no_cues_rate:.4f} (цель <= 0.01)")

    print("\n-- house: найден, но не совпал (до 10) --")
    for ex in house_miss_examples:
        print(ex)
    print("\n-- street_type: не найден при наличии типа в тексте (до 10) --")
    for ex in type_miss_examples:
        print(ex)
    print("\n-- негативы тип-1 с ложным has_address_cues=True (до 10) --")
    for ex in type1_examples:
        print(ex)
    print("\n-- позитивы с has_address_cues=False (до 10) --")
    for ex in positive_no_cues_examples:
        print(ex)
    # END_BLOCK_REPORT


if __name__ == "__main__":
    main()
