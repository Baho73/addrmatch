# START_MODULE_CONTRACT
#   PURPOSE: Проверить файл predictions на соответствие формату ТЗ (id/candidates/scores).
#   SCOPE: CLI-валидатор формата ТЗ (число строк, ключи, длины, ограничение candidates ≤ 3).
#   DEPENDS: none
#   LINKS: V-M-RUN
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   main - CLI: путь к файлу, --expect N, проверки формата, "OK N" либо ненулевой код возврата
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-001: проверка числа строк и формата {id, candidates, scores}
# END_CHANGE_SUMMARY

"""tools/check_preds.py — валидация файла predictions (см. test_task_adress_match/README.md)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# START_CONTRACT: main
#   PURPOSE: Проверить predictions-файл: существование, точное число строк, формат каждой строки.
#   INPUTS: { path: str - путь к файлу predictions (позиционный аргумент), --expect: int - ожидаемое число строк }
#   OUTPUTS: none (печатает "OK N" в stdout при успехе)
#   SIDE_EFFECTS: sys.exit(1) с сообщением в stderr при любом несоответствии
# END_CONTRACT: main
def main() -> None:
    # START_BLOCK_ARGS
    parser = argparse.ArgumentParser(description="Проверка формата predictions (id/candidates/scores)")
    parser.add_argument("path", help="путь к JSONL-файлу predictions")
    parser.add_argument("--expect", type=int, required=True, help="ожидаемое число строк")
    args = parser.parse_args()
    # END_BLOCK_ARGS

    # START_BLOCK_EXISTS
    path = Path(args.path)
    if not path.is_file():
        print(f"FAIL: файл не найден: {path}", file=sys.stderr)
        sys.exit(1)
    # END_BLOCK_EXISTS

    # START_BLOCK_VALIDATE
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != args.expect:
        print(f"FAIL: строк {len(lines)}, ожидалось {args.expect}", file=sys.stderr)
        sys.exit(1)

    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"FAIL: строка {i}: не JSON ({exc})", file=sys.stderr)
            sys.exit(1)

        missing = [key for key in ("id", "candidates", "scores") if key not in obj]
        if missing:
            print(f"FAIL: строка {i}: нет ключей {missing}", file=sys.stderr)
            sys.exit(1)

        candidates = obj["candidates"]
        scores = obj["scores"]
        if len(candidates) != len(scores):
            print(f"FAIL: строка {i}: len(candidates)={len(candidates)} != len(scores)={len(scores)}", file=sys.stderr)
            sys.exit(1)
        if len(candidates) > 3:
            print(f"FAIL: строка {i}: candidates содержит {len(candidates)} > 3 элементов", file=sys.stderr)
            sys.exit(1)
    # END_BLOCK_VALIDATE

    print(f"OK {len(lines)}")


if __name__ == "__main__":
    main()
