# START_MODULE_CONTRACT
#   PURPOSE: Разобрать промахи Matcher на adresses_labeled.jsonl по типам ошибок (A4) и сохранить
#            docs/errors-a4.md; печатает сводку по классам и распределение decision (7 исходов x
#            позитивы/негативы) для readme (A7) и T-006 (H5 — негативы вида 2 -> reject/ask_street).
#   SCOPE: CLI-отчёт, ничего не меняет в addrmatch/; строит собственный BruteForceIndex только
#          для диагностики (top-3 имён при поиске улицы) - Matcher не отдаёт внутреннее состояние.
#   DEPENDS: M-MATCHER, M-NORMALIZER, M-PARSER, M-INDEX
#   LINKS: V-M-MATCHER, V-M-DECIDER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   classify_positive - top1_hit | top1_miss-подкласс (street_not_found/wrong_object/
#                        house_not_parsed/house_mismatch/wrong_ranking/rejected_positive)
#   classify_negative - correctly_rejected | accepted_negative (type1/type2, эвристика check_parser.py)
#   build_report - прогон Matcher по всем labeled, сбор классов+примеров+decision-распределения
#   render_markdown - таблицы классов + до 10 примеров на класс -> docs/errors-a4.md
#   main - CLI: --adresses --etalon [--N], печать сводки, запись markdown
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-005: errors_report.py — классификация top1_miss (6 подклассов) и
#   негативов (accepted/type1/type2) на всём labeled, docs/errors-a4.md, decision x positive/negative.
#   note-колонка помечает house_mismatch, объяснимый дубликатом строки в etalon.jsonl (одинаковый
#   city/street_type/street/house под двумя etalon_id) — сводка в "## Наблюдения".
# END_CHANGE_SUMMARY

"""tools/errors_report.py — разбор ошибок A4 (docs/concept.md §5.5, plan.xml T-005)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# tools/ не пакет — добавляем корень репозитория в sys.path (как check_parser.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from addrmatch import Matcher  # noqa: E402
from addrmatch.index import BruteForceIndex  # noqa: E402
from addrmatch.matcher import DECISIONS  # noqa: E402
from addrmatch.normalizer import normalize  # noqa: E402
from addrmatch.parser import parse  # noqa: E402

POSITIVE_MISS_CLASSES = (
    "street_not_found",
    "wrong_object",
    "house_not_parsed",
    "house_mismatch",
    "wrong_ranking",
    "rejected_positive",
)
NEGATIVE_CLASSES = ("correctly_rejected", "accepted_negative_type1", "accepted_negative_type2")


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


def _norm_field(value: str) -> str:
    return normalize(value or "", channel="webchat").text


def _same_object(a: dict, b: dict) -> bool:
    """Тот же объект (город x тип x имя), сравнение той же нормализацией, что строит Index."""
    return (
        _norm_field(a.get("city", "")) == _norm_field(b.get("city", ""))
        and a.get("street_type") == b.get("street_type")
        and _norm_field(a.get("street", "")) == _norm_field(b.get("street", ""))
    )


def _street_query(norm_text: str, parse_result: Any) -> str:
    """Та же логика street_q, что Matcher._match_impl (без ablate/слотов — их у labeled нет)."""
    street_q = parse_result.street
    if not street_q and not parse_result.house:
        street_q = norm_text
    return street_q or ""


def _fmt_etalon(row: dict | None) -> str | None:
    if row is None:
        return None
    return f"{row.get('city')}, {row.get('street_type')} {row.get('street')}, {row.get('house')}"


# START_CONTRACT: classify_positive
#   PURPOSE: Классифицировать top1 позитива: top1_hit либо один из 6 подклассов top1_miss.
#   INPUTS: { etalon_id: str, candidates: list[str], parse_result: ParseResult, etalon_by_id: dict,
#             diag_index: BruteForceIndex - для проверки street_not_found }
#   OUTPUTS: { str - "top1_hit" | POSITIVE_MISS_CLASSES }
#   SIDE_EFFECTS: none
# END_CONTRACT: classify_positive
def classify_positive(
    etalon_id: str,
    candidates: list[str],
    norm_text: str,
    parse_result: Any,
    etalon_by_id: dict[str, dict],
    diag_index: BruteForceIndex,
) -> str:
    # START_BLOCK_CLASSIFY
    if candidates and candidates[0] == etalon_id:
        return "top1_hit"
    if not candidates:
        return "rejected_positive"
    if etalon_id in candidates[1:3]:
        return "wrong_ranking"

    got_row = etalon_by_id.get(candidates[0])
    true_row = etalon_by_id[etalon_id]
    if got_row is not None and _same_object(got_row, true_row):
        if parse_result.house is None or parse_result.ambiguous_number:
            return "house_not_parsed"
        return "house_mismatch"

    street_q = _street_query(norm_text, parse_result)
    name_cands = diag_index.candidates(street_q, k=3)
    top3_names = {nc.name for nc in name_cands}
    if _norm_field(true_row.get("street", "")) not in top3_names:
        return "street_not_found"
    return "wrong_object"
    # END_BLOCK_CLASSIFY


# START_CONTRACT: classify_negative
#   PURPOSE: Классифицировать негатив: correctly_rejected либо accepted_negative (тип-1/тип-2).
#   INPUTS: { candidates: list[str], norm_text: str, street_type: str|None - parse_result.street_type }
#   OUTPUTS: { str - одно из NEGATIVE_CLASSES }
#   SIDE_EFFECTS: none
# END_CONTRACT: classify_negative
def classify_negative(candidates: list[str], norm_text: str, street_type: str | None) -> str:
    # START_BLOCK_CLASSIFY
    if not candidates:
        return "correctly_rejected"
    # Эвристика tools/check_parser.py: тип-1 "не про адрес" — ни цифры, ни тип улицы в строке.
    has_digit = any(ch.isdigit() for ch in norm_text)
    if not has_digit and street_type is None:
        return "accepted_negative_type1"
    return "accepted_negative_type2"
    # END_BLOCK_CLASSIFY


# START_CONTRACT: build_report
#   PURPOSE: Прогнать Matcher по всем labeled, собрать классы/примеры/decision-распределение.
#   INPUTS: { rows: list[dict], etalon: list[dict], N: float }
#   OUTPUTS: { dict - class_counts, decision_counts, examples (class -> list[dict]) }
#   SIDE_EFFECTS: построение Matcher и отдельного диагностического BruteForceIndex (только чтение)
# END_CONTRACT: build_report
def build_report(rows: list[dict], etalon: list[dict], N: float = 10) -> dict[str, Any]:
    # START_BLOCK_RUN
    etalon_by_id = {r["etalon_id"]: r for r in etalon}
    matcher = Matcher(etalon, N=N)
    diag_index = BruteForceIndex(etalon)

    class_counts: Counter[str] = Counter()
    decision_counts: dict[str, Counter[str]] = {"positive": Counter(), "negative": Counter()}
    examples: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        raw = row.get("raw_adress", "")
        channel = row.get("channel", "voice")
        city = row.get("city", "")
        etalon_id = row.get("etalon_id")

        norm = normalize(raw, channel=channel)
        parse_result = parse(norm, diag_index.street_types, has_name=diag_index.has_name)
        result = matcher.match(raw, channel=channel, slots={"city": city})

        kind = "positive" if etalon_id is not None else "negative"
        decision_counts[kind][result.decision] += 1

        feats = result.explain.get("feats", {}) if result.explain else {}
        example = {
            "id": row.get("id"),
            "raw": raw,
            "city": city,
            "etalon": _fmt_etalon(etalon_by_id.get(etalon_id)),
            "got": _fmt_etalon(etalon_by_id.get(result.candidates[0])) if result.candidates else None,
            "decision": result.decision,
            "p": result.explain.get("p") if result.explain else None,
            "street_sim": result.explain.get("street_sim") if result.explain else None,
            "city_match": feats.get("city_match"),
            "house_in_list": feats.get("house_in_list"),
            "n_close": feats.get("n_close"),
        }
        example["note"] = "дубликат эталона" if example["got"] and example["got"] == example["etalon"] else ""

        if etalon_id is not None:
            cls = classify_positive(etalon_id, result.candidates, norm.text, parse_result, etalon_by_id, diag_index)
        else:
            cls = classify_negative(result.candidates, norm.text, parse_result.street_type)

        class_counts[cls] += 1
        if cls != "top1_hit" and cls != "correctly_rejected" and len(examples[cls]) < 10:
            examples[cls].append(example)

    return {"class_counts": class_counts, "decision_counts": decision_counts, "examples": examples}
    # END_BLOCK_RUN


def _print_summary(report: dict[str, Any], n_positive: int, n_negative: int) -> None:
    # START_BLOCK_PRINT
    cc = report["class_counts"]
    print(f"n_positive={n_positive} n_negative={n_negative}")
    print(f"top1_hit: {cc['top1_hit']}/{n_positive} = {cc['top1_hit'] / n_positive:.4f}")
    print("top1_miss по подклассам:")
    for cls in POSITIVE_MISS_CLASSES:
        n = cc[cls]
        print(f"  {cls}: {n} ({n / n_positive:.4f})")
    print(f"correctly_rejected: {cc['correctly_rejected']}/{n_negative} = {cc['correctly_rejected'] / n_negative:.4f}")
    for cls in ("accepted_negative_type1", "accepted_negative_type2"):
        n = cc[cls]
        print(f"  {cls}: {n} ({n / n_negative:.4f})")

    print("\ndecision x positive/negative:")
    for d in DECISIONS:
        pos = report["decision_counts"]["positive"][d]
        neg = report["decision_counts"]["negative"][d]
        print(f"  {d}: positive={pos} negative={neg}")
    # END_BLOCK_PRINT


# START_CONTRACT: render_markdown
#   PURPOSE: Собрать docs/errors-a4.md — таблица по классам + до 10 примеров на класс.
#   INPUTS: { report: dict - результат build_report, n_positive: int, n_negative: int }
#   OUTPUTS: { str - готовый markdown }
#   SIDE_EFFECTS: none
# END_CONTRACT: render_markdown
def render_markdown(report: dict[str, Any], n_positive: int, n_negative: int) -> str:
    # START_BLOCK_RENDER
    cc = report["class_counts"]
    lines: list[str] = ["# A4 — разбор ошибок на labeled (tools/errors_report.py)", ""]

    lines.append("## Позитивы (top1)")
    lines.append("")
    lines.append(f"n_positive = {n_positive}")
    lines.append("")
    lines.append("| класс | число | доля |")
    lines.append("|---|---|---|")
    lines.append(f"| top1_hit | {cc['top1_hit']} | {cc['top1_hit'] / n_positive:.4f} |")
    for cls in POSITIVE_MISS_CLASSES:
        lines.append(f"| {cls} | {cc[cls]} | {cc[cls] / n_positive:.4f} |")
    lines.append("")

    lines.append("## Негативы")
    lines.append("")
    lines.append(f"n_negative = {n_negative}")
    lines.append("")
    lines.append("| класс | число | доля |")
    lines.append("|---|---|---|")
    lines.append(f"| correctly_rejected | {cc['correctly_rejected']} | {cc['correctly_rejected'] / n_negative:.4f} |")
    for cls in ("accepted_negative_type1", "accepted_negative_type2"):
        lines.append(f"| {cls} | {cc[cls]} | {cc[cls] / n_negative:.4f} |")
    lines.append("")

    lines.append("## Распределение decision (7 исходов x позитивы/негативы)")
    lines.append("")
    lines.append("| decision | positive | negative |")
    lines.append("|---|---|---|")
    for d in DECISIONS:
        pos = report["decision_counts"]["positive"][d]
        neg = report["decision_counts"]["negative"][d]
        lines.append(f"| {d} | {pos} | {neg} |")
    lines.append("")

    dup_in_house_mismatch = sum(1 for ex in report["examples"].get("house_mismatch", []) if ex["note"])
    if dup_in_house_mismatch:
        lines.append("## Наблюдения")
        lines.append("")
        lines.append(
            f"- {dup_in_house_mismatch}/{cc['house_mismatch']} house_mismatch — дубликаты в etalon.jsonl "
            "(одинаковый city/street_type/street/house под двумя etalon_id); любой из двух — равно верный "
            "ответ, промах top1 неустраним без изменения эталона."
        )
        lines.append("")

    lines.append("## Примеры по классам (до 10 на класс)")
    for cls in (*POSITIVE_MISS_CLASSES, "accepted_negative_type1", "accepted_negative_type2"):
        ex_list = report["examples"].get(cls, [])
        if not ex_list:
            continue
        lines.append("")
        lines.append(f"### {cls} ({cc[cls]})")
        lines.append("")
        lines.append("| id | raw | city | эталон | получили | decision | p | street_sim | city_match | house_in_list | n_close | note |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for ex in ex_list:
            p = "" if ex["p"] is None else f"{ex['p']:.3f}"
            ssim = "" if ex["street_sim"] is None else f"{ex['street_sim']:.3f}"
            lines.append(
                f"| {ex['id']} | {ex['raw']} | {ex['city']} | {ex['etalon'] or ''} | {ex['got'] or ''} | "
                f"{ex['decision']} | {p} | {ssim} | {ex['city_match']} | {ex['house_in_list']} | {ex['n_close']} | {ex['note']} |"
            )
    lines.append("")
    return "\n".join(lines)
    # END_BLOCK_RENDER


# START_CONTRACT: main
#   PURPOSE: CLI: прогнать build_report по --adresses/--etalon, напечатать сводку, записать markdown.
#   INPUTS: { --adresses: путь к labeled JSONL, --etalon: путь к эталону JSONL, --N: стоимость
#             ложного answer (N4, default 10) }
#   OUTPUTS: none (печатает сводку в stdout; пишет docs/errors-a4.md)
#   SIDE_EFFECTS: чтение JSONL с диска; запись docs/errors-a4.md
# END_CONTRACT: main
def main() -> None:
    # START_BLOCK_ARGS
    argp = argparse.ArgumentParser(description="Разбор ошибок Matcher на labeled (A4, T-005)")
    argp.add_argument("--adresses", required=True)
    argp.add_argument("--etalon", required=True)
    argp.add_argument("--N", type=float, default=10, help="стоимость ложного answer в переспросах (N4)")
    argp.add_argument("--out", default="docs/errors-a4.md", help="куда писать markdown-отчёт")
    args = argp.parse_args()
    # END_BLOCK_ARGS

    rows = load_jsonl(args.adresses)
    etalon = load_jsonl(args.etalon)
    n_positive = sum(1 for r in rows if r.get("etalon_id") is not None)
    n_negative = len(rows) - n_positive

    report = build_report(rows, etalon, N=args.N)
    _print_summary(report, n_positive, n_negative)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(report, n_positive, n_negative), encoding="utf-8")
    print(f"\n[ErrorsReport][main][WRITE] {out_path}")


if __name__ == "__main__":
    main()
