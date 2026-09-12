# START_MODULE_CONTRACT
#   PURPOSE: Обучить LogregRanker (T-006, docs/concept.md §5.4) на adresses_labeled.jsonl, откалибровать
#            по стратам, напечатать reliability_report и метрики val (H3/H3a/H5), сохранить JSON-модель.
#   SCOPE: CLI-обёртка над LogregRanker.fit()/save(); собственный BruteForceIndex — только для
#          диагностики негативов тип-2 (has_address_cues), как в tools/errors_report.py.
#   DEPENDS: M-RANKER, M-MATCHER, M-INDEX, M-NORMALIZER, M-PARSER
#   LINKS: V-M-RANKER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   load_jsonl - чтение JSONL (UTF-8)
#   is_negative_type2 - эвристика tools/errors_report.py: не тип-1 ("не про адрес") => тип-2
#   row_top1 - top1 по подмножеству labeled-строк (train/val) для гейта H3a (расхождение <= 5 п.п.)
#   val_gates - H3 (ошибки answer/answer_soft, покрытие) и H5 (негативы тип-2 -> reject/ask_street) на val
#   main - CLI: --adresses --etalon --seed [--out] [--N]; fit -> reliability_report -> гейты H3/H3a/H5
#          -> ranker.meets_gates -> save (JSON несёт meets_gates - run.py включает logreg по умолчанию
#          только если гейты пройдены)
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-006: train_ranker.py — обучение LogregRanker, reliability_report, гейты
#   H3/H3a/H5 на val-сплите (75/25, стратифицированный по channel x is_negative внутри LogregRanker.fit).
#   Прогон seed=42: H3 (0 ошибок answer/answer_soft, покрытие 0.97) и H3a (max_dev 0.003, gap 4.3 п.п.)
#   сходятся; H5 не сходится (негативы тип-2 -> reject/ask_street 0.40 вместо >= 0.85) — logreg хуже
#   ManualRanker по top1/top3/reject_precision/reject_recall на полном run.py (docs/readme "что дальше").
#   meets_gates=False -> run.py по умолчанию остаётся на manual (T-006 п.7 плана).
#   C-ADDRMATCH-PHASE-A T-006b: + флаги --C, --class-weight none|balanced, --neg-per-query K
#   (пробрасываются в LogregRanker.fit, T-006 диагноз — class_weight=balanced при дисбалансе 1:14.6
#   и коррелированных lev/phon/ngram/token_set); summary теперь несёт C/class_weight/neg_per_query/
#   n_features прогона для сравнения вариантов.
# END_CHANGE_SUMMARY

"""tools/train_ranker.py — обучение LogregRanker (docs/concept.md §5.4, plan.xml T-006)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# tools/ не пакет — добавляем корень репозитория в sys.path (как errors_report.py/check_parser.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from addrmatch import Matcher  # noqa: E402
from addrmatch.index import BruteForceIndex  # noqa: E402
from addrmatch.normalizer import normalize  # noqa: E402
from addrmatch.parser import parse  # noqa: E402
from addrmatch.ranker import LogregRanker  # noqa: E402


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


# START_CONTRACT: is_negative_type2
#   PURPOSE: Отличить негатив вида 2 ("улица не из эталона") от вида 1 ("не про адрес") — та же
#            эвристика, что tools/errors_report.py: тип-1, если ни цифры, ни тип улицы в строке.
#   INPUTS: { raw: str, channel: str, diag_index: BruteForceIndex }
#   OUTPUTS: { bool - True для типа 2 }
#   SIDE_EFFECTS: none
# END_CONTRACT: is_negative_type2
def is_negative_type2(raw: str, channel: str, diag_index: BruteForceIndex) -> bool:
    norm = normalize(raw or "", channel=channel)
    parse_result = parse(norm, diag_index.street_types, has_name=diag_index.has_name)
    has_digit = any(ch.isdigit() for ch in norm.text)
    if not has_digit and parse_result.street_type is None:
        return False
    return True


def row_top1(matcher: Matcher, rows: list[dict]) -> float:
    """top1 по подмножеству labeled-строк (train/val) — для гейта H3a (расхождение train/val)."""
    positives = [r for r in rows if r.get("etalon_id") is not None]
    if not positives:
        return 0.0
    hits = 0
    for row in positives:
        result = matcher.match(row.get("raw_adress", ""), channel=row.get("channel", "voice"), slots={"city": row.get("city", "")})
        if result.candidates and result.candidates[0] == row["etalon_id"]:
            hits += 1
    return hits / len(positives)


# START_CONTRACT: val_gates
#   PURPOSE: Посчитать H3 (ошибки answer/answer_soft, покрытие) и H5 (негативы тип-2 -> reject/ask_street) на val.
#   INPUTS: { matcher: Matcher (с обученным LogregRanker), val_rows: list[dict], diag_index: BruteForceIndex }
#   OUTPUTS: { dict - answer_errors, coverage_answer_soft, n_positive_val, negative_type2_reject_rate, n_negative_type2_val }
#   SIDE_EFFECTS: none
# END_CONTRACT: val_gates
def val_gates(matcher: Matcher, val_rows: list[dict], diag_index: BruteForceIndex) -> dict:
    # START_BLOCK_GATES
    positives = [r for r in val_rows if r.get("etalon_id") is not None]
    negatives = [r for r in val_rows if r.get("etalon_id") is None]

    answer_errors = 0
    covered = 0
    for row in positives:
        result = matcher.match(row["raw_adress"], channel=row.get("channel", "voice"), slots={"city": row.get("city", "")})
        if result.decision in ("answer", "answer_soft"):
            covered += 1
            if not result.candidates or result.candidates[0] != row["etalon_id"]:
                answer_errors += 1
    coverage = covered / len(positives) if positives else 0.0

    type2_total = 0
    type2_rejected = 0
    for row in negatives:
        if not is_negative_type2(row.get("raw_adress", ""), row.get("channel", "voice"), diag_index):
            continue
        type2_total += 1
        result = matcher.match(row["raw_adress"], channel=row.get("channel", "voice"), slots={"city": row.get("city", "")})
        if result.decision in ("reject", "ask_street"):
            type2_rejected += 1
    type2_rate = type2_rejected / type2_total if type2_total else 1.0

    return {
        "n_positive_val": len(positives),
        "n_negative_val": len(negatives),
        "answer_errors": answer_errors,
        "coverage_answer_soft": coverage,
        "n_negative_type2_val": type2_total,
        "negative_type2_reject_rate": type2_rate,
    }
    # END_BLOCK_GATES


# START_CONTRACT: main
#   PURPOSE: CLI: обучить LogregRanker, сохранить модель, напечатать reliability_report и метрики (H3/H3a/H5).
#   INPUTS: { --adresses, --etalon, --seed (default 42), --out (default addrmatch/ranker_model.json), --N (default 10) }
#   OUTPUTS: none (печатает reliability_report + JSON метрик в stdout; пишет --out)
#   SIDE_EFFECTS: чтение JSONL с диска; запись модели на диск
# END_CONTRACT: main
def main() -> None:
    # START_BLOCK_ARGS
    argp = argparse.ArgumentParser(description="Обучение LogregRanker (T-006, docs/concept.md §5.4)")
    argp.add_argument("--adresses", required=True)
    argp.add_argument("--etalon", required=True)
    argp.add_argument("--seed", type=int, default=42)
    argp.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "addrmatch" / "ranker_model.json"))
    argp.add_argument("--N", type=float, default=10, help="стоимость ложного answer в переспросах (N4)")
    argp.add_argument("--C", type=float, default=0.3, help="C sklearn LogisticRegression (T-006b, default — лучший из 3 прогонов T-006b)")
    argp.add_argument(
        "--class-weight", default="balanced", choices=["none", "balanced"],
        help="class_weight LogisticRegression: none|balanced (T-006b, диагноз T-006 — balanced при 1:14.6 сдвигал границу)",
    )
    argp.add_argument(
        "--neg-per-query", type=int, default=10,
        help="отсечка кандидатов на строку в обучающей выборке — top-K по street_sim (T-006b)",
    )
    args = argp.parse_args()
    # END_BLOCK_ARGS

    rows = load_jsonl(args.adresses)
    etalon = load_jsonl(args.etalon)

    ranker = LogregRanker.fit(
        rows, etalon, seed=args.seed, C=args.C, class_weight=args.class_weight, neg_per_query=args.neg_per_query
    )

    report_text = ranker.reliability_report()
    print()
    print(report_text)
    print()
    global_max_dev = float(report_text.strip().splitlines()[-1].split("\t")[-1])

    eval_matcher = Matcher(etalon, ranker=ranker, N=args.N)
    diag_index = BruteForceIndex(etalon)

    top1_train = row_top1(eval_matcher, ranker.train_rows or [])
    top1_val = row_top1(eval_matcher, ranker.val_rows or [])
    top1_gap_pp = round(abs(top1_train - top1_val) * 100.0, 2)
    gates = val_gates(eval_matcher, ranker.val_rows or [], diag_index)

    # START_BLOCK_GATES_VERDICT
    # H3: ошибок answer/answer_soft <= 3 при покрытии >= 0.75. H3a: ни один бин (n>=10) не
    # отклоняется > 0.1, train/val top1 расходятся <= 5 п.п. H5: негативы тип-2 -> reject/ask_street >= 0.85.
    h3_ok = gates["answer_errors"] <= 3 and gates["coverage_answer_soft"] >= 0.75
    h3a_ok = global_max_dev <= 0.1 and top1_gap_pp <= 5.0
    h5_ok = gates["negative_type2_reject_rate"] >= 0.85
    ranker.meets_gates = bool(h3_ok and h3a_ok and h5_ok)
    # END_BLOCK_GATES_VERDICT

    ranker.save(args.out)
    print(f"[TrainRanker][main][SAVE] {args.out} (meets_gates={ranker.meets_gates})")

    summary = {
        "C": args.C,
        "class_weight": args.class_weight,
        "neg_per_query": args.neg_per_query,
        "n_features": len(ranker.feature_names),
        "n_train_rows": len(ranker.train_rows or []),
        "n_val_rows": len(ranker.val_rows or []),
        "top1_train": top1_train,
        "top1_val": top1_val,
        "top1_train_val_gap_pp": top1_gap_pp,
        "reliability_global_max_dev": global_max_dev,
        **gates,
        "H3_ok": h3_ok,
        "H3a_ok": h3a_ok,
        "H5_ok": h5_ok,
        "meets_gates": ranker.meets_gates,
    }
    print(f"[TrainRanker][main][METRICS] {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
