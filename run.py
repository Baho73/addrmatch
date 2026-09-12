# START_MODULE_CONTRACT
#   PURPOSE: Точка входа ТЗ: прогнать Matcher по размеченным строкам, напечатать метрики (F5).
#   SCOPE: CLI run.py — чтение JSONL, замер времени match(), подсчёт 9 метрик ТЗ, опциональный --predictions.
#   DEPENDS: M-MATCHER
#   LINKS: V-M-RUN
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   load_jsonl - чтение JSONL (UTF-8) построчно в список словарей
#   compute_metrics - подсчёт 9 метрик ТЗ (test_task_adress_match/README.md) по predictions
#   main - CLI: argparse, прогон Matcher, печать JSON, запись --predictions
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-001: каркас run.py — метрики ТЗ, --predictions, замер времени вокруг match()
#   C-ADDRMATCH-PHASE-A T-004: --N (стоимость ложного answer, N4) пробрасывается в Matcher;
#   --channel voice|webchat фильтрует строки до подсчёта метрик (H2 — абляция на voice-подмножестве).
#   C-ADDRMATCH-PHASE-A T-006: --ranker logreg|manual.
#   post-A follow-up: без флага — всегда manual (детерминированно, не зависит от meets_gates в JSON);
#   логрег только явным --ranker logreg. Причина: на всех 500 labeled логрег хуже ручных весов.
# END_CHANGE_SUMMARY

"""run.py — точка входа ТЗ (см. test_task_adress_match/README.md «Критерии приёмки»)."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from addrmatch import Matcher

logger = logging.getLogger("addrmatch.run")


# START_CONTRACT: load_jsonl
#   PURPOSE: Прочитать JSONL-файл в список словарей.
#   INPUTS: { path: str - путь к файлу в кодировке UTF-8 }
#   OUTPUTS: { list[dict] - строки файла как объекты, пустые строки пропущены }
#   SIDE_EFFECTS: чтение файла с диска
# END_CONTRACT: load_jsonl
def load_jsonl(path: str) -> list[dict]:
    # START_BLOCK_READ
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
    # END_BLOCK_READ


# START_CONTRACT: compute_metrics
#   PURPOSE: Посчитать 9 метрик ТЗ по предсказаниям и разметке (README «Критерии оценивания»/«Критерии приёмки»).
#   INPUTS: { rows: list[dict] - размеченные строки (id, etalon_id), preds: list[dict] - {id, candidates, scores}, latencies_ms: list[float] - время match() на строку в мс }
#   OUTPUTS: { dict - ровно 9 ключей: n, n_positive, n_negative, top1, top3, reject_precision, reject_recall, ms_per_request_p50, ms_per_request_p95 }
#   SIDE_EFFECTS: none
# END_CONTRACT: compute_metrics
def compute_metrics(rows: list[dict], preds: list[dict], latencies_ms: list[float]) -> dict[str, Any]:
    # START_BLOCK_SPLIT
    pred_by_id = {p["id"]: p for p in preds}
    positives = [r for r in rows if r.get("etalon_id") is not None]
    negatives = [r for r in rows if r.get("etalon_id") is None]
    n_positive = len(positives)
    n_negative = len(negatives)
    # END_BLOCK_SPLIT

    # START_BLOCK_TOPK
    top1_hits = 0
    top3_hits = 0
    for r in positives:
        cand = pred_by_id[r["id"]]["candidates"]
        if cand and cand[0] == r["etalon_id"]:
            top1_hits += 1
        if r["etalon_id"] in cand[:3]:
            top3_hits += 1
    top1 = top1_hits / n_positive if n_positive else 0.0
    top3 = top3_hits / n_positive if n_positive else 0.0
    # END_BLOCK_TOPK

    # START_BLOCK_REJECT
    # Пустой candidates == отказ (README: «Пустой список кандидатов означает отказ»).
    rejected_ids = {p["id"] for p in preds if not p["candidates"]}
    n_rejects = len(rejected_ids)
    true_negatives_rejected = sum(1 for r in negatives if r["id"] in rejected_ids)
    reject_precision = true_negatives_rejected / n_rejects if n_rejects else 0.0
    reject_recall = true_negatives_rejected / n_negative if n_negative else 0.0
    # END_BLOCK_REJECT

    # START_BLOCK_LATENCY
    if latencies_ms:
        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
    else:
        p50 = 0.0
        p95 = 0.0
    # END_BLOCK_LATENCY

    return {
        "n": len(rows),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "top1": top1,
        "top3": top3,
        "reject_precision": reject_precision,
        "reject_recall": reject_recall,
        "ms_per_request_p50": p50,
        "ms_per_request_p95": p95,
    }


# START_CONTRACT: main
#   PURPOSE: CLI ТЗ: прогнать Matcher по --adresses, напечатать метрики JSON, опционально выгрузить --predictions.
#   INPUTS: { --adresses: путь к labeled JSONL, --etalon: путь к эталону JSONL, --predictions: опц. путь для выгрузки,
#             --ablate: опц. имя абляции (пробрасывается в Matcher), --N: стоимость ложного answer (N4, default 10),
#             --channel: опц. voice|webchat — считать метрики только по этому каналу (H2) }
#   OUTPUTS: none (печатает ровно один JSON-объект с 9 ключами в stdout)
#   SIDE_EFFECTS: чтение JSONL с диска; запись --predictions на диск; лог-маркер в stderr
# END_CONTRACT: main
def main() -> None:
    # START_BLOCK_ARGS
    parser = argparse.ArgumentParser(description="addrmatch run.py — метрики ТЗ")
    parser.add_argument("--adresses", required=True, help="JSONL с размеченными строками")
    parser.add_argument("--etalon", required=True, help="JSONL с эталонным справочником")
    parser.add_argument("--predictions", default=None, help="куда выгрузить predictions JSONL")
    parser.add_argument("--ablate", default=None, help="имя абляции (пробрасывается в Matcher)")
    parser.add_argument("--N", type=float, default=10, help="стоимость ложного answer в переспросах (N4)")
    parser.add_argument("--channel", default=None, choices=["voice", "webchat"], help="считать метрики только по одному каналу (H2)")
    parser.add_argument(
        "--ranker",
        default=None,
        choices=["logreg", "manual"],
        help="ранкер: logreg|manual; без флага — manual (логрег только явно, см. readme)",
    )
    args = parser.parse_args()
    # END_BLOCK_ARGS

    # START_BLOCK_LOAD
    # Загрузка данных и построение Matcher — вне замера латентности (N1: без загрузки/прогрева).
    rows = load_jsonl(args.adresses)
    if args.channel:
        rows = [r for r in rows if r.get("channel") == args.channel]
    etalon = load_jsonl(args.etalon)
    # Без явного --ranker — всегда manual: на всех 500 labeled логрег хуже ручных весов
    # (top1 0.9525 vs 0.9725), поэтому выбор по умолчанию не зависит от файла модели и её флагов.
    # Логрег — только явным --ranker logreg (см. readme §4/§7).
    ranker_choice = args.ranker or "manual"
    matcher = Matcher(etalon, ranker=ranker_choice, ablate=args.ablate, N=args.N)
    # END_BLOCK_LOAD

    # START_BLOCK_RUN
    preds: list[dict] = []
    latencies_ms: list[float] = []
    for row in rows:
        raw = row.get("raw_adress", "")
        channel = row.get("channel", "voice")
        slots = {"city": row.get("city", "")}
        t0 = time.perf_counter()
        result = matcher.match(raw, channel=channel, slots=slots)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        preds.append({"id": row["id"], "candidates": result.candidates, "scores": result.scores})
    # END_BLOCK_RUN

    # START_BLOCK_METRICS
    metrics = compute_metrics(rows, preds, latencies_ms)
    logger.info("[Run][main][METRICS] %s", metrics)
    # END_BLOCK_METRICS

    # START_BLOCK_OUTPUT
    if args.predictions:
        out_path = Path(args.predictions)
        with open(out_path, "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    # Контракт ТЗ: ровно один JSON-объект в stdout, больше ничего.
    print(json.dumps(metrics, ensure_ascii=False))
    # END_BLOCK_OUTPUT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
