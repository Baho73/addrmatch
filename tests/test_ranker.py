# START_MODULE_CONTRACT
#   PURPOSE: Тесты LogregRanker — интерфейс совместим с ManualRanker, save/load round-trip,
#            калибратор монотонен, fit() не падает на маленькой выборке, Matcher fallback на manual
#            при отсутствии addrmatch/ranker_model.json (docs/concept.md §5.4, T-006).
#   SCOPE: Юнит-тесты; fit() гоняет реальный конвейер на подмножестве adresses_labeled.jsonl/etalon.jsonl.
#   DEPENDS: M-RANKER, M-MATCHER
#   LINKS: V-M-RANKER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_logreg_interface_matches_manual_ranker - тест
#   test_save_load_roundtrip_same_p - тест
#   test_calibrator_monotonic - тест
#   test_fit_on_small_sample_does_not_crash - тест
#   test_reliability_report_has_global_max_dev - тест
#   test_matcher_falls_back_to_manual_when_model_missing - тест
#   test_fit_respects_custom_feature_names_and_hyperparams - тест
#   test_load_rejects_unknown_feature_name - тест
#   test_feature_value_street_sim_is_synthetic - тест
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-006: test_ranker.py — 6 тестов (интерфейс, save/load, калибратор
#   монотонен, fit на 50 строках, reliability_report, Matcher fallback manual без файла модели).
#   C-ADDRMATCH-PHASE-A T-006b: + 3 теста — fit(feature_names=...) обучает модель на явном
#   подмножестве фичей (+ синтетическая "street_sim"), load() отвергает неизвестное имя фичи в
#   модели, _feature_value("street_sim", ...) == street_sim(...) (T-006b: LogregRanker больше не
#   обязан использовать все FEATURE_NAMES — диагноз T-006, мультиколлинеарность lev/phon/ngram/token_set).
# END_CHANGE_SUMMARY

"""tests/test_ranker.py — docs/concept.md §5.4, plan.xml T-006/T-006b."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from addrmatch.matcher import Matcher
from addrmatch.ranker import (
    FEATURES,
    LogregRanker,
    ManualRanker,
    _apply_calibrator,
    _feature_value,
    street_sim,
)

_ETALON_PATH = Path(__file__).resolve().parent.parent / "test_task_adress_match" / "data" / "etalon.jsonl"
_LABELED_PATH = Path(__file__).resolve().parent.parent / "test_task_adress_match" / "data" / "adresses_labeled.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sample_feats() -> dict[str, float]:
    return {name: (0.7 if "sim" not in name and name not in ("lev", "phon", "ngram") else 0.6) for name, _ in FEATURES}


def _dummy_ranker() -> LogregRanker:
    # Веса по одному на фичу (детерминированные, без обучения) — для тестов интерфейса/save-load.
    coef = [0.1 * (i + 1) for i in range(len(FEATURES))]
    return LogregRanker(coef=coef, intercept=-1.0, freq_median=0.3, seed=42)


def test_logreg_interface_matches_manual_ranker():
    """score(feats)->float в [0,1] и explain(feats)->dict с ключом 'p' — тот же интерфейс, что ManualRanker."""
    manual = ManualRanker()
    logreg = _dummy_ranker()
    feats = _sample_feats()

    for ranker in (manual, logreg):
        p = ranker.score(feats)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0
        explanation = ranker.explain(feats)
        assert "p" in explanation and "marker" in explanation and "contributions" in explanation


def test_save_load_roundtrip_same_p(tmp_path):
    ranker = _dummy_ranker()
    ranker.strata_calibrators = {"voice|freq_high|resolved": {"type": "platt", "a": 1.3, "b": -0.2, "n": 20}}
    feats_a = _sample_feats()
    feats_b = {**feats_a, "channel_voice": 0.0, "city_match": 1.0, "name_freq": 0.9}

    model_path = tmp_path / "ranker_model.json"
    ranker.save(str(model_path))
    loaded = LogregRanker.load(str(model_path))

    assert loaded.score(feats_a) == ranker.score(feats_a)
    assert loaded.score(feats_b) == ranker.score(feats_b)
    assert loaded.coef == ranker.coef
    assert loaded.strata_calibrators == ranker.strata_calibrators


def test_calibrator_monotonic():
    """Платт и изотоническая калибровка не должны разворачивать порядок по z/p_raw (H3a: надёжность)."""
    platt = {"type": "platt", "a": 2.0, "b": -0.5}
    zs = [-2.0, -0.5, 0.0, 0.7, 1.5, 3.0]
    p_raws = [1.0 / (1.0 + pow(2.718281828, -z)) for z in zs]
    platt_out = [_apply_calibrator(platt, z, p) for z, p in zip(zs, p_raws)]
    assert all(a <= b + 1e-9 for a, b in zip(platt_out, platt_out[1:]))

    iso = {"type": "isotonic", "x": [0.1, 0.3, 0.5, 0.8], "y": [0.05, 0.2, 0.55, 0.9]}
    iso_out = [_apply_calibrator(iso, 0.0, p) for p in [0.0, 0.1, 0.3, 0.5, 0.65, 0.8, 1.0]]
    assert all(a <= b + 1e-9 for a, b in zip(iso_out, iso_out[1:]))


def test_fit_on_small_sample_does_not_crash():
    rows = _load_jsonl(_LABELED_PATH)[:50]
    etalon = _load_jsonl(_ETALON_PATH)

    ranker = LogregRanker.fit(rows, etalon, seed=42)

    assert len(ranker.coef) == len(ranker.feature_names)
    assert ranker.train_rows is not None and ranker.val_rows is not None
    assert len(ranker.train_rows) + len(ranker.val_rows) == 50
    feats = _sample_feats()
    p = ranker.score(feats)
    assert 0.0 <= p <= 1.0


def test_reliability_report_has_global_max_dev():
    ranker = _dummy_ranker()
    examples = [
        {"feats": {**_sample_feats(), "channel_voice": 1.0, "name_freq": 0.9, "city_match": 1.0}, "y": 1 if i % 3 == 0 else 0}
        for i in range(30)
    ]
    report = ranker.reliability_report(examples)
    assert "GLOBAL_MAX_DEV" in report
    assert isinstance(report, str) and len(report) > 0


def test_matcher_falls_back_to_manual_when_model_missing(tmp_path, monkeypatch):
    import addrmatch.matcher as matcher_module

    monkeypatch.setattr(matcher_module, "DEFAULT_MODEL_PATH", tmp_path / "no_such_model.json")

    etalon = [
        {"etalon_id": "e_0000", "city": "Москва", "street_type": "улица", "street": "Ленина", "house": "1"},
    ]
    matcher = Matcher(etalon, ranker="logreg")

    assert matcher.ready() is True
    result = matcher.match("улица ленина 1", channel="voice", slots={"city": "Москва"})
    assert result.error is None
    assert result.explain.get("ranker_fallback_warning")
    assert "manual" in result.explain["ranker_fallback_warning"]


def test_fit_respects_custom_feature_names_and_hyperparams():
    """T-006b: fit(feature_names=...) обучает модель ровно на переданном подмножестве (+ synthetic
    "street_sim"), а не на всех FEATURE_NAMES; C/class_weight/neg_per_query пробрасываются в sklearn."""
    rows = _load_jsonl(_LABELED_PATH)[:50]
    etalon = _load_jsonl(_ETALON_PATH)
    names = ["street_sim", "phon", "city_match"]

    ranker = LogregRanker.fit(
        rows, etalon, seed=42, C=0.5, class_weight="none", neg_per_query=5, feature_names=names
    )

    assert ranker.feature_names == names
    assert len(ranker.coef) == 3
    feats = _sample_feats()
    p = ranker.score(feats)
    assert 0.0 <= p <= 1.0
    # contributions ключи — ровно переданные имена (+ bias), не полный FEATURE_NAMES.
    contributions = ranker.explain(feats)["contributions"]
    assert set(contributions) == {"bias", *names}


def test_load_rejects_unknown_feature_name(tmp_path):
    """T-006b: load() проверяет каждое имя в feature_names модели против FEATURE_NAMES/street_sim —
    защита от рассинхрона модели с текущим кодом (было: точное совпадение с полным FEATURE_NAMES)."""
    bad_path = tmp_path / "bad_model.json"
    payload = {
        "version": 1,
        "feature_names": ["street_sim", "not_a_real_feature"],
        "coef": [0.1, 0.2],
        "intercept": -1.0,
        "freq_median": 0.0,
        "global_calibrator": {"type": "identity", "n": 0},
        "strata_calibrators": {},
        "seed": 42,
        "meets_gates": False,
    }
    with open(bad_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    with pytest.raises(ValueError):
        LogregRanker.load(str(bad_path))


def test_feature_value_street_sim_is_synthetic():
    """T-006b: _feature_value("street_sim", feats) считает агрегат по street_sim(), не feats["street_sim"]
    (такого ключа в feats вообще нет — matcher.py его не производит)."""
    feats = _sample_feats()
    assert "street_sim" not in feats
    assert _feature_value("street_sim", feats) == street_sim(feats)
    assert _feature_value("city_match", feats) == feats["city_match"]
    assert _feature_value("no_such_key", {}) == 0.0
