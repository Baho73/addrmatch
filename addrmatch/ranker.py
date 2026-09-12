# START_MODULE_CONTRACT
#   PURPOSE: Оценить объект-кандидат (имя × город × тип) вероятностью "это верный объект" (§5.4).
#   SCOPE: Таблица фичей FEATURES (контракт, общий для ManualRanker и LogregRanker) + ManualRanker
#          - ручные веса через сигмоиду, explain() раскладывает скор по сигналам. LogregRanker (T-006)
#          - sklearn LogisticRegression + Platt/изотоническая калибровка по стратам, JSON save/load.
#   DEPENDS: none для score()/explain() (принимают готовый словарь фичей). LogregRanker.fit() —
#            отложенный импорт M-MATCHER (нужен Matcher.candidate_features для обучающей выборки;
#            прямой импорт на уровне модуля дал бы цикл matcher.py<->ranker.py, см. комментарий у fit()).
#   LINKS: V-M-RANKER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   FEATURES - список (имя, значение-по-умолчанию) - контракт фичей Ranker (§5.4)
#   street_sim - 0.35*lev + 0.35*phon + 0.2*ngram + 0.1*token_set (used by ManualRanker/LogregRanker/matcher/train_ranker)
#   _feature_value - значение фичи по имени для LogregRanker: "street_sim" - синтетическая (вызов
#                    street_sim()), остальные - прямой feats.get (T-006b: LogregRanker может учиться
#                    на подмножестве FEATURES, не только на полном наборе)
#   ManualRanker - score(feats)->p, explain(feats)->разложение, ручные веса + сигмоида
#   LogregRanker - score/explain (тот же интерфейс) + fit/save/load; калибровка по стратам + reliability_report;
#                  fit(C, class_weight, neg_per_query, feature_names) - гиперпараметры T-006b
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-004: ranker.py — ManualRanker (street_sim + house/city/type + interaction),
#   стартовые веса докрутки docs/concept.md §5.4.
#   C-ADDRMATCH-PHASE-A T-006: + фича sim_x_house (interaction street_sim*house_in_list); LogregRanker
#   (sklearn LogisticRegression, class_weight=balanced) с калибровкой по стратам
#   {channel}x{freq_class}x{city_status} (Platt <40 val-точек в страте, изотоническая >=40, глобальный
#   калибратор — fallback); JSON save/load (без pickle); reliability_report() — бины надёжности (H3a).
#   freq_class: name_freq > median (строго), не >= — большинство имён однообъектны и их name_freq
#   равен медиане (floor распределения), ">=" клал весь floor в freq_high и вырождал freq_low.
#   meets_gates: bool, выставляет tools/train_ranker.py по факту H3/H3a/H5, сохраняется в JSON —
#   run.py включает logreg по умолчанию только если гейты пройдены (иначе manual, см. п.7 плана:
#   в этом прогоне (seed=42) H5 не сошёлся — logreg хуже manual по всем метрикам run.py, см. readme).
#   C-ADDRMATCH-PHASE-A T-006b: диагноз T-006 — class_weight=balanced при перекосе 1:14.6
#   (296 позитивов/4327 негативов) двигал границу к похожим-но-неверным кандидатам; коррелированные
#   lev/phon/ngram/token_set давали нестабильные веса (ngram +6.47). Правки: (1) _build_training_examples
#   — единая отсечка top-K по street_sim на строку (neg_per_query, было top-10 ∪ street_sim>=0.5 для
#   позитивных строк и ВСЕ кандидаты для негативных — вносило лишние «лёгкие» негативы и раздувало
#   дисбаланс); (2) LogregRanker.fit(C, class_weight, feature_names) — гиперпараметры и состав фичей
#   настраиваемы (feature_names — подмножество FEATURE_NAMES + синтетическая "street_sim", см.
#   _feature_value); coef/feature_names модели теперь пара переменной длины (не обязательно все
#   FEATURES) — save/load хранят и проверяют feature_names модели, а не жёстко глобальный FEATURE_NAMES.
#   Итог трёх прогонов (см. train_ranker.py --C/--class-weight/--neg-per-query) — таблица в отчёте
#   воркера; финальный выбор see meets_gates в ranker_model.json.
# END_CHANGE_SUMMARY

"""ranker.py — Ranker.score(features) -> p (docs/concept.md §5.4)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Контракт фичей (§5.4): имя -> значение при отсутствии сигнала. Общий для ManualRanker и
# LogregRanker (T-006) - оба принимают/производят словарь с этими ключами. sim_x_house (T-006) -
# interaction street_sim*house_in_list; ManualRanker её не использует (вес неявно 0), LogregRanker
# учит вес сам.
FEATURES: list[tuple[str, float]] = [
    ("lev", 0.0),
    ("token_set", 0.0),
    ("phon", 0.0),
    ("ngram", 0.0),
    ("alias_hit", 0.0),
    ("type_match", 0.5),
    ("city_match", 0.5),
    ("house_found", 0.0),
    ("house_in_list", 0.0),
    ("channel_voice", 0.0),
    ("name_freq", 0.0),
    ("n_close", 0.0),
    ("freq_x_city", 0.0),
    ("freq_x_phon", 0.0),
    ("voice_x_lev", 0.0),
    ("sim_x_house", 0.0),
]
FEATURE_NAMES: list[str] = [name for name, _ in FEATURES]
FEATURE_DEFAULTS: dict[str, float] = dict(FEATURES)

# T-006b: набор фичей для LogregRanker.fit(feature_names=...) без lev/token_set по отдельности —
# заменены агрегатной "street_sim" (синтетическая, см. _feature_value), phon/ngram оставлены как
# самостоятельные сигналы. Диагноз T-006: lev/phon/ngram/token_set сильно коррелируют (все меряют
# похожесть строки улицы) - LogisticRegression на 4 коррелированных входах даёт нестабильные веса
# (ngram уходил в +6.47 - VIF-эффект), не разрешая противоречивость по отдельности не может отличить.
FEATURE_NAMES_REDUCED_STREET: list[str] = [
    "street_sim",
    "phon",
    "ngram",
    "alias_hit",
    "type_match",
    "city_match",
    "house_found",
    "house_in_list",
    "channel_voice",
    "name_freq",
    "n_close",
    "freq_x_city",
    "freq_x_phon",
    "voice_x_lev",
    "sim_x_house",
]

# Стартовые ручные веса (§5.4): street_sim = 0.35*lev + 0.35*phon + 0.2*ngram + 0.1*token_set;
# z = 6*street_sim + 2.5*house_in_list + 1.5*city_match + 0.7*type_match - 0.8*(1-house_found) - 4.
# T-004 A4-докрутка: изначальная формула max(phon,lev) давала H2 (абляция no_phonetic) с обратным
# знаком - на этом словаре ASR-шум однобуквенный, ngram и lev уже сами ловят фонетически близкие
# варианты, phon был чистой избыточностью и иногда проигрывал (см. readme "где ломается"/H2).
# Взвешенная сумма даёт phon собственный вес независимо от lev - без потери top1 на A3-датасете.
DEFAULT_WEIGHTS: dict[str, float] = {
    "bias": -4.0,
    "street_sim": 6.0,
    "house_in_list": 2.5,
    "city_match": 1.5,
    "type_match": 0.7,
    "house_missing": -0.8,
    "name_freq": 0.0,
    "n_close": 0.0,
    "freq_x_city": 0.3,
    "freq_x_phon": 0.3,
    "voice_x_lev": 0.3,
    "alias_hit": 0.0,
}
_STREET_SIM_W = {"lev": 0.35, "phon": 0.35, "ngram": 0.2, "token_set": 0.1}


def street_sim(feats: dict[str, float]) -> float:
    """street_sim = 0.35*lev + 0.35*phon + 0.2*ngram + 0.1*token_set (§5.4). Публична: используется
    ManualRanker, LogregRanker (sim_x_house) и matcher.py/tools/train_ranker.py (сырой street-скор)."""
    return (
        _STREET_SIM_W["lev"] * feats.get("lev", 0.0)
        + _STREET_SIM_W["phon"] * feats.get("phon", 0.0)
        + _STREET_SIM_W["ngram"] * feats.get("ngram", 0.0)
        + _STREET_SIM_W["token_set"] * feats.get("token_set", 0.0)
    )


def _feature_value(name: str, feats: dict[str, float]) -> float:
    """Значение фичи по имени для LogregRanker (T-006b): "street_sim" - синтетическая агрегатная
    (street_sim(feats), не ключ feats), остальные - прямой feats.get с дефолтом из FEATURE_DEFAULTS.
    Позволяет LogregRanker.fit(feature_names=...) обучаться на подмножестве FEATURES + street_sim,
    не расширяя контракт feats, который производит matcher.py."""
    if name == "street_sim":
        return street_sim(feats)
    return feats.get(name, FEATURE_DEFAULTS.get(name, 0.0))


class ManualRanker:
    # START_CONTRACT: __init__
    #   PURPOSE: Ranker с ручными весами (fallback/baseline для LogregRanker, T-006).
    #   INPUTS: { weights: dict|None - переопределение DEFAULT_WEIGHTS (частичное) }
    #   OUTPUTS: none
    #   SIDE_EFFECTS: none
    # END_CONTRACT: __init__
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    # START_CONTRACT: score
    #   PURPOSE: Оценить объект-кандидат вероятностью "это верный объект" (§5.4).
    #   INPUTS: { feats: dict[str, float] - фичи по таблице FEATURES (отсутствующие берутся по умолчанию) }
    #   OUTPUTS: { float - p в [0,1] }
    #   SIDE_EFFECTS: none
    # END_CONTRACT: score
    def score(self, feats: dict[str, float]) -> float:
        # START_BLOCK_SCORE
        z = self._logit(feats)
        return 1.0 / (1.0 + math.exp(-z))
        # END_BLOCK_SCORE

    def _logit(self, feats: dict[str, float]) -> float:
        w = self.weights
        sim = street_sim(feats)
        z = w["bias"]
        z += w["street_sim"] * sim
        z += w["house_in_list"] * feats.get("house_in_list", 0.0)
        z += w["city_match"] * feats.get("city_match", 0.5)
        z += w["type_match"] * feats.get("type_match", 0.5)
        z += w["house_missing"] * (1.0 - feats.get("house_found", 0.0))
        z += w["name_freq"] * feats.get("name_freq", 0.0)
        z += w["n_close"] * feats.get("n_close", 0.0)
        z += w["freq_x_city"] * feats.get("freq_x_city", 0.0)
        z += w["freq_x_phon"] * feats.get("freq_x_phon", 0.0)
        z += w["voice_x_lev"] * feats.get("voice_x_lev", 0.0)
        z += w["alias_hit"] * feats.get("alias_hit", 0.0)
        return z

    # START_CONTRACT: explain
    #   PURPOSE: Разложить скор лидера по сигналам для MatchResult.explain (N6).
    #   INPUTS: { feats: dict[str, float] }
    #   OUTPUTS: { dict - marker, p, street_sim, contributions (вклад каждого слагаемого в logit) }
    #   SIDE_EFFECTS: none
    # END_CONTRACT: explain
    def explain(self, feats: dict[str, float]) -> dict[str, Any]:
        # START_BLOCK_EXPLAIN
        w = self.weights
        sim = street_sim(feats)
        contributions = {
            "bias": w["bias"],
            "street_sim": w["street_sim"] * sim,
            "house_in_list": w["house_in_list"] * feats.get("house_in_list", 0.0),
            "city_match": w["city_match"] * feats.get("city_match", 0.5),
            "type_match": w["type_match"] * feats.get("type_match", 0.5),
            "house_missing": w["house_missing"] * (1.0 - feats.get("house_found", 0.0)),
            "freq_x_city": w["freq_x_city"] * feats.get("freq_x_city", 0.0),
            "freq_x_phon": w["freq_x_phon"] * feats.get("freq_x_phon", 0.0),
            "voice_x_lev": w["voice_x_lev"] * feats.get("voice_x_lev", 0.0),
        }
        p = 1.0 / (1.0 + math.exp(-self._logit(feats)))
        return {
            "marker": "[Ranker][score][MANUAL]",
            "p": p,
            "street_sim": sim,
            "contributions": contributions,
            "feats": dict(feats),
        }
        # END_BLOCK_EXPLAIN
    # marker: [Ranker][ManualRanker][DONE]


# START_BLOCK_CALIBRATION
def _fit_one_calibrator(zs: np.ndarray, p_raw: np.ndarray, ys: np.ndarray) -> dict[str, Any]:
    """Platt (<40 точек) или изотоническая (>=40) калибровка одной страты/глобали (§5.4/H3a)."""
    n = len(ys)
    if n == 0 or len(set(ys.tolist())) < 2:
        # Нечем откалибровать (пусто или один класс) - тождественный калибратор поверх сырого p.
        return {"type": "identity", "n": n}
    if n >= 40:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_raw, ys)
        return {"type": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist(), "n": n}
    platt = LogisticRegression(C=1e6, max_iter=1000)
    platt.fit(zs.reshape(-1, 1), ys)
    return {"type": "platt", "a": float(platt.coef_[0][0]), "b": float(platt.intercept_[0]), "n": n}


def _apply_calibrator(calib: dict[str, Any], z: float, p_raw: float) -> float:
    kind = calib.get("type", "identity")
    if kind == "platt":
        zz = calib["a"] * z + calib["b"]
        return 1.0 / (1.0 + math.exp(-zz))
    if kind == "isotonic":
        xs, ys = calib["x"], calib["y"]
        return float(np.interp(p_raw, xs, ys, left=ys[0], right=ys[-1]))
    return p_raw
# END_BLOCK_CALIBRATION


def _stratum_key(feats: dict[str, float], freq_median: float) -> str:
    """Страта калибровки {channel} x {freq_class} x {city_status} (§5.4).

    freq_class сравнивается строгим ">" (не ">="): name_freq = log(1+count)/log(1+max_count) -
    большинство имён в учебной базе однообъектны (count=1), их значение и есть медиана freq по
    кандидатам, т.е. floor распределения. ">=" клал бы весь floor (~90% кандидатов) в freq_high и
    вырождал freq_low в пустую страту (проверено прогоном - без единой freq_low-точки на val)."""
    channel = "voice" if feats.get("channel_voice", 0.0) >= 0.5 else "webchat"
    freq_class = "freq_high" if feats.get("name_freq", 0.0) > freq_median else "freq_low"
    city_status = "resolved" if feats.get("city_match", 0.5) != 0.5 else "unresolved"
    return f"{channel}|{freq_class}|{city_status}"


def _build_training_examples(matcher: Any, rows: list[dict], neg_per_query: int = 10) -> list[dict[str, Any]]:
    """Прогнать конвейер (Matcher.candidate_features) по labeled-строкам -> обучающие примеры.

    Отсечка на строку: top-`neg_per_query` кандидатов по сырому street_sim (T-006b — было
    top-10 ∪ {street_sim>=0.5} для позитивных строк и ВСЕ кандидаты для негативных; объединение с
    порогом 0.5 и безлимитные негативы раздували число "лёгких" (уже далёких по рангу) негативов и
    усиливали дисбаланс 1:14.6, который class_weight=balanced сдвигал в пользу похожих-но-неверных
    кандидатов, см. docs/errors-a4.md). Позитив: кандидат, чей houses содержит etalon_id строки
    (y=1), остальные из top-K - y=0. Негатив labeled (etalon_id=None): top-K кандидатов как y=0
    (docs/concept.md §5.4, plan.xml T-006/T-006b).
    """
    # START_BLOCK_BUILD
    examples: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("raw_adress", "")
        channel = row.get("channel", "voice")
        slots = {"city": row.get("city", "")}
        etalon_id = row.get("etalon_id")

        pairs = matcher.candidate_features(raw, channel=channel, slots=slots)
        if not pairs:
            continue

        sims = sorted(range(len(pairs)), key=lambda i: -street_sim(pairs[i][1]))
        keep = set(sims[:neg_per_query])

        if etalon_id is None:
            for i in keep:
                _, feats = pairs[i]
                examples.append({"feats": feats, "y": 0, "id": row.get("id")})
            continue

        pos_idx = next((i for i, (oc, _) in enumerate(pairs) if etalon_id in oc.houses.values()), None)
        if pos_idx is None:
            # Позитив, которого конвейер вообще не нашёл среди top-k кандидатов - не обучающий пример
            # (тайм-боксовый ranker не может научиться доставать несуществующего кандидата).
            continue

        keep.add(pos_idx)
        for i in keep:
            _, feats = pairs[i]
            examples.append({"feats": feats, "y": 1 if i == pos_idx else 0, "id": row.get("id")})
    return examples
    # END_BLOCK_BUILD


class LogregRanker:
    # START_CONTRACT: __init__
    #   PURPOSE: Собрать LogregRanker из уже обученных параметров (используется fit() и load()).
    #   INPUTS: { coef: list[float] - веса по feature_names, intercept: float, freq_median: float -
    #             порог freq_class для страт калибровки, global_calibrator: dict|None, strata_calibrators:
    #             dict[str, dict]|None, seed: int, feature_names: list[str]|None - подмножество
    #             FEATURE_NAMES (+ синтетическая "street_sim", T-006b); по умолчанию — все FEATURE_NAMES }
    #   OUTPUTS: none
    #   SIDE_EFFECTS: none
    # END_CONTRACT: __init__
    def __init__(
        self,
        coef: list[float],
        intercept: float,
        freq_median: float = 0.0,
        global_calibrator: dict[str, Any] | None = None,
        strata_calibrators: dict[str, dict[str, Any]] | None = None,
        seed: int = 42,
        feature_names: list[str] | None = None,
    ) -> None:
        self.feature_names: list[str] = list(feature_names) if feature_names is not None else list(FEATURE_NAMES)
        if len(coef) != len(self.feature_names):
            raise ValueError(
                f"LogregRanker: coef длиной {len(coef)}, ожидалось {len(self.feature_names)} (feature_names)"
            )
        self.coef = list(coef)
        self.intercept = float(intercept)
        self.freq_median = float(freq_median)
        self.global_calibrator = global_calibrator or {"type": "identity", "n": 0}
        self.strata_calibrators = dict(strata_calibrators or {})
        self.seed = seed
        # meets_gates: выставляет tools/train_ranker.py по факту H3/H3a/H5 (§5.4, T-006 п.7 - при
        # срыве тайм-бокса конфиг должен остаться manual); сохраняется в JSON, run.py читает его при
        # выборе ранкера по умолчанию (--ranker не задан) - "logreg" только если файл есть И гейты
        # пройдены, иначе manual. Явный --ranker logreg игнорирует флаг (форсировать можно всегда).
        self.meets_gates: bool = False
        # Транзиентно (не сохраняется в JSON) - train_rows/val_rows/*_examples заполняет fit(),
        # нужны tools/train_ranker.py для метрик train/val и reliability_report по умолчанию.
        self.train_rows: list[dict] | None = None
        self.val_rows: list[dict] | None = None
        self._train_examples: list[dict[str, Any]] | None = None
        self._val_examples: list[dict[str, Any]] | None = None

    def _logit(self, feats: dict[str, float]) -> float:
        z = self.intercept
        for name, w in zip(self.feature_names, self.coef):
            z += w * _feature_value(name, feats)
        return z

    def _calibrator_for(self, feats: dict[str, float]) -> dict[str, Any]:
        key = _stratum_key(feats, self.freq_median)
        return self.strata_calibrators.get(key, self.global_calibrator)

    # START_CONTRACT: score
    #   PURPOSE: Оценить объект-кандидат вероятностью "это верный объект" (§5.4) — интерфейс ManualRanker.
    #   INPUTS: { feats: dict[str, float] }
    #   OUTPUTS: { float - калиброванная p в [0,1] }
    #   SIDE_EFFECTS: none
    # END_CONTRACT: score
    def score(self, feats: dict[str, float]) -> float:
        # START_BLOCK_SCORE
        z = self._logit(feats)
        p_raw = 1.0 / (1.0 + math.exp(-z))
        return _apply_calibrator(self._calibrator_for(feats), z, p_raw)
        # END_BLOCK_SCORE

    # START_CONTRACT: explain
    #   PURPOSE: Разложить скор по вкладам фичей + метаданные калибровки (интерфейс ManualRanker).
    #   INPUTS: { feats: dict[str, float] }
    #   OUTPUTS: { dict - marker, p, p_raw, z, stratum, calibrator, contributions, feats }
    #   SIDE_EFFECTS: none
    # END_CONTRACT: explain
    def explain(self, feats: dict[str, float]) -> dict[str, Any]:
        # START_BLOCK_EXPLAIN
        z = self._logit(feats)
        p_raw = 1.0 / (1.0 + math.exp(-z))
        key = _stratum_key(feats, self.freq_median)
        calib = self.strata_calibrators.get(key, self.global_calibrator)
        p = _apply_calibrator(calib, z, p_raw)
        contributions = {name: w * _feature_value(name, feats) for name, w in zip(self.feature_names, self.coef)}
        contributions["bias"] = self.intercept
        return {
            "marker": "[Ranker][score][LOGREG]",
            "p": p,
            "p_raw": p_raw,
            "z": z,
            "stratum": key,
            "calibrator": calib.get("type"),
            "contributions": contributions,
            "feats": dict(feats),
        }
        # END_BLOCK_EXPLAIN

    # START_CONTRACT: fit
    #   PURPOSE: Обучить LogregRanker на labeled-строках (§5.4, T-006/T-006b): сплит 75/25
    #            стратифицированный по (channel, is_negative), LogisticRegression(C, class_weight) на
    #            train, калибровка по стратам на val.
    #   INPUTS: { rows: list[dict] - adresses_labeled (id/channel/city/raw_adress/etalon_id), etalon:
    #             list[dict] - эталонный справочник, seed: int, C: float - обратная сила регуляризации
    #             sklearn LogisticRegression (T-006b, default 0.3 - лучший из 3 прогонов T-006b),
    #             class_weight: "balanced"|None - T-006b (default "balanced"), neg_per_query: int -
    #             отсечка кандидатов на строку в _build_training_examples (T-006b, default 10),
    #             feature_names: list[str]|None - подмножество FEATURE_NAMES (+"street_sim") для модели;
    #             по умолчанию FEATURE_NAMES_REDUCED_STREET (T-006b — street_sim агрегирует
    #             lev/token_set вместо их раздельного использования, снимает мультиколлинеарность) }
    #   OUTPUTS: { LogregRanker - обучен и откалиброван; .train_rows/.val_rows/._train_examples/
    #              ._val_examples заполнены для tools/train_ranker.py }
    #   SIDE_EFFECTS: none (Matcher строится только in-memory для извлечения фичей)
    # END_CONTRACT: fit
    @classmethod
    def fit(
        cls,
        rows: list[dict],
        etalon: list[dict],
        seed: int = 42,
        C: float = 0.3,
        class_weight: str | None = "balanced",
        neg_per_query: int = 10,
        feature_names: list[str] | None = None,
    ) -> "LogregRanker":
        # START_BLOCK_FIT
        # Отложенный импорт: matcher.py импортирует ranker.py на уровне модуля (ManualRanker/
        # LogregRanker) - импорт Matcher здесь на уровне модуля дал бы цикл импортов. К моменту
        # вызова fit() (из tools/train_ranker.py) addrmatch.matcher уже полностью загружен.
        from addrmatch.matcher import Matcher

        names = list(feature_names) if feature_names is not None else list(FEATURE_NAMES_REDUCED_STREET)
        cw = None if class_weight in (None, "none") else class_weight

        stratify_keys = [
            f"{row.get('channel', 'voice')}|{'neg' if row.get('etalon_id') is None else 'pos'}" for row in rows
        ]
        train_rows, val_rows = train_test_split(
            rows, test_size=0.25, random_state=seed, stratify=stratify_keys
        )

        # candidate_features не зависит от ranker - ManualRanker тут используется только затем,
        # что Matcher его требует в конструкторе; на обучающую выборку он не влияет.
        matcher = Matcher(etalon)
        train_examples = _build_training_examples(matcher, train_rows, neg_per_query=neg_per_query)
        val_examples = _build_training_examples(matcher, val_rows, neg_per_query=neg_per_query)

        X_train = np.array([[_feature_value(n, ex["feats"]) for n in names] for ex in train_examples])
        y_train = np.array([ex["y"] for ex in train_examples])

        clf = LogisticRegression(C=C, class_weight=cw, max_iter=1000, random_state=seed)
        clf.fit(X_train, y_train)

        freq_values = [ex["feats"].get("name_freq", 0.0) for ex in train_examples]
        freq_median = float(np.median(freq_values)) if freq_values else 0.0

        ranker = cls(
            coef=clf.coef_[0].tolist(),
            intercept=float(clf.intercept_[0]),
            freq_median=freq_median,
            seed=seed,
            feature_names=names,
        )
        ranker._fit_calibration(val_examples)
        ranker.train_rows = train_rows
        ranker.val_rows = val_rows
        ranker._train_examples = train_examples
        ranker._val_examples = val_examples
        return ranker
        # END_BLOCK_FIT

    def _fit_calibration(self, val_examples: list[dict[str, Any]]) -> None:
        # START_BLOCK_CALIBRATE
        def zs_praw_ys(exs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            zs = np.array([self._logit(ex["feats"]) for ex in exs]) if exs else np.array([])
            p_raw = 1.0 / (1.0 + np.exp(-zs)) if exs else np.array([])
            ys = np.array([ex["y"] for ex in exs]) if exs else np.array([])
            return zs, p_raw, ys

        zs, p_raw, ys = zs_praw_ys(val_examples)
        self.global_calibrator = _fit_one_calibrator(zs, p_raw, ys)

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ex in val_examples:
            groups[_stratum_key(ex["feats"], self.freq_median)].append(ex)

        strata_calibrators: dict[str, dict[str, Any]] = {}
        for key, exs in groups.items():
            if len(exs) < 15:
                continue  # мало точек - страта пользуется global_calibrator (§5.4)
            zs_s, p_raw_s, ys_s = zs_praw_ys(exs)
            strata_calibrators[key] = _fit_one_calibrator(zs_s, p_raw_s, ys_s)
        self.strata_calibrators = strata_calibrators
        # END_BLOCK_CALIBRATE

    # START_CONTRACT: reliability_report
    #   PURPOSE: Бины надёжности (mean_pred vs frac_pos) по стратам на val для гейта H3a.
    #   INPUTS: { rows_val: list[dict]|None - примеры {feats, y}; по умолчанию self._val_examples (fit()) }
    #   OUTPUTS: { str - таблица (TSV) strata/n/bin/n_bin/mean_pred/frac_pos/abs_dev + строка
    #              GLOBAL_MAX_DEV с максимальным |отклонением| среди бинов с n_bin >= 10 }
    #   SIDE_EFFECTS: none
    # END_CONTRACT: reliability_report
    def reliability_report(self, rows_val: list[dict[str, Any]] | None = None) -> str:
        # START_BLOCK_REPORT
        examples = rows_val if rows_val is not None else (self._val_examples or [])
        if not examples:
            return "reliability_report: нет данных (val пуст)\nGLOBAL_MAX_DEV\t0.0000"

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ex in examples:
            groups[_stratum_key(ex["feats"], self.freq_median)].append(ex)

        lines = ["strata\tn\tbin\tn_bin\tmean_pred\tfrac_pos\tabs_dev"]
        global_max_dev = 0.0
        for key in sorted(groups):
            exs = groups[key]
            preds = [self.score(ex["feats"]) for ex in exs]
            ys = [ex["y"] for ex in exs]
            bins: dict[int, list[tuple[float, int]]] = defaultdict(list)
            for p, y in zip(preds, ys):
                idx = min(int(p * 10), 9)
                bins[idx].append((p, y))
            strata_max_dev = 0.0
            for b in range(10):
                items = bins.get(b, [])
                if not items:
                    continue
                n_bin = len(items)
                mean_pred = sum(p for p, _ in items) / n_bin
                frac_pos = sum(y for _, y in items) / n_bin
                dev = abs(mean_pred - frac_pos)
                if n_bin >= 10:
                    strata_max_dev = max(strata_max_dev, dev)
                lines.append(f"{key}\t{len(exs)}\t{b}\t{n_bin}\t{mean_pred:.3f}\t{frac_pos:.3f}\t{dev:.3f}")
            global_max_dev = max(global_max_dev, strata_max_dev)
            lines.append(f"{key}\t{len(exs)}\tSTRATA_MAX_DEV(n>=10)\t\t\t\t{strata_max_dev:.3f}")
        lines.append(f"GLOBAL_MAX_DEV\t{global_max_dev:.4f}")
        return "\n".join(lines)
        # END_BLOCK_REPORT

    # START_CONTRACT: save
    #   PURPOSE: Сохранить модель в JSON (веса, калибраторы, версия) — без pickle (technology.xml).
    #   INPUTS: { path: str }
    #   OUTPUTS: none
    #   SIDE_EFFECTS: запись файла на диск (UTF-8)
    # END_CONTRACT: save
    def save(self, path: str) -> None:
        # START_BLOCK_SAVE
        payload = {
            "version": 1,
            "feature_names": self.feature_names,
            "coef": self.coef,
            "intercept": self.intercept,
            "freq_median": self.freq_median,
            "global_calibrator": self.global_calibrator,
            "strata_calibrators": self.strata_calibrators,
            "seed": self.seed,
            "meets_gates": bool(self.meets_gates),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        # END_BLOCK_SAVE

    # START_CONTRACT: load
    #   PURPOSE: Загрузить LogregRanker из JSON, сохранённого save().
    #   INPUTS: { path: str }
    #   OUTPUTS: { LogregRanker }
    #   SIDE_EFFECTS: чтение файла с диска
    # END_CONTRACT: load
    @classmethod
    def load(cls, path: str) -> "LogregRanker":
        # START_BLOCK_LOAD
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        feature_names = payload.get("feature_names") or FEATURE_NAMES
        # T-006b: модель может быть обучена на подмножестве FEATURE_NAMES (+ синтетическая
        # "street_sim") - проверяем, что каждое имя известно текущему коду, а не точное совпадение
        # с полным FEATURE_NAMES (было в T-006, когда LogregRanker всегда использовал все FEATURES).
        valid_names = set(FEATURE_NAMES) | {"street_sim"}
        if not all(name in valid_names for name in feature_names):
            raise ValueError("LogregRanker.load: в модели есть неизвестное имя фичи (не FEATURE_NAMES/street_sim)")
        ranker = cls(
            coef=payload["coef"],
            intercept=payload["intercept"],
            freq_median=payload.get("freq_median", 0.0),
            global_calibrator=payload.get("global_calibrator"),
            strata_calibrators=payload.get("strata_calibrators"),
            seed=payload.get("seed", 42),
            feature_names=feature_names,
        )
        ranker.meets_gates = bool(payload.get("meets_gates", False))
        return ranker
        # END_BLOCK_LOAD
    # marker: [Ranker][LogregRanker][DONE]
