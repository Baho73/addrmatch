# START_MODULE_CONTRACT
#   PURPOSE: Оценить объект-кандидат (имя × город × тип) вероятностью "это верный объект" (§5.4).
#   SCOPE: Таблица фичей FEATURES (контракт, общий с будущим LogregRanker из T-006) + ManualRanker
#          - ручные веса через сигмоиду, explain() раскладывает скор по сигналам.
#   DEPENDS: none (принимает готовый словарь фичей; ничего не знает про Index/Parser)
#   LINKS: V-M-RANKER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   FEATURES - список (имя, значение-по-умолчанию) - контракт фичей Ranker (§5.4)
#   ManualRanker - score(feats)->p, explain(feats)->разложение, ручные веса + сигмоида
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-004: ranker.py — ManualRanker (street_sim + house/city/type + interaction),
#   стартовые веса докрутки docs/concept.md §5.4.
# END_CHANGE_SUMMARY

"""ranker.py — Ranker.score(features) -> p (docs/concept.md §5.4)."""

from __future__ import annotations

import math
from typing import Any

# Контракт фичей (§5.4): имя -> значение при отсутствии сигнала. Общий для ManualRanker и
# будущего LogregRanker (T-006) - оба принимают/производят словарь с этими ключами.
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


def _street_sim(feats: dict[str, float]) -> float:
    return (
        _STREET_SIM_W["lev"] * feats.get("lev", 0.0)
        + _STREET_SIM_W["phon"] * feats.get("phon", 0.0)
        + _STREET_SIM_W["ngram"] * feats.get("ngram", 0.0)
        + _STREET_SIM_W["token_set"] * feats.get("token_set", 0.0)
    )


class ManualRanker:
    # START_CONTRACT: __init__
    #   PURPOSE: Ranker с ручными весами (fallback/baseline для будущего LogregRanker, T-006).
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
        street_sim = _street_sim(feats)
        z = w["bias"]
        z += w["street_sim"] * street_sim
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
        street_sim = _street_sim(feats)
        contributions = {
            "bias": w["bias"],
            "street_sim": w["street_sim"] * street_sim,
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
            "street_sim": street_sim,
            "contributions": contributions,
            "feats": dict(feats),
        }
        # END_BLOCK_EXPLAIN
    # marker: [Ranker][ManualRanker][DONE]
