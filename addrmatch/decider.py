# START_MODULE_CONTRACT
#   PURPOSE: Выбрать исход диалога (answer/answer_soft/confirm/ask_city/ask_house/ask_street/
#            reject) по ранжированным объектам и спроецировать его в формат ТЗ (§5.5).
#   SCOPE: Чистые функции/класс над простыми структурами (list[dict], CityResolution-подобный
#          объект по duck-typing) - без импорта из index/ranker/parser (N5-подобная изоляция).
#   DEPENDS: none
#   LINKS: V-M-DECIDER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   DecisionResult - decision/candidates/scores/missing_slot/options/slots_resolved/rule
#   to_tz - decision + ranked -> (candidates<=3, scores) по таблице проекции §5.5
#   Decider - decide(): пороги из N, порядок правил §5.5, rule = номер сработавшего правила
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-004: decider.py — пороги answer/answer_soft/confirm из N, порядок
#   правил reject->ask_city->ask_house->answer*->ask_street->reject, проекция to_tz (§5.5).
#   C-ADDRMATCH-PHASE-A T-005: ask_house (правило 3) требует city_res.status=="resolved";
#   без разрешённого города и неподтверждённого дома решение падает в ask_street/reject вместо
#   угадывания дома (docs/errors-a4.md: 19 accepted_negative на нерешённом городе, top1 не задет).
#   C-ADDRMATCH-PHASE-A T-007: (1) DecisionResult.rule — номер сработавшего правила §5.5 (1..6),
#   Matcher кладёт его в explain["decision_rule"]/explain["marker"]. (2) _should_ask_city — минимальный
#   порог уверенности лидера-имени (street_sim>=0.75 или p>=theta_confirm), иначе правило 2 не
#   срабатывает и негатив с похожим-но-неверным частым именем уходит дальше в ask_street/reject, а
#   не в ложный ask_city (найдено на T-006 — без порога любой мусор с частым именем давал ask_city).
# END_CHANGE_SUMMARY

"""decider.py — RankedList + CityResolution + ParseResult -> решение диалога (docs/concept.md §5.5).

Вход `ranked` — список простых словарей (не ObjCand из index.py, чтобы не тянуть DEPENDS):
    {"street_id": str, "name_id": int, "name": str, "city_id": str, "street_type": str,
     "houses": dict[str, str], "p": float, "name_freq": float}
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecisionResult:
    """Результат Decider.decide() — проекция уже посчитана (candidates/scores по §5.5)."""

    decision: str
    candidates: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    missing_slot: str | None = None
    options: list[Any] = field(default_factory=list)
    slots_resolved: dict[str, Any] = field(default_factory=dict)
    rule: int = 0  # T-007: номер сработавшего правила §5.5 (1..6), для explain["decision_rule"]


# START_CONTRACT: to_tz
#   PURPOSE: Спроецировать decision+ranked в формат ТЗ: candidates(<=3 etalon_id) + scores (§5.5).
#   INPUTS: { decision: str, ranked: list[dict] - см. шапку модуля, house_norm: str|None -
#             распознанный дом запроса (norm_house), city_options: list[str]|None - city_id для ask_city }
#   OUTPUTS: { tuple[list[str], list[float]] }
#   SIDE_EFFECTS: none
# END_CONTRACT: to_tz
def to_tz(
    decision: str,
    ranked: list[dict],
    house_norm: str | None = None,
    city_options: list[str] | None = None,
) -> tuple[list[str], list[float]]:
    # START_BLOCK_PROJECT
    if decision in ("ask_street", "reject") or not ranked:
        return [], []

    ranked_sorted = sorted(ranked, key=lambda r: -r["p"])
    leader = ranked_sorted[0]
    second = ranked_sorted[1] if len(ranked_sorted) > 1 else None

    if decision == "ask_house":
        house_keys = sorted(leader["houses"].keys())[:3]
        cands = [leader["houses"][h] for h in house_keys]
        return cands, [leader["p"]] * len(cands)

    if decision == "ask_city":
        cands: list[str] = []
        scores: list[float] = []
        for city_id in (city_options or [])[:3]:
            match = next(
                (r for r in ranked_sorted if r["city_id"] == city_id and r["name_id"] == leader["name_id"]),
                None,
            )
            if match is None or not match["houses"]:
                continue
            house_key = house_norm if house_norm in match["houses"] else next(iter(sorted(match["houses"])), None)
            cands.append(match["houses"][house_key])
            scores.append(match["p"])
        return cands, scores

    # answer / answer_soft / confirm: дом лидера (совпавший первым), потом другие дома лидера
    # по алфавиту номера, потом дома второго объекта (§5.5).
    cands = []
    scores = []
    matched_key = house_norm if house_norm in leader["houses"] else None
    if matched_key:
        cands.append(leader["houses"][matched_key])
        scores.append(leader["p"])
    for h in sorted(leader["houses"].keys()):
        if len(cands) >= 3:
            break
        if h == matched_key:
            continue
        cands.append(leader["houses"][h])
        scores.append(leader["p"] * 0.8)
    if len(cands) < 3 and second is not None:
        for h in sorted(second["houses"].keys()):
            if len(cands) >= 3:
                break
            cands.append(second["houses"][h])
            scores.append(second["p"])
    return cands[:3], scores[:3]
    # END_BLOCK_PROJECT


class Decider:
    # START_CONTRACT: __init__
    #   PURPOSE: Зафиксировать стоимость ошибки N и пороги решений (§5.5).
    #   INPUTS: { N: float - "один ложный answer = N переспросов" (N4), c_confirm: float - память
    #             стоимости confirm (не используется в пороге, зафиксирован 0.6 по брифу T-004),
    #             c_soft: float - снижение цены ошибки для answer_soft }
    #   OUTPUTS: none
    #   SIDE_EFFECTS: none
    # END_CONTRACT: __init__
    def __init__(self, N: float = 10, c_confirm: float = 0.5, c_soft: float = 0.3) -> None:
        self.N = N
        self.c_confirm = c_confirm
        self.c_soft = c_soft
        self.theta_answer = N / (N + 1.0)
        self.theta_soft = 1.0 - 1.0 / (N * (1.0 - c_soft))
        self.theta_confirm = 0.6  # зафиксировано по брифу T-004 (не выводится из c_confirm)

    def _should_ask_city(self, leader: dict, ranked_sorted: list[dict], city_res: Any) -> bool:
        if getattr(city_res, "status", "none") not in ("none", "unresolved", "ambiguous"):
            return False
        same_name = [r for r in ranked_sorted if r["name_id"] == leader["name_id"]]
        if len({r["city_id"] for r in same_name}) <= 1:
            return False
        median_freq = statistics.median(r["name_freq"] for r in ranked_sorted)
        if leader["name_freq"] < median_freq:
            return False
        # T-007: минимальный порог уверенности лидера-имени - без него мусор с похожей-но-неверной
        # частой улицей уходил в ask_city вместо ask_street/reject (докрутка T-006, см. CHANGE_SUMMARY).
        return leader.get("street_sim", 0.0) >= 0.75 or leader["p"] >= self.theta_confirm

    def _city_options(self, leader: dict, ranked_sorted: list[dict]) -> list[str]:
        same_name = sorted(
            (r for r in ranked_sorted if r["name_id"] == leader["name_id"]), key=lambda r: -r["p"]
        )
        seen: list[str] = []
        for r in same_name:
            if r["city_id"] not in seen:
                seen.append(r["city_id"])
            if len(seen) >= 3:
                break
        return seen

    @staticmethod
    def _house_ok(leader: dict, house_norm: str | None, ambiguous_number: bool) -> bool:
        return bool(house_norm) and not ambiguous_number and house_norm in leader["houses"]

    # START_CONTRACT: decide
    #   PURPOSE: Выбрать исход диалога по порядку правил §5.5 и спроецировать в формат ТЗ.
    #   INPUTS: { ranked: list[dict] - объекты+p (см. шапку модуля), city_res: CityResolution-
    #             подобный объект (status/city_id), parse: ParseResult-подобный объект
    #             (has_address_cues/ambiguous_number), asked_slot: str|None, house_norm: str|None -
    #             norm_house(parse.house) }
    #   OUTPUTS: { DecisionResult }
    #   SIDE_EFFECTS: none
    # END_CONTRACT: decide
    def decide(
        self,
        ranked: list[dict],
        city_res: Any,
        parse: Any,
        asked_slot: str | None = None,
        house_norm: str | None = None,
    ) -> DecisionResult:
        # START_BLOCK_RULES
        has_cues = bool(getattr(parse, "has_address_cues", False))
        ambiguous_number = bool(getattr(parse, "ambiguous_number", False))
        ranked_sorted = sorted(ranked, key=lambda r: -r["p"]) if ranked else []
        leader = ranked_sorted[0] if ranked_sorted else None

        decision: str
        rule: int
        missing_slot: str | None = None
        options: list[Any] = []
        slots_resolved: dict[str, Any] = {}

        house_ok = leader is not None and self._house_ok(leader, house_norm, ambiguous_number)

        if not has_cues and asked_slot is None:
            # Правило 1: reject по отсутствию cues.
            decision, rule = "reject", 1
        elif leader is not None and self._should_ask_city(leader, ranked_sorted, city_res):
            # Правило 2: ask_city.
            decision, rule = "ask_city", 2
            missing_slot = "city"
            options = self._city_options(leader, ranked_sorted)
        elif leader is not None and leader["p"] >= self.theta_confirm and house_ok:
            # Правило 4: answer / answer_soft / confirm (дом подтверждён — проверяется раньше
            # ask_house, порядок между 3/4 не влияет на исход, т.к. условия по house_ok
            # взаимоисключающие; см. ниже T-005 про требование resolved у Правила 3).
            rule = 4
            p1 = leader["p"]
            if p1 >= self.theta_answer:
                decision = "answer"
            elif p1 >= self.theta_soft:
                decision = "answer_soft"
            else:
                decision = "confirm"
            matched_house = house_norm if house_norm in leader["houses"] else next(iter(leader["houses"]), None)
            slots_resolved = {"city_id": leader["city_id"], "street_id": leader["street_id"], "house": matched_house}
        elif (
            leader is not None
            and leader["p"] >= self.theta_confirm
            and getattr(city_res, "status", "none") == "resolved"
        ):
            # Правило 3: ask_house. T-005-докрутка: раньше срабатывало при любом city_status,
            # включая none/unresolved - на labeled 19 негативов (город не из эталона/не
            # разрешён) получали house-опции и засчитывались как "принятый" ответ (reject_recall
            # 0.65->35 accepted_negative). На всех 400 позитивах ни один настоящий ask_house не
            # требует неразрешённого города (city_match=0.5 не встречается среди позитивов с
            # house_ok=False) - гейт по city_status=="resolved" не стоил top1 ни одного случая
            # (docs/errors-a4.md). Без разрешённого города дом уточнять не у чего - падаем в
            # Правило 5/6 (ask_street/reject), безопасный отказ вместо угадывания дома.
            decision, rule = "ask_house", 3
            missing_slot = "house"
            options = sorted(leader["houses"].keys())[:3]
        elif getattr(city_res, "status", "none") == "resolved" and has_cues:
            # Правило 5: ask_street.
            decision, rule = "ask_street", 5
            missing_slot = "street"
        else:
            # Правило 6: reject (остальное, включая city не resolved + дом не подтверждён).
            decision, rule = "reject", 6

        candidates, scores = to_tz(
            decision, ranked_sorted, house_norm=house_norm, city_options=options if decision == "ask_city" else None
        )
        return DecisionResult(
            decision=decision,
            candidates=candidates,
            scores=scores,
            missing_slot=missing_slot,
            options=options,
            slots_resolved=slots_resolved,
            rule=rule,
        )
        # END_BLOCK_RULES
    # marker: [Decider][decide][DONE]
