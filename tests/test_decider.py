# START_MODULE_CONTRACT
#   PURPOSE: Тесты Decider — пороги из N, порядок правил §5.5, все семь исходов на синтетике,
#            проекция to_tz в формат ТЗ (docs/concept.md §5.5, T-004).
#   SCOPE: Юнит-тесты на простых dict/SimpleNamespace (без Index/Ranker/Parser).
#   DEPENDS: M-DECIDER
#   LINKS: V-M-DECIDER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_thresholds_derived_from_N - тест
#   test_reject_without_cues_and_no_asked_slot - тест
#   test_answer_high_confidence - тест
#   test_answer_soft_mid_confidence - тест
#   test_confirm_low_but_above_theta_confirm - тест
#   test_ask_house_when_house_not_in_list - тест
#   test_ask_street_when_city_resolved_but_leader_weak - тест
#   test_reject_when_city_unresolved_and_leader_weak - тест
#   test_ask_city_wins_over_confirm_for_same_name_two_cities - тест
#   test_to_tz_ask_house_gives_up_to_3_etalons_of_one_street - тест
#   test_to_tz_reject_and_ask_street_are_empty - тест
#   test_to_tz_empty_ranked_is_empty - тест
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-004: test_decider.py — пороги из N, все исходы, ask_city раньше confirm
#   при одноимённой улице в двух городах, проекция to_tz (ask_house <=3 эталона, reject -> []).
# END_CHANGE_SUMMARY

"""tests/test_decider.py — docs/concept.md §5.5."""

from __future__ import annotations

from types import SimpleNamespace

from addrmatch.decider import Decider, to_tz


def _parse(has_cues=True, ambiguous=False, house=None):
    return SimpleNamespace(has_address_cues=has_cues, ambiguous_number=ambiguous, house=house)


def _city(status="resolved", city_id="москва"):
    return SimpleNamespace(status=status, city_id=city_id)


def _cand(name_id=0, city_id="москва", p=0.9, houses=None, name_freq=0.5, street_id=None, street_type="улица"):
    return {
        "street_id": street_id or f"{city_id}||{street_type}||{name_id}",
        "name_id": name_id,
        "name": "ленина",
        "city_id": city_id,
        "street_type": street_type,
        "houses": houses if houses is not None else {"1": "e_0001", "2": "e_0002"},
        "p": p,
        "name_freq": name_freq,
    }


# START_BLOCK_THRESHOLDS
def test_thresholds_derived_from_N():
    d10 = Decider(N=10)
    assert abs(d10.theta_answer - 10 / 11) < 1e-9
    d4 = Decider(N=4)
    assert abs(d4.theta_answer - 4 / 5) < 1e-9
    assert d10.theta_answer > d10.theta_soft > d10.theta_confirm
# END_BLOCK_THRESHOLDS


# START_BLOCK_OUTCOMES
def test_reject_without_cues_and_no_asked_slot():
    d = Decider(N=10)
    res = d.decide(ranked=[], city_res=_city(status="none"), parse=_parse(has_cues=False))
    assert res.decision == "reject"
    assert res.candidates == []


def test_answer_high_confidence():
    d = Decider(N=10)  # theta_answer ~= 0.909
    ranked = [_cand(p=0.97, houses={"1": "e_0001"})]
    res = d.decide(ranked=ranked, city_res=_city(), parse=_parse(house="1"), house_norm="1")
    assert res.decision == "answer"
    assert res.candidates[0] == "e_0001"
    assert res.slots_resolved["house"] == "1"


def test_answer_soft_mid_confidence():
    d = Decider(N=10)  # theta_soft ~= 0.857, theta_answer ~= 0.909
    ranked = [_cand(p=0.88, houses={"1": "e_0001"})]
    res = d.decide(ranked=ranked, city_res=_city(), parse=_parse(house="1"), house_norm="1")
    assert res.decision == "answer_soft"


def test_confirm_low_but_above_theta_confirm():
    d = Decider(N=10)  # theta_confirm = 0.6
    ranked = [_cand(p=0.65, houses={"1": "e_0001"})]
    res = d.decide(ranked=ranked, city_res=_city(), parse=_parse(house="1"), house_norm="1")
    assert res.decision == "confirm"


def test_ask_house_when_house_not_in_list():
    d = Decider(N=10)
    ranked = [_cand(p=0.8, houses={"1": "e_0001", "2": "e_0002", "3": "e_0003", "4": "e_0004"})]
    res = d.decide(ranked=ranked, city_res=_city(), parse=_parse(house="99"), house_norm="99")
    assert res.decision == "ask_house"
    assert res.missing_slot == "house"
    assert len(res.options) <= 3
    assert set(res.candidates) <= {"e_0001", "e_0002", "e_0003", "e_0004"}


def test_ask_street_when_city_resolved_but_leader_weak():
    d = Decider(N=10)
    ranked = [_cand(p=0.3, houses={"1": "e_0001"})]
    res = d.decide(ranked=ranked, city_res=_city(status="resolved"), parse=_parse(house=None), house_norm=None)
    assert res.decision == "ask_street"
    assert res.candidates == []


def test_reject_when_city_unresolved_and_leader_weak():
    d = Decider(N=10)
    ranked = [_cand(p=0.3, city_id="москва", houses={"1": "e_0001"})]
    res = d.decide(ranked=ranked, city_res=_city(status="unresolved"), parse=_parse(house=None), house_norm=None)
    assert res.decision == "reject"
    assert res.candidates == []
# END_BLOCK_OUTCOMES


# START_BLOCK_RULE_ORDER
def test_ask_city_wins_over_confirm_for_same_name_two_cities():
    """Одноимённая частая улица в двух городах, город не разрешён -> ask_city, а не confirm/answer,
    даже если p лидера уже выше theta_confirm (порядок правил §5.5: ask_city перед answer*/confirm)."""
    d = Decider(N=10)
    ranked = [
        _cand(name_id=5, city_id="москва", p=0.95, name_freq=0.7, houses={"1": "e_0001"}),
        _cand(name_id=5, city_id="казань", p=0.6, name_freq=0.7, houses={"1": "e_0101"}),
    ]
    res = d.decide(ranked=ranked, city_res=_city(status="none"), parse=_parse(house="1"), house_norm="1")
    assert res.decision == "ask_city"
    assert res.missing_slot == "city"
    assert set(res.options) <= {"москва", "казань"}
    assert len(res.options) <= 3
# END_BLOCK_RULE_ORDER


# START_BLOCK_TO_TZ
def test_to_tz_ask_house_gives_up_to_3_etalons_of_one_street():
    ranked = [_cand(p=0.8, houses={"1": "e_0001", "2": "e_0002", "3": "e_0003", "4": "e_0004", "5": "e_0005"})]
    cands, scores = to_tz("ask_house", ranked)
    assert len(cands) <= 3
    assert len(cands) == len(scores)
    assert set(cands) <= {"e_0001", "e_0002", "e_0003", "e_0004", "e_0005"}


def test_to_tz_reject_and_ask_street_are_empty():
    ranked = [_cand(p=0.9)]
    assert to_tz("reject", ranked) == ([], [])
    assert to_tz("ask_street", ranked) == ([], [])


def test_to_tz_empty_ranked_is_empty():
    assert to_tz("answer", []) == ([], [])
# END_BLOCK_TO_TZ
