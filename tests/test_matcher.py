# START_MODULE_CONTRACT
#   PURPOSE: Тесты Matcher.match() — семь исходов на реальном эталоне, порядок правил, слоты
#            id/текст (T-007, F1/F4), проекция to_tz (<=3, пусто для ask_street/reject), контракт
#            "никогда не бросает" (F5).
#   SCOPE: Юнит/интеграционные тесты поверх реального test_task_adress_match/data (7 исходов) +
#          маленькие синтетические эталоны там, где нужна детерминированность (порядок правил,
#          слот-текст vs city_hint, слот-id улицы + asked_slot=house).
#   DEPENDS: M-MATCHER
#   LINKS: V-M-MATCHER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_answer_real_kirovskaya - тест
#   test_reject_no_address_cues_real - тест
#   test_ask_city_real_mira - тест
#   test_ask_house_real - тест
#   test_confirm_real - тест
#   test_answer_soft_real - тест
#   test_ask_street_or_reject_real - тест
#   test_ask_city_wins_over_confirm_for_same_name_two_cities - тест
#   test_to_tz_projection_le_3_and_empty_for_reject_ask_street - тест
#   test_never_throws_on_garbage_raw - тест
#   test_never_throws_on_garbage_slots_asked_slot_scope - тест
#   test_never_throws_on_broken_etalon_ready_false - тест
#   test_slot_city_text_wins_over_city_hint_from_raw - тест
#   test_slot_city_id_narrows_objects_to_that_city - тест
#   test_slot_street_id_plus_asked_house_word_number_gives_answer - тест
#   test_forced_house_cycle_real - тест
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-007: test_matcher.py — по одному тесту на каждый из 7 исходов (реальные
#   строки labeled/etalon), тест порядка правил (ask_city раньше confirm), проекция to_tz,
#   "никогда не бросает" (raw=None/123, slots/asked_slot/scope мусор, битый etalon), слоты
#   city/street текст vs id (city text побеждает city_hint, city-id сужает объекты через scope,
#   street-id + asked_slot="house" + числительное словом -> answer).
#   C-ADDRMATCH-PHASE-A T-007b: + тест полного цикла forced_house (tools/simulate_dialog.strip_house)
#   на реальной строке a_0425 «ул. таганская 8п/2» — срезать дом -> ask_house -> второй вызов с
#   slots-id улицы/города, asked_slot="house", raw=истинный дом -> answer с верным candidates[0].
# END_CHANGE_SUMMARY

"""tests/test_matcher.py — docs/concept.md §3 F1/F2/F4, §5.5; plan.xml T-007."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from addrmatch.matcher import Matcher

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


@pytest.fixture(scope="module")
def etalon() -> list[dict]:
    return _load_jsonl(_ETALON_PATH)


@pytest.fixture(scope="module")
def labeled() -> list[dict]:
    return _load_jsonl(_LABELED_PATH)


@pytest.fixture(scope="module")
def matcher(etalon: list[dict]) -> Matcher:
    # ranker="manual" явно - независимо от того, прошла ли ranker_model.json гейты (T-006/T-006b).
    return Matcher(etalon, ranker="manual")


def _row(labeled: list[dict], row_id: str) -> dict:
    return next(r for r in labeled if r["id"] == row_id)


def _match_row(matcher: Matcher, row: dict):
    return matcher.match(row["raw_adress"], channel=row.get("channel", "voice"), slots={"city": row.get("city", "")})


# START_BLOCK_SEVEN_OUTCOMES
# По одному тесту на каждый из 7 исходов (docs/concept.md F2) на реальных строках
# test_task_adress_match/data/adresses_labeled.jsonl + etalon.jsonl.

def test_answer_real_kirovskaya(matcher: Matcher, labeled: list[dict]) -> None:
    """a_0193 «это киравсгой 16» (НН) — design-context.xml сценарий 1: фонетика -> answer."""
    row = _row(labeled, "a_0193")
    r = _match_row(matcher, row)
    assert r.decision == "answer"
    assert r.candidates[:1] == [row["etalon_id"]]
    assert len(r.candidates) == len(r.scores) <= 3
    assert r.missing_slot is None
    assert set(r.slots_resolved) == {"city_id", "street_id", "house"}
    assert r.explain["decision_rule"] == 4
    assert r.explain["marker"] == "[Decider][decide][RULE_4]"


def test_reject_no_address_cues_real(matcher: Matcher, labeled: list[dict]) -> None:
    """a_0382 «да куда скажете туда и приду» — design-context.xml сценарий 4: не про адрес."""
    row = _row(labeled, "a_0382")
    r = _match_row(matcher, row)
    assert r.decision == "reject"
    assert r.candidates == [] and r.scores == []
    assert r.missing_slot is None
    assert r.explain["decision_rule"] == 1
    assert r.explain["marker"] == "[Decider][decide][RULE_1]"


def test_ask_city_real_mira(matcher: Matcher, labeled: list[dict]) -> None:
    """a_0267 «я живу на улица мира 27» — design-context.xml сценарий 2 (город пуст): «Мира»
    есть в 4 городах эталона -> ask_city. Город слота обнулён намеренно (в самой строке a_0267
    город уже указан верно — сценарий воспроизводит случай, когда извлечение города выше по
    пайплайну не сработало, ровно как в design-context.xml)."""
    row = _row(labeled, "a_0267")
    r = matcher.match(row["raw_adress"], channel=row["channel"], slots={"city": ""})
    assert r.decision == "ask_city"
    assert r.missing_slot == "city"
    assert 1 <= len(r.options) <= 3
    for opt in r.options:
        assert set(opt) == {"id", "name"}
    assert len(r.candidates) == len(r.scores) <= 3
    assert r.explain["decision_rule"] == 2


def test_ask_house_real(matcher: Matcher, labeled: list[dict]) -> None:
    """a_0192 «...проспект ленинградский три один» — улица уверенная, дом неоднозначен -> ask_house."""
    row = _row(labeled, "a_0192")
    r = _match_row(matcher, row)
    assert r.decision == "ask_house"
    assert r.missing_slot == "house"
    assert r.slots_resolved == {}  # §5.5: slots_resolved только для answer*/confirm
    assert 1 <= len(r.options) <= 3
    assert all(isinstance(h, str) for h in r.options)
    assert len(r.candidates) == len(r.scores) <= 3
    assert r.explain["decision_rule"] == 3


def test_confirm_real(matcher: Matcher, labeled: list[dict]) -> None:
    """a_0396 «у меня адрес 3-й дом три», город «Кстово» не из эталона — p в поясе confirm."""
    row = _row(labeled, "a_0396")
    r = _match_row(matcher, row)
    assert r.decision == "confirm"
    assert r.missing_slot is None
    assert set(r.slots_resolved) == {"city_id", "street_id", "house"}
    assert len(r.candidates) == len(r.scores) <= 3
    assert r.explain["decision_rule"] == 4


def test_answer_soft_real(matcher: Matcher, labeled: list[dict]) -> None:
    """a_0434 «новосибирск большевистскя 171», город в слоте перепутан — p в поясе answer_soft."""
    row = _row(labeled, "a_0434")
    r = _match_row(matcher, row)
    assert r.decision == "answer_soft"
    assert r.candidates[:1] == [row["etalon_id"]]
    assert set(r.slots_resolved) == {"city_id", "street_id", "house"}
    assert r.explain["decision_rule"] == 4


def test_ask_street_or_reject_real(matcher: Matcher, labeled: list[dict]) -> None:
    """a_0485 «удобно ул. новоселов 67» — design-context.xml сценарий 5: имени нет в словаре;
    ask_street и reject проецируются в одинаковый пустой список кандидатов (§5.5)."""
    row = _row(labeled, "a_0485")
    r = _match_row(matcher, row)
    assert r.decision in ("ask_street", "reject")
    assert r.candidates == [] and r.scores == []
# END_BLOCK_SEVEN_OUTCOMES


# START_BLOCK_RULE_ORDER
_TWO_CITY_ETALON = [
    {"etalon_id": "ro_1", "city": "Москва", "street_type": "улица", "street": "Ленина", "house": "5"},
    {"etalon_id": "ro_2", "city": "Самара", "street_type": "улица", "street": "Ленина", "house": "5"},
]


def test_ask_city_wins_over_confirm_for_same_name_two_cities() -> None:
    """Одноимённая улица (с одинаковым домом) в двух городах, город пуст -> ask_city, а не
    confirm/answer, даже при p лидера много выше theta_confirm (порядок правил §5.5)."""
    m = Matcher(_TWO_CITY_ETALON, ranker="manual")
    r = m.match("ленина 5", channel="voice", slots={"city": ""})
    assert r.decision == "ask_city"
    assert r.decision not in ("confirm", "answer", "answer_soft")
    assert r.missing_slot == "city"
    assert {opt["id"] for opt in r.options} <= {"москва", "самара"}
    assert r.explain["decision_rule"] == 2
# END_BLOCK_RULE_ORDER


# START_BLOCK_PROJECTION
def test_to_tz_projection_le_3_and_empty_for_reject_ask_street(matcher: Matcher, labeled: list[dict]) -> None:
    """Проекция §5.5: candidates/scores одной длины, <=3 при любом исходе; [] у ask_street/reject."""
    for row_id in ("a_0193", "a_0267", "a_0192", "a_0396", "a_0434", "a_0382", "a_0485"):
        r = _match_row(matcher, _row(labeled, row_id))
        assert len(r.candidates) == len(r.scores) <= 3
        if r.decision in ("ask_street", "reject"):
            assert r.candidates == []
# END_BLOCK_PROJECTION


# START_BLOCK_NEVER_THROWS
_ONE_ROW_ETALON = [{"etalon_id": "x1", "city": "Москва", "street_type": "улица", "street": "Ленина", "house": "1"}]


def test_never_throws_on_garbage_raw() -> None:
    m = Matcher(_ONE_ROW_ETALON, ranker="manual")
    for raw in (None, 123, ["не", "строка"], {"raw": "dict"}):
        r = m.match(raw, channel="voice")
        assert r.decision == "reject"  # F5: любой не-строковый/пустой raw -> честный отказ, не исключение


def test_never_throws_on_garbage_slots_asked_slot_scope() -> None:
    m = Matcher(_ONE_ROW_ETALON, ranker="manual")
    r = m.match("улица ленина 1", channel="voice", slots="junk")
    assert r.decision == "reject" and r.error is not None

    r = m.match("улица ленина 1", channel="voice", slots={"city": 12345, "street": {"id": object()}, "house": {"id": "x"}})
    assert r.decision == "reject" and r.error is not None

    r = m.match("улица ленина 1", channel="voice", asked_slot=object())
    assert isinstance(r.decision, str)  # не бросает; неизвестный asked_slot просто не совпадает ни с одним слотом

    r = m.match("улица ленина 1", channel="voice", scope=42)
    assert r.decision == "reject" and r.error is not None


def test_never_throws_on_broken_etalon_ready_false() -> None:
    m = Matcher(["не словарь", "тоже не словарь"], ranker="manual")
    assert m.ready() is False
    r = m.match("улица ленина 1", channel="voice")
    assert r.decision == "reject"
    assert r.error is not None
# END_BLOCK_NEVER_THROWS


# START_BLOCK_SLOTS
_TEXT_VS_HINT_ETALON = [
    {"etalon_id": "sl_m1", "city": "Москва", "street_type": "улица", "street": "Ленина", "house": "1"},
    {"etalon_id": "sl_s1", "city": "Самара", "street_type": "улица", "street": "Ленина", "house": "1"},
]


def test_slot_city_text_wins_over_city_hint_from_raw() -> None:
    """F4: слот-текст города приоритетнее parser.city_hint из raw ("самара" в начале строки)."""
    m = Matcher(_TEXT_VS_HINT_ETALON, ranker="manual")
    r = m.match("самара улица ленина 1", channel="voice", slots={"city": "Москва"})
    assert r.decision in ("answer", "answer_soft", "confirm")
    assert r.candidates[:1] == ["sl_m1"]


def test_slot_city_id_narrows_objects_to_that_city(matcher: Matcher) -> None:
    """F1/F4: слот-id города — точный фильтр (не просто сигнал ранжирования, K5): «Мира 27»
    существует и в Новосибирске (дом 27), но с city id-slot=«москва» кандидаты — только Москвы."""
    r = matcher.match("я живу на улица мира 27", channel="voice", slots={"city": {"id": "москва"}})
    assert r.decision == "ask_house"
    assert set(r.candidates) <= {"e_0262", "e_0629"}  # оба дома «Мира» в Москве, не e_0098 (НСК)
    assert r.missing_slot == "house"


_STREET_ID_ETALON = [{"etalon_id": "sid_60", "city": "Москва", "street_type": "улица", "street": "Мира", "house": "60"}]


def test_slot_street_id_plus_asked_house_word_number_gives_answer() -> None:
    """F1/F4 + T-007 п.1: слот-id улицы (сужение без поиска имени) + asked_slot="house" с
    raw «шестьдесят» (числительное словом, numerals.words_to_digits) -> дом 60 -> answer."""
    m = Matcher(_STREET_ID_ETALON, ranker="manual")
    first = m.match("мира", channel="voice", slots={"city": "Москва"})
    assert first.decision == "ask_house"
    street_id = first.explain.get("leader_street_id")
    city_id = first.explain.get("leader_city_id")
    assert street_id and city_id

    second = m.match(
        "шестьдесят",
        channel="voice",
        slots={"street": {"id": street_id}, "city": {"id": city_id}},
        asked_slot="house",
    )
    assert second.decision == "answer"
    assert second.candidates[:1] == ["sid_60"]
# END_BLOCK_SLOTS


# START_BLOCK_FORCED_HOUSE_CYCLE
def test_forced_house_cycle_real(matcher: Matcher, labeled: list[dict], etalon: list[dict]) -> None:
    """T-007b tools/simulate_dialog.strip_house на реальной строке a_0425 «ул. таганская 8п/2»:
    срезать дом (ParseResult.city_hint+street_type+street) -> первый match() без дома -> ask_house;
    второй с slots-id улицы/города, asked_slot="house", raw=истинный дом -> answer, верный лидер."""
    from tools.simulate_dialog import strip_house  # tools/ не пакет с __init__.py, но repo root на sys.path

    row = _row(labeled, "a_0425")
    true_etalon = next(e for e in etalon if e["etalon_id"] == row["etalon_id"])

    stripped, cut_house = strip_house(row["raw_adress"], row["channel"])
    assert stripped is not None and cut_house  # дом в исходном raw parser нашёл
    assert stripped == "улица таганская"

    first = matcher.match(stripped, channel=row["channel"], slots={"city": row["city"]})
    assert first.decision == "ask_house"

    street_id = first.explain.get("leader_street_id")
    city_id = first.explain.get("leader_city_id")
    assert street_id and city_id

    second = matcher.match(
        str(true_etalon["house"]),
        channel=row["channel"],
        slots={"street": {"id": street_id}, "city": {"id": city_id}},
        asked_slot="house",
    )
    assert second.decision in ("answer", "answer_soft")
    assert second.candidates[:1] == [row["etalon_id"]]
# END_BLOCK_FORCED_HOUSE_CYCLE
