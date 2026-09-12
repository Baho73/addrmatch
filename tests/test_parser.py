# START_MODULE_CONTRACT
#   PURPOSE: Тесты parser.parse() — формы дома, тип/улица/город, has_address_cues,
#            ambiguous_number (docs/concept.md §5.2, T-003).
#   SCOPE: Юнит-тесты, без чтения данных с диска.
#   DEPENDS: M-PARSER
#   LINKS: V-M-PARSER
# END_MODULE_CONTRACT
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-003: test_parser.py — формы дома (16/16а/16к2/16/2/16-18/16 кв 5),
#   тип+улица+дом, без типа, city_hint, слипание типа, дозапрос числом/словом, cues, ambiguous.
# END_CHANGE_SUMMARY

"""tests/test_parser.py — docs/concept.md §5.2."""

from __future__ import annotations

from addrmatch.normalizer import NormResult
from addrmatch.parser import DEFAULT_STREET_TYPES, parse


def _p(text: str, has_name=None):
    return parse(text, DEFAULT_STREET_TYPES, has_name=has_name)


# START_BLOCK_HOUSE_FORMS
def test_house_plain():
    assert _p("улица ленина 16").house == "16"


def test_house_letter():
    assert _p("улица ленина 16а").house == "16а"


def test_house_korpus_num():
    assert _p("улица ленина 16к2").house == "16к2"


def test_house_slash_is_korpus():
    assert _p("улица ленина 16/2").house == "16к2"


def test_house_range_flag():
    r = _p("улица ленина 16-18")
    assert r.house == "16"
    assert r.house_range is True


def test_house_apartment_dropped():
    r = _p("улица ленина 16 кв 5")
    assert r.house == "16"
    assert r.street == "ленина"
    assert r.tail_tokens == []
# END_BLOCK_HOUSE_FORMS


# START_BLOCK_SLOTS
def test_type_street_house():
    r = _p("проспект ленинградский 49")
    assert r.street_type == "проспект"
    assert r.street == "ленинградский"
    assert r.house == "49"


def test_no_type():
    r = _p("шахтеров 111")
    assert r.street_type is None
    assert r.street == "шахтеров"
    assert r.house == "111"


def test_city_hint_before_type():
    r = _p("красноярск улица космонавтов 17а")
    assert r.city_hint == "красноярск"
    assert r.street == "космонавтов"
    assert r.house == "17а"


def test_no_type_no_city_hint():
    # без типа улицы город не угадывается (docs/concept.md §5.2).
    r = _p("шахтеров 111")
    assert r.city_hint is None


def test_glued_street_type():
    r = _p("шоссемосковское 5")
    assert r.street_type == "шоссе"
    assert r.street == "московское"
    assert r.house == "5"
# END_BLOCK_SLOTS


# START_BLOCK_DIALOG_ANSWERS
def test_single_number_answer():
    r = _p("60")
    assert r.house == "60"
    assert r.street is None


def test_single_word_answer():
    r = _p("красноярск")
    assert r.street == "красноярск"
    assert r.house is None
    assert r.city_hint is None
# END_BLOCK_DIALOG_ANSWERS


# START_BLOCK_CUES
def test_cues_true_by_type():
    assert _p("улица ленина").has_address_cues is True


def test_cues_true_by_house_only():
    assert _p("60").has_address_cues is True


def test_cues_false_without_hints():
    assert _p("хочу заказать пиццу").has_address_cues is False


def test_cues_true_via_has_name():
    r = _p("шахтеров", has_name=lambda tok: tok == "шахтеров")
    assert r.has_address_cues is True


def test_cues_false_via_has_name_no_match():
    r = _p("случайный текст", has_name=lambda tok: False)
    assert r.has_address_cues is False
# END_BLOCK_CUES


# START_BLOCK_AMBIGUOUS
def test_two_bare_numbers_not_merged():
    r = _p("проспект ленинградский 3 1")
    assert r.house == "3"
    assert r.tail_tokens == ["1"]
    assert r.ambiguous_number is True


def test_ambiguous_number_propagated_from_normresult():
    nr = NormResult(text="улица ленина 5", ambiguous_number=True)
    r = parse(nr, DEFAULT_STREET_TYPES)
    assert r.ambiguous_number is True
# END_BLOCK_AMBIGUOUS


def test_tokens_field():
    assert _p("улица ленина 16").tokens == ["улица", "ленина", "16"]


def test_empty_string():
    r = _p("")
    assert r.house is None
    assert r.street is None
    assert r.street_type is None
    assert r.has_address_cues is False
