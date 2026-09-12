# START_MODULE_CONTRACT
#   PURPOSE: Дымовой тест каркаса A0: Matcher строится и не бросает на пустой строке.
#   SCOPE: tests/test_smoke.py — минимальная проверка контракта Matcher до появления реального конвейера.
#   DEPENDS: M-MATCHER
#   LINKS: V-M-MATCHER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_matcher_stub_rejects_on_empty_string - Matcher на 3 эталонах отвечает reject и не бросает на ""
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-001: дымовой тест каркаса Matcher
# END_CHANGE_SUMMARY

"""Дымовой тест каркаса A0."""

from addrmatch import Matcher


def test_matcher_stub_rejects_on_empty_string():
    etalon = [
        {"etalon_id": "e_0000", "city": "Москва", "street_type": "улица", "street": "Ленина", "house": "1"},
        {"etalon_id": "e_0001", "city": "Москва", "street_type": "улица", "street": "Мира", "house": "2"},
        {"etalon_id": "e_0002", "city": "Казань", "street_type": "проспект", "street": "Победы", "house": "3"},
    ]
    matcher = Matcher(etalon)

    assert matcher.ready() is True

    result = matcher.match("", channel="voice")

    assert result.decision == "reject"
    assert result.candidates == []
    assert result.scores == []
    assert result.error is None
