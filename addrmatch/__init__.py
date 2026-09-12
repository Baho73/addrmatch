# START_MODULE_CONTRACT
#   PURPOSE: Публичный API пакета addrmatch.
#   SCOPE: Реэкспорт Matcher/MatchResult для внешних потребителей (run.py, тесты).
#   DEPENDS: M-MATCHER
#   LINKS: V-M-MATCHER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Matcher - см. addrmatch.matcher
#   MatchResult - см. addrmatch.matcher
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-001: экспорт Matcher/MatchResult
# END_CHANGE_SUMMARY

"""addrmatch — локальный матчер адресов (docs/concept.md)."""

from addrmatch.matcher import Matcher, MatchResult

__all__ = ["Matcher", "MatchResult"]
