# START_MODULE_CONTRACT
#   PURPOSE: Публичный API пакета addrmatch.
#   SCOPE: Реэкспорт Matcher/MatchResult для внешних потребителей (run.py, тесты).
#   DEPENDS: M-MATCHER
#   LINKS: V-M-MATCHER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   Matcher - см. addrmatch.matcher (лениво, см. __getattr__)
#   MatchResult - см. addrmatch.matcher (лениво, см. __getattr__)
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-001: экспорт Matcher/MatchResult
#   C-ADDRMATCH-PHASE-A T-002: ленивый импорт (PEP 562) - `import addrmatch.numerals`/
#   `addrmatch.normalizer` не должен тянуть за собой addrmatch.matcher (N5, tests/test_imports.py).
#   Публичный доступ `from addrmatch import Matcher` не меняется.
# END_CHANGE_SUMMARY

"""addrmatch — локальный матчер адресов (docs/concept.md)."""

__all__ = ["Matcher", "MatchResult"]


def __getattr__(name: str):
    # START_BLOCK_LAZY_MATCHER
    if name in __all__:
        from addrmatch.matcher import Matcher, MatchResult

        globals().update(Matcher=Matcher, MatchResult=MatchResult)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # END_BLOCK_LAZY_MATCHER
