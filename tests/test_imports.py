# START_MODULE_CONTRACT
#   PURPOSE: Тест графа импортов (N5): numerals/normalizer переиспользуемы отдельно от справочника,
#            тяжёлых зависимостей и остального пайплайна.
#   SCOPE: tests/test_imports.py — проверка sys.modules после импорта numerals/normalizer.
#   DEPENDS: M-NUMERALS, M-NORMALIZER
#   LINKS: V-M-NUMERALS, V-M-NORMALIZER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_numerals_and_normalizer_do_not_pull_in_index_parser_matcher_or_heavy_deps - N5
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-002: тест графа импортов numerals/normalizer (N5)
# END_CHANGE_SUMMARY

"""Тест графа импортов: numerals/normalizer без index/parser/matcher/sklearn/rapidfuzz (N5)."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_FORBIDDEN = (
    "addrmatch.index",
    "addrmatch.parser",
    "addrmatch.matcher",
    "sklearn",
    "rapidfuzz",
)

# Запускаем в отдельном процессе: если этот тестовый файл (или его сосед) уже успел где-то
# импортировать addrmatch.matcher раньше в том же интерпретаторе, проверка по sys.modules
# станет недостоверной (модуль останется в кэше). Отдельный процесс - чистый sys.modules.
_CHECK_SCRIPT = """
import sys
import addrmatch.numerals
import addrmatch.normalizer

forbidden = {forbidden!r}
loaded = sorted(m for m in forbidden if m in sys.modules)
print(",".join(loaded))
"""


def test_numerals_and_normalizer_do_not_pull_in_index_parser_matcher_or_heavy_deps():
    script = _CHECK_SCRIPT.format(forbidden=_FORBIDDEN)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = [m for m in proc.stdout.strip().split(",") if m]
    assert loaded == [], f"forbidden modules loaded: {loaded}"
