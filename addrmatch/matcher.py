# START_MODULE_CONTRACT
#   PURPOSE: Публичная точка сопоставления адреса со справочником; MatchResult + Matcher.
#   SCOPE: Контракт F2/F5 (docs/concept.md §3, §5). A0 — каркас-заглушка: всегда reject.
#   DEPENDS: none
#   LINKS: V-M-MATCHER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   MatchResult - структура ответа match() по контракту F2
#   Matcher - ready()/match(): загрузка эталона, сопоставление строки
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-001: каркас MatchResult/Matcher-заглушки (всегда reject, никогда не бросает)
# END_CHANGE_SUMMARY

"""Матчер адресов: контракт F2 (docs/concept.md). A0 — тривиальная заглушка."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# семь исходов диалога (docs/concept.md F2)
DECISIONS = ("answer", "answer_soft", "confirm", "ask_city", "ask_house", "ask_street", "reject")


@dataclass
class MatchResult:
    """Результат match() — см. docs/concept.md §3 F2."""

    decision: str
    candidates: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    missing_slot: str | None = None
    options: list[Any] = field(default_factory=list)
    slots_resolved: dict[str, Any] = field(default_factory=dict)
    explain: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Matcher:
    # START_CONTRACT: __init__
    #   PURPOSE: Загрузить эталонный справочник и подготовить матчер к работе.
    #   INPUTS: { etalon: list[dict] - эталонные адреса (etalon_id/city/street_type/street/house/...), ablate: str|None - имя абляции (F5; заглушкой игнорируется) }
    #   OUTPUTS: none
    #   SIDE_EFFECTS: none
    # END_CONTRACT: __init__
    def __init__(self, etalon: list[dict], ablate: str | None = None) -> None:
        # START_BLOCK_INIT
        self._etalon = etalon
        self._ablate = ablate
        self._ready = True
        # END_BLOCK_INIT

    def ready(self) -> bool:
        """Готовность матчера; False только при ошибке загрузки (F5)."""
        return self._ready

    # START_CONTRACT: match
    #   PURPOSE: Сопоставить сырую строку адреса с эталоном; вернуть решение для диалога (F2).
    #   INPUTS: { raw: str - сырая строка адреса, channel: str - voice|webchat, slots: dict|None - слоты-id/текст, asked_slot: str|None - какой слот спрашивал бот, scope: Any - область поиска (city_id/регион/None) }
    #   OUTPUTS: { MatchResult - decision/candidates/scores/missing_slot/options/slots_resolved/explain/error }
    #   SIDE_EFFECTS: none; никогда не бросает исключений (F5) — ошибка уходит в error
    # END_CONTRACT: match
    def match(
        self,
        raw: str,
        channel: str = "voice",
        slots: dict | None = None,
        asked_slot: str | None = None,
        scope: Any = None,
    ) -> MatchResult:
        # START_BLOCK_GUARD
        try:
            # START_BLOCK_STUB
            # A0: тривиальный матчер — всегда отказ; заменяется реальным конвейером в A1-A7.
            return MatchResult(
                decision="reject",
                candidates=[],
                scores=[],
                missing_slot=None,
                options=[],
                slots_resolved={},
                explain={"marker": "[Matcher][match][STUB]"},
                error=None,
            )
            # END_BLOCK_STUB
        except Exception as exc:  # noqa: BLE001 - контракт F5: match() никогда не бросает наружу
            return MatchResult(
                decision="reject",
                candidates=[],
                scores=[],
                missing_slot=None,
                options=[],
                slots_resolved={},
                explain={"marker": "[Matcher][match][STUB]"},
                error=str(exc),
            )
        # END_BLOCK_GUARD
