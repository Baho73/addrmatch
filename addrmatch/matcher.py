# START_MODULE_CONTRACT
#   PURPOSE: Публичная точка сопоставления адреса со справочником; MatchResult + Matcher.
#   SCOPE: Контракт F2/F5 (docs/concept.md §3, §5). A3 — рабочий конвейер: Normalizer -> Parser ->
#          Index -> Ranker -> Decider. Поддерживает --ablate whole_string|no_phonetic (H1a, H2).
#   DEPENDS: M-NORMALIZER, M-PARSER, M-INDEX, M-RANKER, M-DECIDER
#   LINKS: V-M-MATCHER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   MatchResult - структура ответа match() по контракту F2
#   Matcher - ready()/match(): normalize->parse->candidates->resolve_objects->score->decide
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-001: каркас MatchResult/Matcher-заглушки (всегда reject, никогда не бросает)
#   C-ADDRMATCH-PHASE-A T-004: рабочий match() — сборка фичей Ranker из Index.candidates/
#   resolve_objects, n_close вторым проходом, Decider.decide(); ablate whole_string/no_phonetic.
# END_CHANGE_SUMMARY

"""Матчер адресов: контракт F2 (docs/concept.md). A3 — рабочий конвейер §5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from addrmatch.decider import Decider
from addrmatch.index import BruteForceIndex, norm_house
from addrmatch.normalizer import normalize
from addrmatch.parser import parse
from addrmatch.ranker import ManualRanker

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


def _type_match(parsed_type: str | None, obj_type: str) -> float:
    if parsed_type is None:
        return 0.5
    return 1.0 if parsed_type == obj_type else 0.0


class Matcher:
    # START_CONTRACT: __init__
    #   PURPOSE: Загрузить эталонный справочник и подготовить матчер к работе.
    #   INPUTS: { etalon: list[dict] - эталонные адреса (etalon_id/city/street_type/street/house/...),
    #             ranker: ManualRanker|None - подмена ранкера (T-006 LogregRanker), N: float -
    #             стоимость ложного answer в переспросах (N4), ablate: str|None - "whole_string"|
    #             "no_phonetic"|None (H1a, H2) }
    #   OUTPUTS: none
    #   SIDE_EFFECTS: none
    # END_CONTRACT: __init__
    def __init__(
        self,
        etalon: list[dict],
        ranker: ManualRanker | None = None,
        N: float = 10,
        ablate: str | None = None,
    ) -> None:
        # START_BLOCK_INIT
        self._ablate = ablate
        try:
            self._index = BruteForceIndex(etalon)
            self._ranker = ranker or ManualRanker()
            self._decider = Decider(N=N)
            self._ready = self._index.ready()
            self._load_error: str | None = None
        except Exception as exc:  # noqa: BLE001 - ошибка загрузки уходит в ready()==False (F5)
            self._index = None
            self._ranker = None
            self._decider = None
            self._ready = False
            self._load_error = str(exc)
        # END_BLOCK_INIT

    def ready(self) -> bool:
        """Готовность матчера; False только при ошибке загрузки (F5)."""
        return self._ready

    # START_CONTRACT: match
    #   PURPOSE: Сопоставить сырую строку адреса с эталоном; вернуть решение для диалога (F2).
    #   INPUTS: { raw: str - сырая строка адреса, channel: str - voice|webchat, slots: dict|None -
    #             {"city": текст, "street": текст, "house": текст} - слоты-текст с приоритетом над
    #             raw, asked_slot: str|None - какой слот спрашивал бот, scope: list[str]|None -
    #             область поиска (city_id) }
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
            if not self._ready:
                return MatchResult(decision="reject", explain={"marker": "[Matcher][match][NOT_READY]"}, error=self._load_error)
            return self._match_impl(raw, channel, slots or {}, asked_slot, scope)
        except Exception as exc:  # noqa: BLE001 - контракт F5: match() никогда не бросает наружу
            return MatchResult(decision="reject", explain={"marker": "[Matcher][match][ERROR]"}, error=str(exc))
        # END_BLOCK_GUARD

    # START_BLOCK_PIPELINE
    def _match_impl(self, raw: str, channel: str, slots: dict, asked_slot: str | None, scope: Any) -> MatchResult:
        norm = normalize(raw or "", channel=channel)
        parse_result = parse(norm, self._index.street_types, has_name=self._index.has_name)

        # START_BLOCK_CITY
        city_text = slots.get("city") or None
        city_res = self._index.resolve_city(city_text, scope=scope)
        if city_res.status in ("none", "unresolved") and parse_result.city_hint:
            alt = self._index.resolve_city(parse_result.city_hint, scope=scope)
            if alt.status == "resolved":
                city_res = alt
        # END_BLOCK_CITY

        # START_BLOCK_STREET_QUERY
        whole_string = self._ablate == "whole_string"
        if whole_string:
            street_q = norm.text
        else:
            street_q = slots.get("street") or parse_result.street
            if not street_q and not parse_result.house:
                street_q = norm.text
        # END_BLOCK_STREET_QUERY

        name_cands = self._index.candidates(street_q or "", k=20, scope=scope)
        if self._ablate == "no_phonetic":
            for nc in name_cands:
                nc.phon = 0.0

        name_cand_by_id = {nc.name_id: nc for nc in name_cands}
        obj_cands = self._index.resolve_objects(list(name_cand_by_id), city_res, scope=scope)

        # START_BLOCK_FEATURES
        house_norm = norm_house(parse_result.house) if parse_result.house else None
        house_slot_text = slots.get("house")
        if house_slot_text:
            house_norm = norm_house(house_slot_text)

        ranked: list[dict] = []
        for oc in obj_cands:
            nc = name_cand_by_id[oc.name_id]
            feats = {
                "lev": nc.lev,
                "token_set": nc.token_set,
                "phon": nc.phon,
                "ngram": nc.ngram,
                "alias_hit": nc.alias_hit,
                "type_match": _type_match(parse_result.street_type, oc.street_type),
                "city_match": oc.city_match,
                "house_found": 1.0 if parse_result.house else 0.0,
                "house_in_list": 0.0 if whole_string else (1.0 if house_norm and house_norm in oc.houses else 0.0),
                "channel_voice": 1.0 if channel == "voice" else 0.0,
                "name_freq": self._index.name_freq(oc.name_id),
            }
            feats["freq_x_city"] = feats["name_freq"] * feats["city_match"]
            feats["freq_x_phon"] = feats["name_freq"] * feats["phon"]
            feats["voice_x_lev"] = feats["channel_voice"] * feats["lev"]
            ranked.append({"obj": oc, "feats": feats})

        # n_close: второй проход — число кандидатов в пределах 0.05 от лучшего сырого скора улицы.
        raw_scores = [max(r["feats"]["lev"], r["feats"]["phon"], r["feats"]["ngram"]) for r in ranked]
        best_raw = max(raw_scores) if raw_scores else 0.0
        n_close = float(sum(1 for s in raw_scores if best_raw - s <= 0.05))
        for r in ranked:
            r["feats"]["n_close"] = n_close
            r["p"] = self._ranker.score(r["feats"])
        # END_BLOCK_FEATURES

        ranked.sort(key=lambda r: -r["p"])
        decider_input = [
            {
                "street_id": r["obj"].street_id,
                "name_id": r["obj"].name_id,
                "name": r["obj"].name,
                "city_id": r["obj"].city_id,
                "street_type": r["obj"].street_type,
                "houses": r["obj"].houses,
                "p": r["p"],
                "name_freq": r["feats"]["name_freq"],
            }
            for r in ranked
        ]

        result = self._decider.decide(
            ranked=decider_input,
            city_res=city_res,
            parse=parse_result,
            asked_slot=asked_slot,
            house_norm=None if whole_string else house_norm,
        )

        explain = self._ranker.explain(ranked[0]["feats"]) if ranked else {"marker": "[Matcher][match][NO_CANDIDATES]"}
        return MatchResult(
            decision=result.decision,
            candidates=result.candidates,
            scores=result.scores,
            missing_slot=result.missing_slot,
            options=result.options,
            slots_resolved=result.slots_resolved,
            explain=explain,
            error=None,
        )
    # END_BLOCK_PIPELINE
    # marker: [Matcher][match][DONE]
