# START_MODULE_CONTRACT
#   PURPOSE: Публичная точка сопоставления адреса со справочником; MatchResult + Matcher.
#   SCOPE: Контракт F2/F5 (docs/concept.md §3, §5). A3 — рабочий конвейер: Normalizer -> Parser ->
#          Index -> Ranker -> Decider. Поддерживает --ablate whole_string|no_phonetic (H1a, H2).
#          T-006: candidate_features() выносит шаг "конвейер -> фичи кандидатов" наружу (без
#          ранжирования/решения) для обучения LogregRanker (tools/train_ranker.py), не дублируя код
#          match(); ranker="logreg"|"manual"|объект переключает Ranker (fallback manual при
#          отсутствии addrmatch/ranker_model.json).
#   DEPENDS: M-NORMALIZER, M-PARSER, M-INDEX, M-RANKER, M-DECIDER
#   LINKS: V-M-MATCHER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   MatchResult - структура ответа match() по контракту F2
#   Matcher - ready()/match()/candidate_features(): normalize->parse->candidates->resolve_objects->
#             ->features(->score->decide только в match())
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-001: каркас MatchResult/Matcher-заглушки (всегда reject, никогда не бросает)
#   C-ADDRMATCH-PHASE-A T-004: рабочий match() — сборка фичей Ranker из Index.candidates/
#   resolve_objects, n_close вторым проходом, Decider.decide(); ablate whole_string/no_phonetic.
#   C-ADDRMATCH-PHASE-A T-006: конвейер до фичей вынесен в _candidate_features_full() (публичная
#   обёртка candidate_features() — для tools/train_ranker.py, LogregRanker.fit()); добавлена фича
#   sim_x_house; ranker: str("logreg"|"manual")|объект|None в конструкторе — "logreg" грузит
#   addrmatch/ranker_model.json, при отсутствии/ошибке — fallback ManualRanker с предупреждением
#   в MatchResult.explain["ranker_fallback_warning"].
# END_CHANGE_SUMMARY

"""Матчер адресов: контракт F2 (docs/concept.md). A3 — рабочий конвейер §5."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from addrmatch.decider import Decider
from addrmatch.index import BruteForceIndex, norm_house
from addrmatch.normalizer import normalize
from addrmatch.parser import parse
from addrmatch.ranker import LogregRanker, ManualRanker, street_sim

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "ranker_model.json"

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
    #             ranker: str("logreg"|"manual")|объект(score/explain)|None - выбор ранкера (T-006);
    #             None/"manual" -> ManualRanker; "logreg" -> LogregRanker.load(addrmatch/ranker_model.json),
    #             fallback ManualRanker + предупреждение при отсутствии/битом файле; объект - подмена
    #             напрямую (тесты/эксперименты). N: float - стоимость ложного answer в переспросах (N4),
    #             ablate: str|None - "whole_string"|"no_phonetic"|None (H1a, H2) }
    #   OUTPUTS: none
    #   SIDE_EFFECTS: none (fallback-ветка "logreg" читает addrmatch/ranker_model.json с диска)
    # END_CONTRACT: __init__
    def __init__(
        self,
        etalon: list[dict],
        ranker: Any = None,
        N: float = 10,
        ablate: str | None = None,
    ) -> None:
        # START_BLOCK_INIT
        self._ablate = ablate
        self._ranker_fallback_warning: str | None = None
        try:
            self._index = BruteForceIndex(etalon)
            self._ranker = self._resolve_ranker(ranker)
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

    # START_BLOCK_RANKER_SELECT
    def _resolve_ranker(self, ranker: Any) -> Any:
        """None/"manual" -> ManualRanker; "logreg" -> LogregRanker.load с fallback на ManualRanker
        (файла нет/битый) + предупреждение в explain (T-006); объект (score/explain) - как есть."""
        if ranker is None or ranker == "manual":
            return ManualRanker()
        if ranker == "logreg":
            if not DEFAULT_MODEL_PATH.exists():
                self._ranker_fallback_warning = f"logreg: {DEFAULT_MODEL_PATH} не найден, fallback manual"
                return ManualRanker()
            try:
                return LogregRanker.load(str(DEFAULT_MODEL_PATH))
            except Exception as exc:  # noqa: BLE001 - битая модель не должна ронять Matcher
                self._ranker_fallback_warning = f"logreg: ошибка загрузки модели ({exc}), fallback manual"
                return ManualRanker()
        if isinstance(ranker, str):
            raise ValueError(f"Matcher: неизвестный ranker={ranker!r} (ожидается logreg|manual|объект)")
        return ranker
    # END_BLOCK_RANKER_SELECT

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

    # START_CONTRACT: candidate_features
    #   PURPOSE: Прогнать конвейер (Normalizer->Parser->Index) до фичей кандидатов-объектов, без
    #            ранжирования/решения (T-006: обучающая выборка LogregRanker.fit(), без дублирования
    #            кода match()).
    #   INPUTS: { raw: str, channel: str, slots: dict|None, scope: list[str]|None }
    #   OUTPUTS: { list[tuple[ObjCand, dict[str, float]]] - кандидаты-объекты + их фичи (FEATURES, §5.4) }
    #   SIDE_EFFECTS: none
    # END_CONTRACT: candidate_features
    def candidate_features(
        self, raw: str, channel: str = "voice", slots: dict | None = None, scope: Any = None
    ) -> list[tuple[Any, dict[str, float]]]:
        # START_BLOCK_CANDIDATE_FEATURES
        pairs, _parse_result, _city_res, _house_norm = self._candidate_features_full(raw, channel, slots or {}, scope)
        return pairs
        # END_BLOCK_CANDIDATE_FEATURES

    # START_BLOCK_PIPELINE
    def _candidate_features_full(
        self, raw: str, channel: str, slots: dict, scope: Any
    ) -> tuple[list[tuple[Any, dict[str, float]]], Any, Any, str | None]:
        """Конвейер до фичей кандидатов (общий шаг match() и candidate_features(), T-006): parse ->
        city -> street_q -> Index.candidates/resolve_objects -> фичи (+ n_close вторым проходом)."""
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

        pairs: list[tuple[Any, dict[str, float]]] = []
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
            feats["sim_x_house"] = street_sim(feats) * feats["house_in_list"]  # T-006
            pairs.append((oc, feats))

        # n_close: второй проход — число кандидатов в пределах 0.05 от лучшего сырого скора улицы.
        raw_scores = [max(f["lev"], f["phon"], f["ngram"]) for _, f in pairs]
        best_raw = max(raw_scores) if raw_scores else 0.0
        n_close = float(sum(1 for s in raw_scores if best_raw - s <= 0.05))
        for _, f in pairs:
            f["n_close"] = n_close
        # END_BLOCK_FEATURES

        return pairs, parse_result, city_res, (None if whole_string else house_norm)

    def _match_impl(self, raw: str, channel: str, slots: dict, asked_slot: str | None, scope: Any) -> MatchResult:
        pairs, parse_result, city_res, house_norm = self._candidate_features_full(raw, channel, slots, scope)

        ranked = [{"obj": oc, "feats": feats, "p": self._ranker.score(feats)} for oc, feats in pairs]
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
            house_norm=house_norm,
        )

        explain = self._ranker.explain(ranked[0]["feats"]) if ranked else {"marker": "[Matcher][match][NO_CANDIDATES]"}
        if self._ranker_fallback_warning:
            explain = {**explain, "ranker_fallback_warning": self._ranker_fallback_warning}
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
