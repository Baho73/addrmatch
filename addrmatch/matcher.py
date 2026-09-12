# START_MODULE_CONTRACT
#   PURPOSE: Публичная точка сопоставления адреса со справочником; MatchResult + Matcher.
#   SCOPE: Контракт F2/F5 (docs/concept.md §3, §5). A3 — рабочий конвейер: Normalizer -> Parser ->
#          Index -> Ranker -> Decider. Поддерживает --ablate whole_string|no_phonetic (H1a, H2).
#          T-006: candidate_features() выносит шаг "конвейер -> фичи кандидатов" наружу (без
#          ранжирования/решения) для обучения LogregRanker (tools/train_ranker.py), не дублируя код
#          match(); ranker="logreg"|"manual"|объект переключает Ranker (fallback manual при
#          отсутствии addrmatch/ranker_model.json).
#          T-007: match(slots, asked_slot, scope) — API диалога (F1/F4). Слоты `city`/`street`
#          принимают либо текст (str, фаззи, приоритет над raw) либо id (dict {"id": <city_id|
#          street_id>}, точный фильтр без фаззи); `house` — всегда текст/номер (str). `street_id`
#          — строка формата "<city_id>||<street_type>||<name_id>" (контракт index.py, тот же, что
#          отдаёт slots_resolved["street_id"] при answer*/confirm) — распознаётся Matcher'ом без
#          изменений в index.py. При `asked_slot="city"` и отсутствии явного слота города raw
#          целиком идёт в resolve_city() (открытый ответ на дозапрос).
#   DEPENDS: M-NORMALIZER, M-PARSER, M-INDEX, M-RANKER, M-DECIDER
#   LINKS: V-M-MATCHER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   MatchResult - структура ответа match() по контракту F2
#   Matcher - ready()/match()/candidate_features(): normalize->parse->candidates->resolve_objects->
#             ->features(->score->decide только в match())
#   Matcher._slot_parts - слот (city/street) -> (текст, id) по контракту T-007
#   Matcher._objects_for_street_id - слот-id улицы -> ObjCand точного объекта (все его дома)
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
#   C-ADDRMATCH-PHASE-A T-007: слоты city/street — текст (str, фаззи) или id (dict {"id": ...},
#   точный фильтр); слот-id улицы пропускает Index.candidates() целиком (F1/F4). asked_slot="city"
#   без явного слота — raw целиком в resolve_city(). city_display (нормализованный city_id ->
#   исходное написание из etalon, собран в __init__ из сырых строк) даёт options ask_city в формате
#   [{"id","name"}]. explain дополнен decision_rule/marker "[Decider][decide][RULE_n]" (Decider.rule)
#   и leader_street_id/leader_city_id (нужны боту для второго вызова после ask_house — slots_resolved
#   для ask_house по §5.5 пуст). tools/simulate_dialog.py (H12) использует оба API.
# END_CHANGE_SUMMARY

"""Матчер адресов: контракт F2 (docs/concept.md). A3 — рабочий конвейер §5."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from addrmatch.decider import Decider
from addrmatch.index import BruteForceIndex, CityResolution, NameCand, norm_house
from addrmatch.normalizer import normalize
from addrmatch.parser import parse
from addrmatch.ranker import LogregRanker, ManualRanker, street_sim

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "ranker_model.json"

# семь исходов диалога (docs/concept.md F2)
DECISIONS = ("answer", "answer_soft", "confirm", "ask_city", "ask_house", "ask_street", "reject")


# START_CONTRACT: _slot_parts
#   PURPOSE: Разобрать значение слота city/street на (текст, id) по контракту T-007 (F1/F4).
#   INPUTS: { value: str|dict|None - текст (фаззи) | {"id": ...} (точный фильтр) | не задан }
#   OUTPUTS: { tuple[str|None, Any|None] - (текст или None, id или None); ровно один из двух }
#   SIDE_EFFECTS: none
# END_CONTRACT: _slot_parts
def _slot_parts(value: Any) -> tuple[str | None, Any | None]:
    if isinstance(value, dict):
        return None, value.get("id")
    if isinstance(value, str) and value.strip():
        return value, None
    return None, None


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
            # T-007: city_id (нормализованный, index.py) -> исходное написание из etalon, для
            # options ask_city [{"id","name"}] (§3 F2) - Index не хранит исходный регистр города.
            self._city_display: dict[str, str] = {}
            for row in etalon:
                city_id = normalize(row.get("city", ""), channel="webchat").text
                if city_id and city_id not in self._city_display:
                    self._city_display[city_id] = row.get("city", city_id)
        except Exception as exc:  # noqa: BLE001 - ошибка загрузки уходит в ready()==False (F5)
            self._index = None
            self._ranker = None
            self._decider = None
            self._ready = False
            self._load_error = str(exc)
            self._city_display = {}
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
    #   INPUTS: { raw: str - сырая строка адреса (любой другой тип - reject+error, не исключение),
    #             channel: str - voice|webchat, slots: dict|None - {"city"|"street": текст(str, фаззи,
    #             приоритет над raw) | {"id": city_id|street_id}(точный фильтр, без фаззи),
    #             "house": текст/номер(str) - слот дома всегда текст}, asked_slot: str|None -
    #             None|"city"|"house"|"street" - какой слот спрашивал бот (при не-None правило 1
    #             Decider - has_address_cues - пропускается; "city" без явного слота city - raw
    #             целиком идёт в resolve_city()), scope: list[str]|None - список city_id, сужает
    #             resolve_objects/options (F1) }
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
        pairs, _parse_result, _city_res, _house_norm = self._candidate_features_full(
            raw, channel, slots or {}, scope, asked_slot=None
        )
        return pairs
        # END_BLOCK_CANDIDATE_FEATURES

    # START_BLOCK_PIPELINE
    def _candidate_features_full(
        self, raw: str, channel: str, slots: dict, scope: Any, asked_slot: str | None = None
    ) -> tuple[list[tuple[Any, dict[str, float]]], Any, Any, str | None]:
        """Конвейер до фичей кандидатов (общий шаг match() и candidate_features(), T-006): parse ->
        city -> street_q -> Index.candidates/resolve_objects -> фичи (+ n_close вторым проходом)."""
        norm = normalize(raw or "", channel=channel)
        parse_result = parse(norm, self._index.street_types, has_name=self._index.has_name)

        # START_BLOCK_CITY
        # T-007: слот-id города - точный, без фаззи (F1/F4). Слот-текст - фаззи (resolve_city),
        # приоритет над raw. asked_slot="city" без явного слота - raw целиком трактуется как ответ
        # на дозапрос про город (parser кладёт голое имя города в street/city_hint только при
        # наличии типа улицы дальше в строке - для "москва" самой по себе нужен явный обход).
        city_text, city_id_slot = _slot_parts(slots.get("city"))
        if city_id_slot is not None:
            city_res = CityResolution(status="resolved", city_id=str(city_id_slot), score=1.0)
        else:
            city_query = city_text
            if city_query is None and asked_slot == "city":
                city_query = norm.text or None
            city_res = self._index.resolve_city(city_query, scope=scope)
            if city_res.status in ("none", "unresolved") and parse_result.city_hint:
                alt = self._index.resolve_city(parse_result.city_hint, scope=scope)
                if alt.status == "resolved":
                    city_res = alt

        # Слот-id города - "точный фильтр" (F1), не просто сигнал ранжирования: в отличие от
        # текста/фаззи-резолва (K5, city не блокирует поиск по имени - Мира-27 в Новосибирске
        # иначе перебивал бы Мира-в-Москве по силе совпадения дома), подтверждённый ботом id из
        # options уже не подлежит сомнению - сужаем resolve_objects() до этого города пересечением
        # со scope (candidates() по контракту §5.3/F1 город не сужает - только resolve_objects/options).
        objects_scope = scope
        if city_id_slot is not None:
            objects_scope = [city_res.city_id] if not scope else [c for c in scope if c == city_res.city_id]
        # END_BLOCK_CITY

        # START_BLOCK_STREET_QUERY
        # T-007: слот-id улицы - точный фильтр, поиск имени (Index.candidates) пропускается
        # целиком (F1/F4); слот-текст - фаззи, приоритет над raw, как и раньше.
        street_text, street_id_slot = _slot_parts(slots.get("street"))
        whole_string = self._ablate == "whole_string"
        if street_id_slot is not None and not whole_string:
            obj_cands = self._objects_for_street_id(str(street_id_slot), objects_scope)
            name_cand_by_id = {
                oc.name_id: NameCand(name_id=oc.name_id, name=oc.name, lev=1.0, token_set=1.0, phon=1.0, ngram=1.0, alias_hit=1.0)
                for oc in obj_cands
            }
            if obj_cands and city_res.status != "resolved":
                # Слот-id улицы уже однозначно определяет город - Decider (ask_house/ask_street,
                # §5.5) читает city_res.status, а не street_id, дальше по конвейеру.
                city_res = CityResolution(status="resolved", city_id=obj_cands[0].city_id, score=1.0)
        else:
            if whole_string:
                street_q = norm.text
            else:
                street_q = street_text or parse_result.street
                if not street_q and not parse_result.house:
                    street_q = norm.text
            name_cands = self._index.candidates(street_q or "", k=20, scope=scope)
            if self._ablate == "no_phonetic":
                for nc in name_cands:
                    nc.phon = 0.0
            name_cand_by_id = {nc.name_id: nc for nc in name_cands}
            obj_cands = self._index.resolve_objects(list(name_cand_by_id), city_res, scope=objects_scope)
        # END_BLOCK_STREET_QUERY

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
    # END_BLOCK_PIPELINE

    # START_CONTRACT: _objects_for_street_id
    #   PURPOSE: Слот-id улицы (T-007, F1/F4) -> точный объект (все его дома), без поиска имени.
    #   INPUTS: { street_id: str - "<city_id>||<street_type>||<name_id>" (формат index.py, тот же,
    #             что отдаёт slots_resolved["street_id"]/explain["leader_street_id"]), scope: список
    #             city_id|None }
    #   OUTPUTS: { list[ObjCand] - 0 или 1 объект (пустой список при неразборчивом/чужом id) }
    #   SIDE_EFFECTS: none
    # END_CONTRACT: _objects_for_street_id
    def _objects_for_street_id(self, street_id: str, scope: Any) -> list[Any]:
        # START_BLOCK_STREET_BY_ID
        parts = street_id.split("||")
        if len(parts) != 3:
            return []
        city_part, _type_part, name_id_str = parts
        try:
            name_id = int(name_id_str)
        except ValueError:
            return []
        exact_city = CityResolution(status="resolved", city_id=city_part, score=1.0)
        found = self._index.resolve_objects([name_id], exact_city, scope=scope)
        return [oc for oc in found if oc.street_id == street_id]
        # END_BLOCK_STREET_BY_ID

    def _match_impl(self, raw: str, channel: str, slots: dict, asked_slot: str | None, scope: Any) -> MatchResult:
        # START_BLOCK_MATCH_IMPL
        pairs, parse_result, city_res, house_norm = self._candidate_features_full(
            raw, channel, slots, scope, asked_slot=asked_slot
        )

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
                "street_sim": street_sim(r["feats"]),  # T-007: гейт ask_city по уверенности имени
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

        # START_BLOCK_EXPLAIN
        # T-007: explain = разложение скора Ranker (переименовано в ranker_marker) + marker/
        # decision_rule Decider (обязателен по контракту) + leader_street_id/city_id, нужные
        # боту для второго вызова после ask_house, где slots_resolved по §5.5 остаётся пустым.
        explain: dict[str, Any] = dict(self._ranker.explain(ranked[0]["feats"])) if ranked else {}
        explain["ranker_marker"] = explain.pop("marker", "[Matcher][match][NO_CANDIDATES]")
        if ranked:
            explain["leader_street_id"] = ranked[0]["obj"].street_id
            explain["leader_city_id"] = ranked[0]["obj"].city_id
        if self._ranker_fallback_warning:
            explain["ranker_fallback_warning"] = self._ranker_fallback_warning
        explain["decision_rule"] = result.rule
        explain["marker"] = f"[Decider][decide][RULE_{result.rule}]"
        # END_BLOCK_EXPLAIN

        # START_BLOCK_OPTIONS
        # T-007: ask_city -> options в формате [{"id","name"}] (§3 F2); Decider отдаёт голые
        # city_id (чистая функция без DEPENDS на данные) - имя достаёт Matcher из city_display.
        options = result.options
        if result.decision == "ask_city":
            options = [{"id": cid, "name": self._city_display.get(cid, cid)} for cid in result.options]
        # END_BLOCK_OPTIONS

        return MatchResult(
            decision=result.decision,
            candidates=result.candidates,
            scores=result.scores,
            missing_slot=result.missing_slot,
            options=options,
            slots_resolved=result.slots_resolved,
            explain=explain,
            error=None,
        )
        # END_BLOCK_MATCH_IMPL
    # marker: [Matcher][match][DONE]
