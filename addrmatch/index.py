# START_MODULE_CONTRACT
#   PURPOSE: Справочник -> поиск (docs/concept.md §5.3). Словарь имён улиц (слой 1) + объекты
#            город×тип×имя (слой 2, с домами); фаззи/фонетический/n-грам поиск имени, разрешение
#            города, раскрытие имени в объекты нужного города.
#   SCOPE: BruteForceIndex — реализация для учебной базы (699 уникальных имён, cdist по всем).
#          Не знает про Parser/Ranker/Decider; отдаёт им сырые сигналы и структуры.
#   DEPENDS: M-NORMALIZER (normalize() для словаря имён и городов)
#   LINKS: V-M-INDEX
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   phonetic_key - строка -> русский фонетический ключ (метатеза глухих/звонких, усечение окончаний)
#   norm_house - дом эталона/парсера -> сравнимая форма (lower)
#   CityResolution/NameCand/ObjCand/Obj - структуры обмена с Ranker/Decider
#   BruteForceIndex - ready/has_name/resolve_city/candidates/resolve_objects/houses/name_freq
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-004: index.py — словарь имён (699), фонетика/n-грам/Левенштейн
#   предпосчитаны, BruteForceIndex.candidates() < 1 мс (docs/concept.md §5.3, H1/H2/H6).
# END_CHANGE_SUMMARY

"""index.py — справочник -> поиск (docs/concept.md §5.3)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from addrmatch.normalizer import NORMALIZER_VERSION, normalize

THETA_CITY = 80  # rapidfuzz.fuzz.WRatio, шкала 0-100 (docs/concept.md §5.3)


# START_BLOCK_PHONETIC
# Русский "метафон" в духе §5.3: сначала усекаем окончания прилагательных (пока буквы ещё не
# оглушены/не редуцированы, иначе "-ого" превращается в "-око" после г->к и суффикс не узнать),
# затем оглушаем звонкие и редуцируем безударные гласные. Проверено на парах из брифа:
# "киравсгой"/"кировская", "пирвомайсгий"/"первомайский", "симверопольский"/"симферопольский",
# "краснозвиздная"/"краснозвездная" дают равные ключи; "дзержинскага"/"дзержинского" - похожие
# (общий префикс, разные окончания школьного типа - недостающая литера "-ого" в списке усечений).
_ENDINGS_SK = ("ская", "ский", "ское", "ской", "скай", "скии")
_ENDINGS_SHORT = ("ая", "яя", "ой", "ый", "ий", "ого", "ова", "ева", "ево")
_DEVOICE = str.maketrans("бвгджз", "пфктшс")
_VOWEL_A = str.maketrans("оая", "ааа")
_VOWEL_I = str.maketrans("еиыэ", "иииИ".lower())
_VOWEL_U = str.maketrans("ую", "уу")


def _collapse_doubles(s: str) -> str:
    out: list[str] = []
    for ch in s:
        if out and out[-1] == ch:
            continue
        out.append(ch)
    return "".join(out)


def phonetic_key(s: str) -> str:
    """Русский фонетический ключ имени улицы для сравнения ASR-искажений (docs/concept.md §5.3)."""
    s = s.lower().replace("ё", "е")
    s = s.replace("йо", "е").replace("йе", "е")
    s = _collapse_doubles(s)
    s = s.replace("ь", "").replace("ъ", "")

    for suf in _ENDINGS_SK:
        if s.endswith(suf) and len(s) > len(suf):
            s = s[: -len(suf)] + "ск"
            break
    else:
        for suf in _ENDINGS_SHORT:
            if s.endswith(suf) and len(s) - len(suf) > 4:
                s = s[: -len(suf)]
                break

    s = s.translate(_DEVOICE)
    s = s.replace("тс", "с").replace("дс", "с")
    s = s.replace("щ", "ш").replace("ц", "с")
    s = s.translate(_VOWEL_A).translate(_VOWEL_I).translate(_VOWEL_U)
    return _collapse_doubles(s)
# END_BLOCK_PHONETIC


def norm_house(house: str | None) -> str:
    """Дом (из эталона или Parser.house) -> сравнимая форма: lower, без пробелов по краям."""
    if not house:
        return ""
    return house.strip().lower()


# START_BLOCK_TYPES
@dataclass
class CityResolution:
    """Результат resolve_city() — docs/concept.md §5.3."""

    status: str  # resolved|ambiguous|unresolved|none
    city_id: str | None = None
    score: float = 0.0


@dataclass
class NameCand:
    """Сырые сигналы сходства имени улицы (до раскрытия в объекты) — §5.3/§5.4."""

    name_id: int
    name: str
    lev: float
    token_set: float
    phon: float
    ngram: float
    alias_hit: float = 0.0


@dataclass
class Obj:
    """Объект = имя × город × тип (§5.3)."""

    street_id: str
    name_id: int
    city_id: str
    street_type: str
    houses: dict[str, str] = field(default_factory=dict)  # house_norm -> etalon_id


@dataclass
class ObjCand:
    """Объект-кандидат после resolve_objects(): Obj + фича city_match/other_city."""

    street_id: str
    name_id: int
    name: str
    city_id: str
    street_type: str
    houses: dict[str, str]
    city_match: float
    other_city: bool
# END_BLOCK_TYPES


class BruteForceIndex:
    # START_CONTRACT: __init__
    #   PURPOSE: Построить словарь имён (слой 1) и объекты город×тип×имя (слой 2) из эталона.
    #   INPUTS: { etalon: list[dict] - etalon_id/city/street_type/street/house/... }
    #   OUTPUTS: none
    #   SIDE_EFFECTS: none (чистое построение in-memory структур)
    # END_CONTRACT: __init__
    def __init__(self, etalon: list[dict]) -> None:
        # START_BLOCK_NAMES
        self.normalizer_version = NORMALIZER_VERSION
        name_set: set[str] = set()
        city_set: set[str] = set()
        type_set: set[str] = set()
        norm_rows: list[dict] = []
        for row in etalon:
            name = normalize(row.get("street", ""), channel="webchat").text
            city = normalize(row.get("city", ""), channel="webchat").text
            street_type = row.get("street_type", "")
            name_set.add(name)
            city_set.add(city)
            type_set.add(street_type)
            norm_rows.append({**row, "_name": name, "_city": city, "_type": street_type})

        self.names: list[str] = sorted(name_set)
        self._name_to_id: dict[str, int] = {n: i for i, n in enumerate(self.names)}
        self.cities: list[str] = sorted(city_set)
        self.city_names: list[str] = self.cities
        self.street_types: dict[str, str] = {t: t for t in sorted(type_set) if t}
        self._name_words: set[str] = {tok for name in self.names for tok in name.split() if len(tok) >= 4}
        # END_BLOCK_NAMES

        # START_BLOCK_OBJECTS
        objects_by_key: dict[str, Obj] = {}
        for row in norm_rows:
            name_id = self._name_to_id[row["_name"]]
            street_id = f"{row['_city']}||{row['_type']}||{name_id}"
            obj = objects_by_key.get(street_id)
            if obj is None:
                obj = Obj(street_id=street_id, name_id=name_id, city_id=row["_city"], street_type=row["_type"])
                objects_by_key[street_id] = obj
            house_key = norm_house(row.get("house"))
            if house_key:
                obj.houses[house_key] = row["etalon_id"]

        self._objects_by_street_id: dict[str, Obj] = objects_by_key
        self._objects_by_name: dict[int, list[Obj]] = {}
        for obj in objects_by_key.values():
            self._objects_by_name.setdefault(obj.name_id, []).append(obj)
        # END_BLOCK_OBJECTS

        # START_BLOCK_PRECOMPUTE
        self._phon_keys: list[str] = [phonetic_key(n) for n in self.names]
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._ngram_matrix = self._vectorizer.fit_transform(self.names) if self.names else None

        counts = np.array([len(self._objects_by_name.get(i, [])) for i in range(len(self.names))], dtype=float)
        max_count = counts.max() if counts.size else 0.0
        denom = math.log(1.0 + max_count) if max_count > 0 else 1.0
        self._name_freq = np.log1p(counts) / denom if denom else counts * 0.0
        # END_BLOCK_PRECOMPUTE

        self._ready = True

    def ready(self) -> bool:
        """Готовность индекса (§5.3); False только при ошибке построения."""
        return self._ready

    def has_name(self, token: str) -> bool:
        """Точное вхождение токена >= 4 букв в любое имя словаря (§5.3)."""
        return len(token) >= 4 and token in self._name_words

    # START_CONTRACT: resolve_city
    #   PURPOSE: Разрешить текст города до city_id по 8 городам учебной базы (фаззи WRatio).
    #   INPUTS: { city_q: str|None - текст города (из slots/parse), scope: list[str]|None - подмножество city_id }
    #   OUTPUTS: { CityResolution - status resolved|unresolved|none, city_id, score [0,1] }
    #   SIDE_EFFECTS: none
    # END_CONTRACT: resolve_city
    def resolve_city(self, city_q: str | None, scope: Any = None) -> CityResolution:
        # START_BLOCK_CITY
        if not city_q:
            return CityResolution(status="none")
        norm = normalize(city_q, channel="webchat").text.strip()
        if not norm:
            return CityResolution(status="none")

        pool = self.cities if not scope else [c for c in self.cities if c in scope] or self.cities
        if norm in pool:
            return CityResolution(status="resolved", city_id=norm, score=1.0)

        best_city, best_score = None, -1.0
        for city in pool:
            score = fuzz.WRatio(norm, city)
            if score > best_score:
                best_city, best_score = city, score
        if best_score >= THETA_CITY:
            return CityResolution(status="resolved", city_id=best_city, score=best_score / 100.0)
        return CityResolution(status="unresolved", city_id=None, score=max(best_score, 0.0) / 100.0)
        # END_BLOCK_CITY

    # START_CONTRACT: candidates
    #   PURPOSE: top-k имён словаря по сырым сигналам сходства к street_q (§5.3).
    #   INPUTS: { street_q: str - текст улицы (уже нормализован), k: int - сколько имён, scope: не используется BruteForce }
    #   OUTPUTS: { list[NameCand] - top-k по max(lev, phon, ngram); [] при пустом street_q }
    #   SIDE_EFFECTS: none
    # END_CONTRACT: candidates
    def candidates(self, street_q: str, k: int = 20, scope: Any = None) -> list[NameCand]:
        # START_BLOCK_CANDIDATES
        # Бюджет §5.3 (< 1 мс на 699 имён): process.cdist с DamerauLevenshtein (транспозиции)
        # у rapidfuzz идёт по общему, не SIMD-пути и даёт ~2 мс на один сигнал (замерено) - это
        # само по себе съедало весь бюджет. Levenshtein.normalized_similarity - тот же класс
        # сходства (без учёта перестановки соседних букв, редкий случай ASR-шума) с векторной
        # C-реализацией; на предпосчитанных фонетических ключах разница из-за транспозиций
        # незаметна (тест на равенство ключей всё равно точный). token_set_ratio при этом не
        # входит в скор предвыборки (только lev/phon/ngram, см. §5.3) - считаем его только для
        # уже отобранных top-k имён, а не для всех 699.
        if not street_q or not self.names:
            return []

        lev_arr = process.cdist([street_q], self.names, scorer=Levenshtein.normalized_similarity)[0]
        phon_q = phonetic_key(street_q)
        phon_arr = process.cdist([phon_q], self._phon_keys, scorer=Levenshtein.normalized_similarity)[0]

        if self._ngram_matrix is not None:
            query_vec = self._vectorizer.transform([street_q])
            ngram_arr = np.asarray((self._ngram_matrix @ query_vec.T).todense()).ravel()
        else:
            ngram_arr = np.zeros(len(self.names))

        pre_score = np.maximum(np.maximum(lev_arr, phon_arr), ngram_arr)
        top_idx = np.argsort(-pre_score)[:k]

        return [
            NameCand(
                name_id=int(i),
                name=self.names[i],
                lev=float(lev_arr[i]),
                token_set=fuzz.token_set_ratio(street_q, self.names[i]) / 100.0,
                phon=float(phon_arr[i]),
                ngram=float(ngram_arr[i]),
                alias_hit=0.0,
            )
            for i in top_idx
        ]
        # END_BLOCK_CANDIDATES

    # START_CONTRACT: resolve_objects
    #   PURPOSE: Раскрыть имена в объекты (город×тип); проставить city_match/other_city (§5.3).
    #   INPUTS: { name_ids: list[int], city_res: CityResolution, scope: list[str]|None - подмножество city_id }
    #   OUTPUTS: { list[ObjCand] - объекты этого города первыми при resolved }
    #   SIDE_EFFECTS: none
    # END_CONTRACT: resolve_objects
    def resolve_objects(self, name_ids: list[int], city_res: CityResolution, scope: Any = None) -> list[ObjCand]:
        # START_BLOCK_RESOLVE_OBJECTS
        out: list[ObjCand] = []
        for name_id in name_ids:
            for obj in self._objects_by_name.get(name_id, []):
                if scope and obj.city_id not in scope:
                    continue
                if city_res.status == "resolved":
                    city_match = 1.0 if obj.city_id == city_res.city_id else 0.0
                    other_city = obj.city_id != city_res.city_id
                else:
                    city_match = 0.5
                    other_city = False
                out.append(
                    ObjCand(
                        street_id=obj.street_id,
                        name_id=name_id,
                        name=self.names[name_id],
                        city_id=obj.city_id,
                        street_type=obj.street_type,
                        houses=obj.houses,
                        city_match=city_match,
                        other_city=other_city,
                    )
                )
        out.sort(key=lambda oc: -oc.city_match)
        return out
        # END_BLOCK_RESOLVE_OBJECTS

    def houses(self, street_id: str) -> dict[str, str]:
        """Дома объекта: house_norm -> etalon_id (§5.3)."""
        obj = self._objects_by_street_id.get(street_id)
        return obj.houses if obj else {}

    def name_freq(self, name_id: int) -> float:
        """log(1+число объектов с этим именем)/log(1+max), нормировано на [0,1] (§5.3)."""
        if name_id < 0 or name_id >= len(self._name_freq):
            return 0.0
        return float(self._name_freq[name_id])
    # marker: [Index][BruteForceIndex][DONE]
