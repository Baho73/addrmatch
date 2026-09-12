# START_MODULE_CONTRACT
#   PURPOSE: Разобрать нормализованную строку адреса на слоты: тип улицы, улица, дом (с
#            корпусом/литерой/диапазоном), city_hint; посчитать has_address_cues и
#            ambiguous_number для Decider (docs/concept.md §5.2).
#   SCOPE: Чистые функции над строкой/NormResult. Типы улиц (полная форма -> каноническая)
#          передаёт вызывающий (в будущем Index); парсер держит только регулярки над словарём.
#   DEPENDS: M-NORMALIZER (только тип NormResult, normalize() не вызывается)
#   LINKS: V-M-PARSER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   ParseResult - street_type/street/house/house_range/city_hint/has_address_cues/
#                 ambiguous_number/tail_tokens/tokens
#   DEFAULT_STREET_TYPES - 10 типов улиц эталона (etalon.jsonl:street_type), тождественная карта
#   parse - NormResult|str, street_types, has_name=None -> ParseResult; основной публичный вход
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-003: parser.py — тип/улица/дом(корпус/литера/диапазон/кв-отброс)/
#   city_hint/has_address_cues/ambiguous_number по docs/concept.md §5.2.
# END_CHANGE_SUMMARY

"""parser — нормализованная строка -> слоты адреса (docs/concept.md §5.2)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from addrmatch.normalizer import NormResult

# START_BLOCK_TYPES
# 10 значений etalon.jsonl:street_type (test_task_adress_match/data/etalon.jsonl). Карта
# "полная форма -> каноническая" тождественна для учебной базы; на ГАР карту строит Index.
DEFAULT_STREET_TYPES: dict[str, str] = {
    t: t
    for t in (
        "улица", "проспект", "переулок", "бульвар", "шоссе",
        "набережная", "площадь", "проезд", "микрорайон", "квартал",
    )
}


@dataclass
class ParseResult:
    """Результат parse() — слоты по docs/concept.md §5.2."""

    street_type: str | None = None
    street: str | None = None
    house: str | None = None
    house_range: bool = False
    city_hint: str | None = None
    has_address_cues: bool = False
    ambiguous_number: bool = False
    tail_tokens: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
# END_BLOCK_TYPES

# START_BLOCK_REGEX
# Токен "буквенный" — только кириллица и внутреннее тире ("рабоче-крестьянская"); дом-токены
# начинаются с цифры и сюда не попадают.
_ALPHA_TOKEN_RE = re.compile(r"^[а-яёА-ЯЁ\-]+$")
_HOUSE_HEAD_RE = re.compile(r"^\d")

# Хвост дома одним токеном (§5.2): "16" | "16а"/"48Б" (литера) | "16к2"/"61к1" (корпус числом,
# в т.ч. уже слитый numerals.merge_literal) | "16кА" (корпус буквой) | "16/2"/"52/а" (= корпус,
# "/N"|"/буква" не трогает normalizer — цифра-цифра/буква) | "16-18" (диапазон, флаг house_range).
_HOUSE_TOKEN_RE = re.compile(
    r"^(?P<num>\d+)"
    r"(?:"
    r"-(?P<range_num>\d+)"
    r"|/(?P<slash_num>\d+)"
    r"|/(?P<slash_letter>[а-яёА-ЯЁ])"
    r"|к(?P<k_num>\d+)"
    r"|к(?P<k_letter>[а-яёА-ЯЁ])"
    r"|(?P<letter>[а-яёА-ЯЁ])"
    r")?$"
)

_APARTMENT_MARKERS = {"кв", "квартира"}

# Числа внутри самого ИМЕНИ улицы, а не дом: порядковые "2-я"/"1-й"/"40-летия" (дефис+буквы —
# форма, которой нет ни в одном валидном хвосте дома §5.2, значит это не может быть домом) и
# годовщины/даты "60 лет ...", "9 мая ..." (число сразу перед этими словами — год/дата в
# названии, реальный дом стоит дальше в строке).
_ORDINAL_NAME_RE = re.compile(r"^\d+-[а-яёА-ЯЁ]+$")
_YEAR_DATE_FOLLOWERS = {
    "лет", "летия",
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
}
# END_BLOCK_REGEX


def _is_alpha_token(tok: str) -> bool:
    return bool(_ALPHA_TOKEN_RE.match(tok))


# START_BLOCK_TYPE_SPLIT
def _try_split_glued_type(tok: str, street_types: dict[str, str]) -> tuple[str, str] | None:
    """Слипание "шоссемосковское": тип >= 5 букв в начале токена, остаток >= 4 буквы."""
    for full in sorted(street_types, key=len, reverse=True):
        if len(full) < 5:
            continue
        if tok.startswith(full) and len(tok) - len(full) >= 4:
            return full, tok[len(full):]
    return None


def _find_street_type(tokens: list[str], street_types: dict[str, str]) -> int | None:
    """Найти индекс токена-типа улицы; при слипании — расщепить tokens на месте (мутация)."""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in street_types:
            return i
        split = _try_split_glued_type(tok, street_types)
        if split is not None:
            full, remainder = split
            tokens[i:i + 1] = [full, remainder]
            return i
        i += 1
    return None
# END_BLOCK_TYPE_SPLIT


# START_BLOCK_HOUSE
def _is_name_number(tok: str, next_tok: str | None) -> bool:
    """Число — часть имени улицы ("2-я портовая", "60 лет октября"), не дом."""
    if _ORDINAL_NAME_RE.match(tok):
        return True
    return next_tok is not None and next_tok in _YEAR_DATE_FOLLOWERS


def _find_house_index(tokens: list[str]) -> int | None:
    """Первый токен с цифры после хотя бы одного буквенного токена (пропуская числа-части
    имени улицы, см. _is_name_number); либо единственный числовой токен строки целиком
    (ответ на дозапрос "60")."""
    if len(tokens) == 1 and _HOUSE_HEAD_RE.match(tokens[0]):
        return 0
    seen_alpha = False
    for i, tok in enumerate(tokens):
        if _is_alpha_token(tok):
            seen_alpha = True
            continue
        if not (seen_alpha and _HOUSE_HEAD_RE.match(tok)):
            continue
        next_tok = tokens[i + 1] if i + 1 < len(tokens) else None
        if _is_name_number(tok, next_tok):
            continue
        return i
    return None


def _parse_house_token(base: str) -> tuple[str, bool]:
    """Разобрать одиночный дом-токен -> (нормализованный house, house_range)."""
    m = _HOUSE_TOKEN_RE.match(base)
    if m is None:
        return base, False
    num = m.group("num")
    if m.group("range_num"):
        return num, True
    if m.group("slash_num"):
        return f"{num}к{m.group('slash_num')}", False
    if m.group("slash_letter"):
        return f"{num}к{m.group('slash_letter')}", False
    if m.group("k_num"):
        return f"{num}к{m.group('k_num')}", False
    if m.group("k_letter"):
        return f"{num}к{m.group('k_letter')}", False
    if m.group("letter"):
        return f"{num}{m.group('letter')}", False
    return num, False


def _consume_house_tail(tokens: list[str], house_idx: int, house: str) -> tuple[str, list[str], bool]:
    """После дома: "кв 5"/"квартира N" отбросить; "с5" (строение, не склеенное normalizer'ом
    из-за обратного порядка буква-цифра) — присоединить; голое число подряд — неоднозначность
    (§5.2: "3 1" -> house "3", хвост в tail_tokens, ambiguous_number=True)."""
    tail_tokens: list[str] = []
    ambiguous = False
    nxt_idx = house_idx + 1
    if nxt_idx >= len(tokens):
        return house, tail_tokens, ambiguous
    nxt = tokens[nxt_idx]
    if nxt in _APARTMENT_MARKERS:
        pass  # квартира отброшена целиком, дом не трогаем
    elif nxt == "с" and nxt_idx + 1 < len(tokens) and tokens[nxt_idx + 1].isdigit():
        house = f"{house}с{tokens[nxt_idx + 1]}"
    elif nxt.isdigit():
        tail_tokens.append(nxt)
        ambiguous = True
    return house, tail_tokens, ambiguous
# END_BLOCK_HOUSE


# START_CONTRACT: parse
#   PURPOSE: Разобрать нормализованную строку адреса на слоты по docs/concept.md §5.2.
#   INPUTS: { norm: NormResult|str - результат normalizer.normalize() либо готовая строка,
#             street_types: dict[str, str] - "полная форма -> каноническая" (из справочника),
#             has_name: Callable[[str], bool]|None - словарь имён улиц (Index.has_name) для
#                       has_address_cues, когда нет ни типа, ни дома }
#   OUTPUTS: { ParseResult }
#   SIDE_EFFECTS: none
# END_CONTRACT: parse
def parse(
    norm: NormResult | str,
    street_types: dict[str, str],
    has_name: Callable[[str], bool] | None = None,
) -> ParseResult:
    # START_BLOCK_TOKENIZE
    if isinstance(norm, NormResult):
        text = norm.text
        ambiguous_number = norm.ambiguous_number
    else:
        text = norm
        ambiguous_number = False
    tokens = text.split() if text else []
    # END_BLOCK_TOKENIZE

    # START_BLOCK_TYPE
    type_idx = _find_street_type(tokens, street_types)
    street_type = street_types[tokens[type_idx]] if type_idx is not None else None
    # END_BLOCK_TYPE

    # START_BLOCK_HOUSE_SLOT
    house_idx = _find_house_index(tokens)
    house: str | None = None
    house_range = False
    tail_tokens: list[str] = []
    if house_idx is not None:
        house, house_range = _parse_house_token(tokens[house_idx])
        if not house_range:
            house, tail_tokens, tail_ambiguous = _consume_house_tail(tokens, house_idx, house)
            ambiguous_number = ambiguous_number or tail_ambiguous
    # END_BLOCK_HOUSE_SLOT

    # START_BLOCK_STREET_CITY
    start = type_idx + 1 if type_idx is not None else 0
    end = house_idx if house_idx is not None else len(tokens)
    street_tokens = tokens[start:end]
    street = " ".join(street_tokens) if street_tokens else None

    city_hint = None
    if type_idx is not None and type_idx > 0:
        pre = tokens[:type_idx]
        if any(any(ch.isalpha() for ch in tok) for tok in pre):
            city_hint = " ".join(pre)
    # END_BLOCK_STREET_CITY

    # START_BLOCK_CUES
    has_cues = street_type is not None or house is not None
    if not has_cues and has_name is not None:
        has_cues = any(has_name(tok) for tok in tokens if len(tok) >= 4)
    # END_BLOCK_CUES

    return ParseResult(
        street_type=street_type,
        street=street,
        house=house,
        house_range=house_range,
        city_hint=city_hint,
        has_address_cues=has_cues,
        ambiguous_number=ambiguous_number,
        tail_tokens=tail_tokens,
        tokens=tokens,
    )
    # marker: [Parser][parse][DONE]
