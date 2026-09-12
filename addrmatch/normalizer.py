# START_MODULE_CONTRACT
#   PURPOSE: Свести сырую строку адреса (ASR/чат) к канонической текстовой форме перед Parser:
#            регистр, сокращения, мусорные слова, числительные словами -> цифры, склейки.
#   SCOPE: Чистая функция над (text, channel); не знает про справочник (Index/Parser/данные) - N5.
#   DEPENDS: M-NUMERALS
#   LINKS: V-M-NORMALIZER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   NormResult - text/truncated/ambiguous_number
#   normalize - str, channel -> NormResult; основной публичный вход модуля
#   NORMALIZER_VERSION - sha1(правила)[:12]; входит в хэш словаря для Index (§5.1)
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-002: normalizer.py — регистр/ё/латиница-двойники (webchat)/пунктуация/
#   сокращения/стоп-слова/words_to_digits/склейки, идемпотентность, NORMALIZER_VERSION.
# END_CHANGE_SUMMARY

"""normalizer — строка -> строка перед Parser (docs/concept.md §5.1)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from addrmatch.numerals import LETTER_NAMES, NUMERALS_VERSION, is_numeral_word, words_to_digits

MAX_LEN = 512


@dataclass
class NormResult:
    """Результат normalize() — см. docs/concept.md §5.1."""

    text: str
    truncated: bool = False
    ambiguous_number: bool = False


# START_BLOCK_RULES
# Латиница-двойники -> кириллица (только webchat, только внутри слов, где уже есть кириллица —
# иначе рискуем испортить настоящий латинский текст).
_LATIN_LOOKALIKE = str.maketrans("aeopcxykmthb", "аеорсхукмтнв")

# Сокращения раскрываются по точному совпадению токена (пунктуация уже заменена на пробелы,
# поэтому "ул." превращается в токен "ул" до этого шага). Значение "" - маркер убирается,
# число (дом/строение) остаётся без него.
_ABBREVIATIONS = {
    "ул": "улица",
    "пр-т": "проспект", "просп": "проспект", "пр": "проспект",
    "пер": "переулок",
    "б-р": "бульвар", "бул": "бульвар",
    "ш": "шоссе",
    "наб": "набережная",
    "пл": "площадь",
    "пр-д": "проезд",
    "мкр": "микрорайон",
    "кв-л": "квартал",
    "д": "", "дом": "",
    "к": "к", "корп": "к", "корпус": "к",
    "стр": "с",
    "лит": "",
}

# Стоп-слова удаляются целыми токенами. "короче"/"а"/"с" исключаются условно (см. _drop_stopwords):
# они нужны numerals для хвоста дома (короче бэ -> Б; 15 а -> литера) или как маркер строения
# ("стр." -> "с" 5) - иначе "с" как предлог ("рядом с домом") съедает и маркер строения.
_STOPWORDS = {
    "живу", "я", "на", "мне", "удобно", "ну", "вот", "это", "самое", "если", "знаете",
    "как", "бы", "давайте", "мой", "адрес", "рядом", "с", "домом", "так", "короче",
    "а", "э", "там", "тут", "вроде", "кажется", "значит",
}


def _build_version() -> str:
    payload = "|".join(
        [
            NUMERALS_VERSION,
            ",".join(sorted(f"{k}={v}" for k, v in _ABBREVIATIONS.items())),
            ",".join(sorted(_STOPWORDS)),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


NORMALIZER_VERSION = _build_version()
# END_BLOCK_RULES

# START_BLOCK_LOWER
# Обычный .lower() необратимо портит уже нормализованную литеру дома (18кА -> 18ка при повторном
# вызове), а идемпотентность обязательна (§5.1). Поэтому бережём хвост "цифры[+к]+ЗАГЛАВНАЯ_БУКВА"
# перед соседним словом/концом строки - остальное лочим как обычно.
_LITERAL_TAIL_RE = re.compile(r"\d+к?[А-ЯЁ](?![а-яА-ЯёЁ0-9])")


def _lower_preserving_literals(text: str) -> str:
    parts = []
    last = 0
    for m in _LITERAL_TAIL_RE.finditer(text):
        parts.append(text[last:m.start()].lower())
        parts.append(m.group(0))
        last = m.end()
    parts.append(text[last:].lower())
    return "".join(parts)


# "/" и "-" оставляем нетронутыми, только когда с обеих сторон буква/цифра (16/2, 22-й,
# рабоче-крестьянская) - остальную пунктуацию всегда в пробел.
_PUNCT_RE = re.compile(r"[^\w\s/\-]|_", re.UNICODE)
_BARE_DASH_SLASH_RE = re.compile(r"(?<![0-9а-яё])[/\-]|[/\-](?![0-9а-яё])")


def _punctuation_to_spaces(text: str) -> str:
    text = _PUNCT_RE.sub(" ", text)
    return _BARE_DASH_SLASH_RE.sub(" ", text)


_DIGIT_LETTER_GLUE_RE = re.compile(r"([а-яё]{3,})(\d+)|(\d+)([а-яё]{3,})")


def _split_digit_letter_glue(text: str) -> str:
    def repl(m: re.Match) -> str:
        if m.group(1) is not None:
            return f"{m.group(1)} {m.group(2)}"
        return f"{m.group(3)} {m.group(4)}"

    return _DIGIT_LETTER_GLUE_RE.sub(repl, text)


def _latin_to_cyrillic(text: str, channel: str) -> str:
    if channel != "webchat":
        return text
    out_words = []
    for word in text.split(" "):
        if any("а" <= ch <= "я" or ch == "ё" for ch in word):
            out_words.append(word.translate(_LATIN_LOOKALIKE))
        else:
            out_words.append(word)
    return " ".join(out_words)


def _drop_stopwords(text: str) -> str:
    tokens = text.split()
    kept: list[str] = []
    for i, tok in enumerate(tokens):
        if tok not in _STOPWORDS:
            kept.append(tok)
            continue
        if tok == "короче":
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is not None and nxt.lower() in LETTER_NAMES:
                kept.append(tok)
            continue
        if tok == "а":
            prev = kept[-1] if kept else None
            if prev is not None and (is_numeral_word(prev) or prev in {"к", "корпус", "корп", "короче"}):
                kept.append(tok)
            continue
        if tok == "с":
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is not None and (nxt.isdigit() or is_numeral_word(nxt)):
                kept.append(tok)
            continue
        # обычный стоп-токен - пропускаем
    return " ".join(kept)


def _expand_abbreviations(text: str) -> str:
    tokens = [_ABBREVIATIONS.get(tok, tok) for tok in text.split()]
    return " ".join(t for t in tokens if t != "")


_BARE_LITERAL_GLUE_RE = re.compile(r"(?<![а-яёА-ЯЁ0-9])(\d+)\s+([а-яё])(?![а-яёА-ЯЁ0-9])")


def _glue_bare_literal(text: str) -> str:
    return _BARE_LITERAL_GLUE_RE.sub(r"\1\2", text)


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
# END_BLOCK_RULES

# START_CONTRACT: normalize
#   PURPOSE: Свести сырую строку адреса к канонической форме перед Parser (§5.1).
#   INPUTS: { text: str - сырая строка, channel: str - voice|webchat (латиница-двойники только webchat) }
#   OUTPUTS: { NormResult - text/truncated/ambiguous_number }
#   SIDE_EFFECTS: none
# END_CONTRACT: normalize
def normalize(text: str, channel: str = "voice") -> NormResult:
    # START_BLOCK_TRUNCATE
    truncated = len(text) > MAX_LEN
    if truncated:
        text = text[:MAX_LEN]
    # END_BLOCK_TRUNCATE

    # START_BLOCK_CASE
    text = _lower_preserving_literals(text)
    text = text.replace("ё", "е")
    text = _latin_to_cyrillic(text, channel)
    # END_BLOCK_CASE

    # START_BLOCK_CLEANUP
    text = _punctuation_to_spaces(text)
    text = _split_digit_letter_glue(text)
    text = _expand_abbreviations(text)
    text = _drop_stopwords(text)
    # END_BLOCK_CLEANUP

    # START_BLOCK_NUMERALS
    text, ambiguous_number = words_to_digits(text)
    text = _glue_bare_literal(text)
    text = _collapse_spaces(text)
    # END_BLOCK_NUMERALS

    return NormResult(text=text, truncated=truncated, ambiguous_number=ambiguous_number)
    # marker: [Normalizer][normalize][DONE]
