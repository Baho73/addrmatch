# START_MODULE_CONTRACT
#   PURPOSE: Заменить русские числительные словами (в т.ч. фонетические/опечаточные варианты и
#            слипания со словом-соседом) на цифры внутри строки; свернуть хвост дома
#            корпус/литера в нормализованную форму (16к2, 48Б).
#   SCOPE: Чистая функция над строкой; ничего не знает про справочник, Index, каналы связи.
#   DEPENDS: none
#   LINKS: V-M-NUMERALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   words_to_digits - str -> (str, ambiguous: bool); основной публичный вход модуля
#   is_numeral_word - True, если токен (после фонетической свёртки) сам является числительным
#   is_house_trigger - True, если токен - триггер хвоста дома (к/корпус/дробь/короче + опечатки)
#   LETTER_NAMES - словарь "слово-литера" (а/бэ/вэ/пэ/ка/...) -> заглавная буква для normalizer
#   NUMERALS_VERSION - версия словаря числительных (входит в NORMALIZER_VERSION)
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-002: numerals.py — словарь числительных 1-999 с фонетическими
#   вариантами (свёртка по озвончению/оглушению и редукции гласных), разбор слипаний
#   ("горькогосто" -> "горького"+"сто"), свёртка корпус/литера ("к а" -> "кА", "короче бэ" -> "Б").
#   C-ADDRMATCH-PHASE-A T-003b: доводка гейта T-003 (house_found_rate 0.865 -> цель 0.95).
#   "дробь" — триггер наравне с "к/корпус" (7 дробь 2 -> 7к2); триггеры сравниваются через
#   _collapse (карпус/корпуз/тробь — опечатки бьют в те же ключи, без отдельного списка);
#   LETTER_NAMES расширен до алфавита эталона (е/ж/п/ка/эс/эф); новая свёртка "число + голая
#   МНОГОбуквенная буква-имя без триггера" (195 пэ -> 195П, два гэ -> 2Г); разбор слипания
#   триггер+суффикс без пробела ("кдва" -> к+2, "кб" -> к+Б); опечатка "двадтсать" (доб. в
#   _TYPO_ALIASES, посимвольная свёртка её не покрывает). NUMERALS_VERSION v1 -> v2.
# END_CHANGE_SUMMARY

"""numerals — русские числительные словами -> цифры (docs/concept.md §5.1)."""

from __future__ import annotations

import re

# START_BLOCK_DICT
# Словарь числительных 1-999: разряд (tier) используется, чтобы решить, можно ли склеить
# соседние числительные в одно число. Разряды строго убывают при склейке: сотни(3) -> десятки
# или спец.числа 11-19(2) -> единицы(1). Пример: "сто" + "пятьдесят" + "два" = 152.
_UNITS = {
    "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
}
_TEENS = {
    "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17,
    "восемнадцать": 18, "девятнадцать": 19,
}
_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
    "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
}
_HUNDREDS = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400, "пятьсот": 500,
    "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
}

# Буквы-литеры дома: "бэ"/"вэ"/"дэ"/"гэ"/... — озвученные названия букв; голая "а" — тоже частый
# способ произнести литеру ("15 а"). Расширено до алфавита, встречающегося в эталоне
# (А,Б,В,Г,Д,К,П + Е/Ж/С/Ф про запас) + "ка"/"эс"/"эф" — фонетические имена букв К/С/Ф.
LETTER_NAMES = {
    "а": "А",
    "б": "Б", "бэ": "Б",
    "в": "В", "вэ": "В",
    "г": "Г", "гэ": "Г",
    "д": "Д", "дэ": "Д",
    "е": "Е",
    "ж": "Ж", "жэ": "Ж",
    "п": "П", "пэ": "П",
    "ка": "К",
    "эс": "С",
    "эф": "Ф",
}

# Триггеры хвоста дома (сырые слова; опечатки не перечисляем - см. _TRIGGER_K/_TRIGGER_KOROCHE
# ниже, построенные через _collapse, чтобы "карпус"/"корпуз"/"тробь" ловились автоматически).
# _TRIGGER_K_WORDS даёт формат N + "к" + X (число или литера); "дробь" - тот же формат
# ("7 дробь 2" -> 7к2). _TRIGGER_KOROCHE_WORDS даёт формат N + ЛИТЕРА без "к" (короче бэ -> 48Б).
_TRIGGER_K_WORDS = {"к", "корпус", "корп", "дробь"}
_TRIGGER_KOROCHE_WORDS = {"короче"}

NUMERALS_VERSION = "v2"
# END_BLOCK_DICT

# START_BLOCK_PHONETIC
# Свёртка фонетических/опечаточных вариантов к одному ключу: озвончение/оглушение согласных
# на стыке (дватцать~двадцать, шестьдесяд~шестьдесят) и редукция гласных о/е/ё/я (васемь~восемь,
# читыре~четыре, деветь/девядь~девять). Проверено на реальных вариантах из adresses_labeled.jsonl.
_VOICED_TO_VOICELESS = str.maketrans("бвгджз", "пфктшс")
_VOWEL_REDUCE = str.maketrans("оеёя", "аиии")


def _collapse(word: str) -> str:
    return word.lower().translate(_VOICED_TO_VOICELESS).translate(_VOWEL_REDUCE)


_ALL_WORDS = {**_UNITS, **_TEENS, **_TENS, **_HUNDREDS}
_TIER_OF = {}
for _w in _UNITS:
    _TIER_OF[_w] = 1
for _w in _TEENS:
    _TIER_OF[_w] = 2
for _w in _TENS:
    _TIER_OF[_w] = 2
for _w in _HUNDREDS:
    _TIER_OF[_w] = 3

# collapsed_key -> (value, tier)
NUMERAL_LOOKUP: dict[str, tuple[int, int]] = {
    _collapse(_w): (_v, _TIER_OF[_w]) for _w, _v in _ALL_WORDS.items()
}

# Опечатки, которые НЕ покрывает посимвольная свёртка озвончения/редукции выше (вставка/замена
# слога, а не одной буквы): "двадтсать" - "ц" услышана как "тс" (из adresses_labeled).
_TYPO_ALIASES = {"двадтсать": "двадцать"}
for _typo, _canonical in _TYPO_ALIASES.items():
    NUMERAL_LOOKUP[_collapse(_typo)] = (_ALL_WORDS[_canonical], _TIER_OF[_canonical])

# Триггеры хвоста дома через ту же свёртку, что и числительные - "карпус"/"корпуз"/"тробь"
# (опечатки корпус/дробь) ловятся автоматически, без отдельного списка вариантов.
_TRIGGER_K = {_collapse(_w) for _w in _TRIGGER_K_WORDS}
_TRIGGER_KOROCHE = {_collapse(_w) for _w in _TRIGGER_KOROCHE_WORDS}
_ALL_TRIGGERS = _TRIGGER_K | _TRIGGER_KOROCHE


def is_numeral_word(word: str) -> bool:
    """True, если word целиком является числительным (цифрой или словом-числительным)."""
    if not word:
        return False
    if word.isdigit():
        return True
    return _collapse(word) in NUMERAL_LOOKUP


def is_house_trigger(word: str) -> bool:
    """True, если word (с опечатками/фонетикой) - триггер хвоста дома (к/корпус/корп/дробь/
    короче); используется normalizer'ом, чтобы не терять литеру после "дробь"."""
    if not word:
        return False
    return _collapse(word) in _ALL_TRIGGERS
# END_BLOCK_PHONETIC

# START_BLOCK_SEGMENT
def _segment_word(word: str) -> list[tuple]:
    """Разбить слово на числительные: ("NUM", value, tier) | ("TEXT", original).
    Только целое слово или слитные ЧИСЛИТЕЛЬНОЕ+ЧИСЛИТЕЛЬНОЕ без текстового остатка
    ("пятьдесятдва" -> пятьдесят+два): любое обычное слово может случайно ОКАНЧИВАТЬСЯ на
    числительное ("индустри" оканчивается на "три"), поэтому склейку с ПРЕДЫДУЩИМ обычным
    словом ("горькогосто" -> "горького"+"сто") решает не эта функция, а _peel_glued_prefix
    ниже - только когда результат реально продолжает цепочку разрядов со следующим числом."""
    key = _collapse(word)
    if key in NUMERAL_LOOKUP:
        value, tier = NUMERAL_LOOKUP[key]
        return [("NUM", value, tier)]

    for split in range(len(word) - 1, 0, -1):
        left, right = word[:split], word[split:]
        if _collapse(right) in NUMERAL_LOOKUP:
            left_parts = _segment_word(left)
            if all(p[0] == "NUM" for p in left_parts):
                value, tier = NUMERAL_LOOKUP[_collapse(right)]
                return left_parts + [("NUM", value, tier)]

    # Триггер "к" слипся со следующим словом без пробела ("кдва" -> "к"+2, "кб" -> "к"+"б") -
    # только префикс "к" (единственный однобуквенный триггер) и только когда слово
    # целиком не является само по себе именем буквы (иначе "ка" перестанет читаться как литера К).
    if len(word) > 1 and word[0].lower() == "к" and word.lower() not in LETTER_NAMES:
        rest = word[1:]
        if _collapse(rest) in NUMERAL_LOOKUP:
            value, tier = NUMERAL_LOOKUP[_collapse(rest)]
            return [("TEXT", word[0]), ("NUM", value, tier)]
        if rest.lower() in LETTER_NAMES:
            return [("TEXT", word[0]), ("TEXT", rest)]

    return [("TEXT", word)]


def _peel_glued_prefix(flat: list[dict]) -> None:
    """Слипание с ПРЕДЫДУЩИМ словом ("горькогосто" -> "горького" 100): отделяем хвостовое
    числительное от обычного слова, только если оно затем реально склеится (строго меньший
    разряд) со следующим числительным - иначе это просто совпадение окончания слова."""
    i = 1
    while i + 2 < len(flat):
        tok, gap, nxt = flat[i], flat[i + 1], flat[i + 2]
        if tok["kind"] == "text" and _is_blank_gap(gap) and nxt["kind"] == "num" and nxt["tier"] is not None:
            word = tok["text"]
            for suf_len in range(len(word) - 4, 0, -1):
                prefix, suffix = word[:-suf_len], word[-suf_len:]
                key = _collapse(suffix)
                if key in NUMERAL_LOOKUP:
                    value, tier = NUMERAL_LOOKUP[key]
                    if tier > nxt["tier"]:
                        replacement = []
                        if prefix:
                            replacement.append({"kind": "text", "text": prefix})
                            replacement.append({"kind": "gap", "text": " "})
                        replacement.append({"kind": "num", "text": str(value), "value": value, "tier": tier})
                        flat[i:i + 1] = replacement
                        i += len(replacement) - 1
                        break
        i += 2
# END_BLOCK_SEGMENT

# START_BLOCK_PIPELINE
_TOKEN_RE = re.compile(r"[а-яёА-ЯЁ]+|\d+")


def _tokenize(text: str) -> tuple[list[str], list[str]]:
    gaps: list[str] = []
    tokens: list[str] = []
    last = 0
    for m in _TOKEN_RE.finditer(text):
        gaps.append(text[last:m.start()])
        tokens.append(m.group(0))
        last = m.end()
    gaps.append(text[last:])
    return gaps, tokens


def _build_flat(gaps: list[str], tokens: list[str]) -> list[dict]:
    flat: list[dict] = [{"kind": "gap", "text": gaps[0]}]
    for i, tok in enumerate(tokens):
        if tok.isdigit():
            flat.append({"kind": "num", "text": tok, "value": int(tok), "tier": None})
        else:
            parts = _segment_word(tok)
            for j, part in enumerate(parts):
                if j > 0:
                    flat.append({"kind": "gap", "text": " "})
                if part[0] == "NUM":
                    flat.append({"kind": "num", "text": str(part[1]), "value": part[1], "tier": part[2]})
                else:
                    flat.append({"kind": "text", "text": part[1]})
        flat.append({"kind": "gap", "text": gaps[i + 1]})
    return flat


def _is_blank_gap(item: dict) -> bool:
    return item["kind"] == "gap" and item["text"].strip() == ""


def _combine_tiers(flat: list[dict]) -> bool:
    """Склеить убывающие разряды подряд идущих числительных-слов в одно число.
    Возвращает ambiguous=True, если порядок разрядов был нарушен (не убывал строго)."""
    ambiguous = False
    i = 1
    while i < len(flat):
        if flat[i]["kind"] == "num" and flat[i]["tier"] is not None:
            while i + 2 < len(flat):
                gap, nxt = flat[i + 1], flat[i + 2]
                if not _is_blank_gap(gap) or nxt["kind"] != "num" or nxt["tier"] is None:
                    break
                if nxt["tier"] < flat[i]["tier"]:
                    merged_value = flat[i]["value"] + nxt["value"]
                    flat[i] = {"kind": "num", "text": str(merged_value), "value": merged_value, "tier": nxt["tier"]}
                    del flat[i + 1:i + 3]
                    continue
                ambiguous = True
                break
        i += 2
    return ambiguous


def _merge_literal(flat: list[dict]) -> None:
    """Свернуть хвост дома: NUM (к|корпус|дробь,опечатки) (NUM|литера) -> "NкX";
    NUM короче литера -> "NX". Триггер сравнивается через _collapse (карпус/корпуз/тробь)."""
    i = 1
    while i + 4 < len(flat):
        cur, gap1, mid, gap2, nxt = flat[i], flat[i + 1], flat[i + 2], flat[i + 3], flat[i + 4]
        if cur["kind"] != "num" or not _is_blank_gap(gap1) or not _is_blank_gap(gap2):
            i += 2
            continue
        mid_key = _collapse(mid["text"]) if mid["kind"] == "text" else None
        nxt_is_letter = nxt["kind"] == "text" and nxt["text"].lower() in LETTER_NAMES
        if mid_key in _TRIGGER_K and (nxt["kind"] == "num" or nxt_is_letter):
            tail = str(nxt["value"]) if nxt["kind"] == "num" else LETTER_NAMES[nxt["text"].lower()]
            flat[i] = {"kind": "text", "text": f"{cur['value']}к{tail}"}
            del flat[i + 1:i + 5]
            continue
        if mid_key in _TRIGGER_KOROCHE and nxt_is_letter:
            flat[i] = {"kind": "text", "text": f"{cur['value']}{LETTER_NAMES[nxt['text'].lower()]}"}
            del flat[i + 1:i + 5]
            continue
        i += 2


def _merge_bare_letter_name(flat: list[dict]) -> None:
    """Число + голая МНОГОБУКВЕННАЯ буква-имя без триггера ("два гэ" -> "2Г", "195 пэ" -> "195П").
    Однобуквенные формы (а/б/.../е/п) сюда не относятся - без триггера их в нижнем
    регистре склеивает normalizer (18а), см. test_letter_name_bare_no_trigger_not_merged_here.
    nxt["text"] == .lower() - обязательное условие идемпотентности: уже свёрнутый триггером
    хвост "26кА" на повторном проходе ретокенизируется в ОДИН смешанный по регистру токен "кА"
    (буквы к/А - один символьный класс без учёта регистра) - это не то же самое, что фонетическое
    имя буквы "ка" (нижний регистр), которое здесь распознаём; без проверки регистра "26кА" на
    втором проходе портился бы в "26К".
    """
    i = 1
    while i + 2 < len(flat):
        cur, gap, nxt = flat[i], flat[i + 1], flat[i + 2]
        if (
            cur["kind"] == "num"
            and _is_blank_gap(gap)
            and nxt["kind"] == "text"
            and len(nxt["text"]) >= 2
            and nxt["text"] == nxt["text"].lower()
            and nxt["text"] in LETTER_NAMES
        ):
            flat[i] = {"kind": "text", "text": f"{cur['value']}{LETTER_NAMES[nxt['text']]}"}
            del flat[i + 1:i + 3]
            continue
        i += 2
# END_BLOCK_PIPELINE

# START_CONTRACT: words_to_digits
#   PURPOSE: Заменить числительные словами на цифры и свернуть хвост дома корпус/литера.
#   INPUTS: { text: str - строка (обычно уже lower/ё->е из normalizer, но функция самодостаточна) }
#   OUTPUTS: { tuple[str, bool] - (текст с цифрами, ambiguous_number: нарушен порядок разрядов) }
#   SIDE_EFFECTS: none
# END_CONTRACT: words_to_digits
def words_to_digits(text: str) -> tuple[str, bool]:
    # START_BLOCK_RUN
    gaps, tokens = _tokenize(text)
    flat = _build_flat(gaps, tokens)
    _peel_glued_prefix(flat)
    ambiguous = _combine_tiers(flat)
    _merge_literal(flat)
    _merge_bare_letter_name(flat)
    result = "".join(item["text"] for item in flat)
    return result, ambiguous
    # END_BLOCK_RUN
