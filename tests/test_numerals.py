# START_MODULE_CONTRACT
#   PURPOSE: Тесты numerals.words_to_digits: числительные 1-999, фонетические варианты,
#            корпус/литера, слипания, неоднозначность.
#   SCOPE: tests/test_numerals.py — только addrmatch.numerals.
#   DEPENDS: M-NUMERALS
#   LINKS: V-M-NUMERALS
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_* - по одному сценарию из docs/concept.md §5.1 и plan.xml T-002 на тест
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-002: тесты числительных (>=12 кейсов из реальных данных ТЗ)
# END_CHANGE_SUMMARY

"""Тесты numerals.words_to_digits (docs/concept.md §5.1)."""

from addrmatch.numerals import words_to_digits


def test_hundred_tens_units_combine():
    assert words_to_digits("сто шестьдесят три") == ("163", False)


def test_teen_plus_hundred():
    assert words_to_digits("сто одиннадцать") == ("111", False)


def test_phonetic_dvatcat_vasem():
    # "дватцать васемь" - оглушение д->т, редукция о->а (акающий говор ASR)
    assert words_to_digits("дватцать васемь") == ("28", False)


def test_phonetic_chitire_unit():
    assert words_to_digits("улица правды читыре") == ("улица правды 4", False)


def test_corpus_trigger_k_with_digit_word():
    assert words_to_digits("два к два") == ("2к2", False)


def test_corpus_trigger_slovo_korpus():
    # "корпус" целиком (не сокращённое normalizer'ом "к") + фонетический вариант "шестьдесяд"
    assert words_to_digits("шестьдесяд один корпус один") == ("61к1", False)


def test_corpus_trigger_k_with_letter_word():
    assert words_to_digits("восемнадцать к а") == ("18кА", False)


def test_corpus_trigger_k_with_letter_already_digit():
    # число уже цифрой (пришло от нормализатора после первого прохода) - работает так же
    assert words_to_digits("18 к а") == ("18кА", False)


def test_korotche_trigger_letter_no_k():
    # "короче" - синоним корпус-маркера, но формат БЕЗ "к": число+ЛИТЕРА
    assert words_to_digits("сорок восемь короче бэ") == ("48Б", False)


def test_glued_ten_unit_then_corpus():
    # "пятьдесятдва" слитно (пятьдесят+два=52), плюс сотня перед ней и корпус после
    assert words_to_digits("сто пятьдесятдва к два") == ("152к2", False)


def test_bare_dom_sto_stays_separate_word():
    # "дом" - не числительное; numerals не трогает соседние обычные слова
    assert words_to_digits("дом сто") == ("дом 100", False)


def test_slippage_with_preceding_word():
    # "горькогосто" = "горького" (обычное слово) + "сто" (числительное), слипшиеся без пробела;
    # разделяем, только когда хвост реально продолжает цепочку разрядов со следующим числом
    assert words_to_digits("горькогосто шестьдесят три") == ("горького 163", False)


def test_ambiguous_order_violation():
    # "три сто" - разряд растёт (единицы -> сотни), порядок нарушен: первое прочтение + флаг
    result, ambiguous = words_to_digits("три сто")
    assert result == "3 100"
    assert ambiguous is True


def test_no_numbers_left_unchanged():
    text = "улица ленина без цифр вообще"
    assert words_to_digits(text) == (text, False)


def test_empty_string_unchanged():
    assert words_to_digits("") == ("", False)


def test_ordinary_word_ending_in_numeral_syllable_not_split():
    # регрессия: "индустри" оканчивается на "три" (числительное), но за ней идёт ГОЛАЯ цифра
    # "57" (не слово-числительное) - склейки с предыдущим словом быть не должно (N5/H4-смежный риск)
    assert words_to_digits("индустри 57") == ("индустри 57", False)


def test_letter_name_bare_no_trigger_not_merged_here():
    # без триггера (к/корпус/короче) numerals не трогает хвост - это забота normalizer (18а)
    assert words_to_digits("восемнадцать а") == ("18 а", False)
