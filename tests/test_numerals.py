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


# START_CHANGE_SUMMARY (доп.)
#   C-ADDRMATCH-PHASE-A T-003b: тесты на доводку house_found_rate (дробь-триггер, опечатки
#   триггеров, голая многобуквенная буква-имя, слипание триггер+суффикс, идемпотентность).
# END_CHANGE_SUMMARY


def test_drob_trigger_with_digit():
    # "дробь" - триггер наравне с "к"/"корпус" (T-003b, 19 промахов гейта T-003)
    assert words_to_digits("семь дробь два") == ("7к2", False)


def test_drob_trigger_typo_and_korpus_typo():
    # "тробь" (опечатка "дробь", д->т) и "карпус" (опечатка "корпус") ловятся через ту же
    # фонетическую свёртку, что и числительные - без отдельного списка вариантов
    assert words_to_digits("сорок шесть тробь один") == ("46к1", False)
    assert words_to_digits("семь карпус один") == ("7к1", False)


def test_bare_multiletter_name_no_trigger():
    # голая буква-имя без триггера (T-003b, ~15 промахов): "195 пэ"->195П, "два гэ"->2Г
    assert words_to_digits("сто девяносто пять пэ") == ("195П", False)
    assert words_to_digits("два гэ") == ("2Г", False)


def test_bare_letter_ka_is_letter_k_not_corpus_a():
    # "ка" - фонетическое имя буквы К (не "к"+"а"): голое "3 ка" -> "3К"
    assert words_to_digits("три ка") == ("3К", False)


def test_merged_kA_tail_is_idempotent_not_reread_as_ka_name():
    # регрессия: уже свёрнутый триггером хвост "26кА" при повторном проходе ретокенизируется в
    # один смешанный по регистру токен "кА" - это НЕ то же самое, что имя буквы "ка" (нижний
    # регистр); без защиты по регистру он бы испортился в "26К" (T-003b)
    assert words_to_digits("26кА") == ("26кА", False)


def test_glued_trigger_prefix_with_numeral_and_letter():
    # слипание триггер+суффикс без пробела (T-003b): "кдва"->к+2, "кб"->к+Б
    assert words_to_digits("три кдва") == ("3к2", False)
    assert words_to_digits("семьдесят восемь кб") == ("78кБ", False)


def test_typo_dvadtsat_phonetic_alias():
    # "двадтсать" ("ц" услышана как "тс") - опечатка, не покрытая посимвольной сверткой
    assert words_to_digits("двадтсать пять") == ("25", False)


def test_corpus_abbreviated_letter_digit_glue_composite():
    # "корпус" + слипшийся код "буква+цифра" ("16 корпус п1" -> "16кП1", T-003b bucket 4)
    assert words_to_digits("шестнадцать корпус п1") == ("16кП1", False)
