# START_MODULE_CONTRACT
#   PURPOSE: Тесты normalizer.normalize: регистр/ё/латиница/пунктуация/сокращения/стоп-слова/
#            числительные/склейки, идемпотентность, NORMALIZER_VERSION.
#   SCOPE: tests/test_normalizer.py — только addrmatch.normalizer (+ addrmatch.numerals внутри).
#   DEPENDS: M-NORMALIZER
#   LINKS: V-M-NORMALIZER
# END_MODULE_CONTRACT
#
# START_MODULE_MAP
#   test_* - по одному сценарию из docs/concept.md §5.1 и plan.xml T-002 на тест
# END_MODULE_MAP
#
# START_CHANGE_SUMMARY
#   C-ADDRMATCH-PHASE-A T-002: тесты normalizer (>=8 кейсов + идемпотентность на 50 labeled)
# END_CHANGE_SUMMARY

"""Тесты normalizer.normalize (docs/concept.md §5.1)."""

import json
from pathlib import Path

from addrmatch.normalizer import NORMALIZER_VERSION, normalize

DATA_PATH = Path(__file__).resolve().parents[1] / "test_task_adress_match" / "data" / "adresses_labeled.jsonl"


def test_abbreviations_and_house_marker_removed():
    result = normalize("ул. красной армии, д. 10", channel="webchat")
    assert result.text == "улица красной армии 10"


def test_yo_to_ye():
    result = normalize("прописан шоссе рублёвское дом 70 к 6", channel="voice")
    assert "ё" not in result.text
    assert result.text == "прописан шоссе рублевское 70к6"


def test_latin_lookalikes_converted_for_webchat():
    # "o" латинское внутри кириллического слова "мoсковская" -> "о" кириллическое (webchat)
    result = normalize("улица мoсковская 5", channel="webchat")
    assert result.text == "улица московская 5"


def test_latin_lookalikes_not_touched_on_voice():
    # тот же вход, но voice: латиница-двойники не трогаются (правило только для webchat)
    result = normalize("улица мoсковская 5", channel="voice")
    assert "мoсковская" in result.text


def test_stopwords_removed():
    result = normalize("живу я на улица мира дом пять", channel="voice")
    assert result.text == "улица мира 5"


def test_stroenie_marker_survives_stopword_removal():
    # "стр." -> "с" (строение); "с" - ТАКЖЕ стоп-предлог ("рядом с домом") - маркер не терять
    result = normalize("проспект ленина стр. 5", channel="webchat")
    assert result.text == "проспект ленина с 5"
    result2 = normalize("рядом с домом улица мира 5", channel="voice")
    assert result2.text == "улица мира 5"


def test_digit_letter_glue_split_metallurgov():
    result = normalize("живу металлургов23", channel="voice")
    assert result.text == "металлургов 23"


def test_bare_digit_letter_joined_no_trigger():
    # "18 а" -> "18а": голая литера без триггера к/корпус/короче (плоская склейка normalizer'а,
    # в нижнем регистре - в отличие от 18кА через триггер в numerals)
    result = normalize("смоленской 1 а", channel="voice")
    assert result.text == "смоленской 1а"


def test_short_letter_suffix_not_split_house_letter():
    # "12а" (буквенная часть 1 символ < 3) - НЕ разделяем, это литера дома, а не слипание
    result = normalize("проспект ленина 12а", channel="voice")
    assert "12 а" not in result.text
    assert "12а" in result.text


def test_compound_street_dash_preserved():
    result = normalize("живу улица рабочи-крестьянская дом 30", channel="voice")
    assert result.text == "улица рабочи-крестьянская 30"


def test_truncated_flag_on_long_input():
    long_text = "улица ленина " * 60  # существенно больше 512 символов
    assert len(long_text) > 512
    result = normalize(long_text, channel="voice")
    assert result.truncated is True
    assert len(result.text) <= 512


def test_normalizer_version_is_stable_hex12():
    from addrmatch.normalizer import _build_version

    assert len(NORMALIZER_VERSION) == 12
    assert all(ch in "0123456789abcdef" for ch in NORMALIZER_VERSION)
    # пересчёт по тем же правилам даёт тот же хэш; вызовы normalize() не меняют версию
    assert _build_version() == NORMALIZER_VERSION
    normalize("тест", channel="voice")
    assert _build_version() == NORMALIZER_VERSION


def test_idempotent_on_50_real_labeled_strings():
    rows = []
    with DATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= 50:
                break

    for row in rows:
        channel = row.get("channel", "voice")
        first = normalize(row["raw_adress"], channel=channel)
        second = normalize(first.text, channel=channel)
        assert second.text == first.text, (row["raw_adress"], first.text, second.text)
