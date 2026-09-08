"""Тесты парсера на реальных файлах ЛИМС/ПАК, не на синтетике.

Синтетический пример не поймал бы главную ловушку формата: у каждого
показателя своя пара колонок с независимым охватом, и ошибка выравнивания
даёт правдоподобные, но чужие значения.
"""

import pandas as pd
import pytest

from neftekod import parse_lims_pak

# Заголовок LIMS заявляет «Количество значений:» — для двух показателей
# заявленное число на 1 больше, чем строк в самом файле. Проверено вручную
# по исходным ячейкам: лишней строки в файле нет, расхождение внесено
# экспортом. Фиксируем как факт данных, чтобы новое расхождение не утонуло.
KNOWN_COUNT_MISMATCH = {
    "LIMS:АВТ.1.CloudPoint": (14, 13),
    "LIMS:АВТ.1.CloudPoint_1": (113, 112),
}


def test_lims_series_are_split_by_indicator(lims_long):
    """54 показателя = 108 колонок исходника / 2 (пара «дата, значение»)."""
    assert lims_long["tag"].nunique() == 54
    assert set(lims_long["source"]) == {"lims"}
    assert set(lims_long["sample_point"]) == {
        "АВТ.1",
        "АВТ.2",
        "АВТ.2.1",
        "АВТ.3",
        "Гидроочистка.1",
        "Гидроочистка.2",
    }


def test_lims_counts_match_declared_header(lims_long):
    """Разобранное число точек сверяется с «Количество значений:» из файла."""
    declared = parse_lims_pak.lims_declared_counts()
    parsed = lims_long.groupby("tag").size().to_dict()
    assert set(declared) == set(parsed)

    mismatched = {
        tag: (declared[tag], parsed[tag]) for tag in declared if declared[tag] != parsed[tag]
    }
    assert mismatched == KNOWN_COUNT_MISMATCH


def test_lims_values_are_numeric_or_explicit_nan(lims_long):
    """'Pt Created' и подобные маркеры становятся NaN, а не 0 и не строкой."""
    assert pd.api.types.is_float_dtype(lims_long["value"])
    # Их мало и они локализованы — если доля вырастет, парсер молчать не должен.
    assert lims_long["value"].isna().sum() == 6


def test_pak_series_keep_independent_coverage(pak_long):
    """Ряды ПАК не выровнены по строкам: у D15 охват начинается на 2 года позже."""
    first_seen = pak_long.groupby("tag")["first_seen"].first()
    assert first_seen["24-2000:Mg.Sulfur"] == pd.Timestamp("2023-01-01 00:00:00")
    assert first_seen["24-2000:D15"] == pd.Timestamp("2025-03-05 10:20:00")

    d15 = pak_long[pak_long["tag"] == "24-2000:D15"]
    # Если бы пары колонок читались как одна таблица, у D15 появились бы
    # значения с датами серы (2023) — то есть чужой ряд под своим именем.
    assert d15["timestamp"].min() == first_seen["24-2000:D15"]
    assert d15["value"].between(700, 1000).all()


def test_pak_sulfur_tag_is_not_silently_renamed(pak_long):
    """ADR-001: имя тега берётся из файла, отождествление с `.Q` не делается."""
    tags = set(pak_long["tag"])
    assert parse_lims_pak.PAK_TAG_IN_FILE in tags
    assert parse_lims_pak.PAK_TAG_IN_REFERENCE_ONLY not in tags


def test_first_seen_is_per_tag_not_global(lims_long, pak_long):
    """`first_seen` — охват конкретного показателя, а не общая дата выгрузки."""
    frame = pd.concat([lims_long, pak_long], ignore_index=True)
    per_tag = frame.groupby("tag")["first_seen"].nunique()
    assert (per_tag == 1).all()
    assert frame["first_seen"].nunique() > 1
    assert (frame.groupby("tag")["timestamp"].min() == frame.groupby("tag")["first_seen"].first()).all()


def test_unit_conflicts_are_reported_not_fixed(lims_long):
    """Единицы блока «АВТ.1» не совпадают с остальными блоками — см. ADR-005.

    Автоматически «починить» их нельзя: какая строка заголовка сдвинута —
    вопрос к организаторам. Парсер обязан отдать расхождение наружу.
    """
    conflicts = parse_lims_pak.unit_conflicts(lims_long)
    assert set(conflicts["indicator"]) == {"50%.T", "D15", "EBP.T", "I350"}
    avt1 = conflicts[conflicts["tag"].str.startswith("LIMS:АВТ.1.")]
    assert dict(zip(avt1["indicator"], avt1["unit_declared"])) == {
        "50%.T": "кг/м3",
        "D15": "°С",
        "EBP.T": "% об.",
        "I350": "°С",
    }


@pytest.mark.parametrize("column", parse_lims_pak.LONG_COLUMNS)
def test_long_format_columns(lims_long, column):
    assert column in lims_long.columns
