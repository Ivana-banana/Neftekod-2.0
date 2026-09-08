"""Единая карта тегов: ничего не потеряно и ничего не додумано."""

import pandas as pd

from neftekod import parse_lims_pak, tag_map


def test_no_record_lost_from_source_sheets():
    """Записей в карте ровно столько, сколько непустых записей на листах.

    Пересчёт по листам сделан отдельной функцией «в лоб» — если сборщик
    пропустит блок (например, второй справочник на листе «КИП»), числа
    разойдутся.
    """
    frame = tag_map.build_tag_map()
    counts = tag_map.sheet_record_counts()
    assert counts == {"КИП": 97, "ЛА": 56, "ПАК": 2, "ВАК": 17}
    assert len(frame) == sum(counts.values())
    assert frame["source_sheet"].value_counts().to_dict() == counts


def test_kip_codes_are_qualified_by_installation():
    """`T6` есть и на АВТ, и на 24-2000 — короткий код не может быть ключом."""
    frame = tag_map.build_tag_map()
    kip = frame[frame["source_sheet"] == "КИП"]
    assert kip["code"].is_unique
    assert "АВТ:T6" in set(kip["code"])
    assert "24-2000:T6" in set(kip["code"])


def test_tags_without_unit_are_flagged_not_dropped():
    """Тег без подтверждённой единицы остаётся в карте с пустым `unit`."""
    frame = tag_map.attach_units_from_exports(
        tag_map.build_tag_map(), parse_lims_pak.parse_pak()
    )
    without_unit = tag_map.tags_without_unit(frame)
    # Единицы в пакете объявлены только в самих выгрузках ЛИМС/ПАК; из
    # справочника единицу получает только `24-2000:D15` (у тега серы имя в
    # справочнике другое — ADR-001, отождествление не делается).
    assert set(frame[frame["unit"].notna()]["code"]) == {"24-2000:D15"}
    assert len(without_unit) == len(frame) - 1
    assert set(without_unit["source_sheet"]) == {"КИП", "ЛА", "ПАК", "ВАК"}


def test_blocked_formulas_carry_adr_reference():
    """Формулы ВАК помечены ADR-002/003, а не выглядят готовыми к применению."""
    frame = tag_map.build_tag_map()
    vak = frame[frame["source_sheet"] == "ВАК"]
    assert vak["notes"].str.contains("ADR-00").all()
    pipeline = vak[vak["description"].str.contains("Pipeline")]
    assert set(pipeline["code"]) == {"24-2000:GODT:D15", "24-2000:GODT:T95"}
    assert pipeline["notes"].str.contains("ADR-003").all()


def test_la_sheet_has_no_machine_codes():
    """Лист «ЛА» — русские наименования анализов; кода показателя там нет."""
    frame = tag_map.build_tag_map()
    la = frame[frame["source_sheet"] == "ЛА"]
    assert la["code"].isna().all()
    assert la["sample_point"].notna().all()
    assert pd.notna(la["description"]).all()
