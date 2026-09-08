"""Единая карта тегов: ничего не потеряно и ничего не додумано."""

import pandas as pd

from neftekod import config, parse_lims_pak, tag_map


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


def test_duplicate_descriptions_are_reported():
    """Коды с одинаковым описанием находятся, а не теряются в справочнике."""
    duplicates = tag_map.duplicate_descriptions(tag_map.build_tag_map())
    found = {
        (row.installation, tuple(row.codes)) for row in duplicates.itertuples()
    }
    assert found == {
        ("24-2000", ("F1", "T12")),
        ("24-2000", ("F17", "T23")),
        ("АВТ", ("F56", "F57")),
        ("АВТ", ("F68", "F69")),
    }


def test_duplicate_description_pairs_disagree_in_telemetry():
    """Одно описание — один поток. Три пары из четырёх этому не отвечают.

    Дефект — не дублирующееся описание само по себе (`F56`/`F57` при
    corr +0.997 действительно один поток с двух приборов), а расхождение
    рядов под общим описанием. См. ADR-007.
    """
    telemetry = {
        "24-2000": pd.read_csv(config.HT_TELEMETRY_CSV),
        "АВТ": pd.read_csv(config.AVT_TELEMETRY_CSV),
    }
    duplicates = tag_map.duplicate_descriptions(tag_map.build_tag_map())
    correlations = {}
    for row in duplicates.itertuples():
        frame = telemetry[row.installation]
        first, second = row.codes
        correlations[(row.installation, first, second)] = frame[first].corr(frame[second])

    agreeing = {key for key, value in correlations.items() if value > 0.95}
    assert agreeing == {("АВТ", "F56", "F57")}
    # Остальные три пары расходятся настолько, что общее описание не может
    # относиться к обоим тегам — вопрос к организаторам, ADR-007.
    for key in set(correlations) - agreeing:
        assert correlations[key] < 0.7


def test_la_sheet_has_no_machine_codes():
    """Лист «ЛА» — русские наименования анализов; кода показателя там нет."""
    frame = tag_map.build_tag_map()
    la = frame[frame["source_sheet"] == "ЛА"]
    assert la["code"].isna().all()
    assert la["sample_point"].notna().all()
    assert pd.notna(la["description"]).all()
