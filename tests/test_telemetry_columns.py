"""Колонки телеметрии = коды листа «КИП».

Совпадение уже проверялось вручную при разборе пакета. Тест нужен, чтобы
расхождение не появилось тихо: если организаторы обновят выгрузку или
справочник, смысл тега `T6` поменяется молча — и агент качества будет
читать не тот столбец, оставаясь «зелёным» по всем остальным проверкам.
"""

import pytest

from neftekod import telemetry


@pytest.mark.parametrize(
    ("installation", "expected_count"),
    [("АВТ", 71), ("24-2000", 26)],
)
def test_telemetry_columns_match_kip_reference(installation, expected_count):
    path = telemetry.TELEMETRY_FILES[installation]
    if not path.exists():
        pytest.skip(f"нет {path} — скачать по ссылке из README")

    columns = telemetry.telemetry_columns(path)
    codes = telemetry.kip_codes(installation)
    assert len(codes) == expected_count
    assert len(columns) == expected_count
    # Порядок колонок в CSV совпадает с порядком строк справочника —
    # проверяем именно список, а не множество.
    assert columns == codes


@pytest.mark.parametrize("installation", ["АВТ", "24-2000"])
def test_no_unexplained_columns(installation):
    if not telemetry.TELEMETRY_FILES[installation].exists():
        pytest.skip("нет файла телеметрии — скачать по ссылке из README")
    assert telemetry.compare_with_reference(installation) == {
        "only_in_telemetry": [],
        "only_in_reference": [],
    }
