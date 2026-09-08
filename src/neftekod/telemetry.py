"""Телеметрия АВТ / 24-2000: чтение заголовка и сверка со справочником «КИП».

Файлы большие (247 МБ и 91 МБ), поэтому сверка колонок читает только первую
строку. Полная загрузка рядов появится вместе с фичами — здесь её нет
намеренно.
"""

from __future__ import annotations

import csv
from pathlib import Path

from . import config, tag_map

# Служебные колонки экспорта: индекс строки и временной ключ (ТЗ, §2).
SERVICE_COLUMNS = {"date"}
SERVICE_PREFIX = "Unnamed:"

TELEMETRY_FILES = {
    "АВТ": config.AVT_TELEMETRY_CSV,
    "24-2000": config.HT_TELEMETRY_CSV,
}


def telemetry_columns(path: Path) -> list[str]:
    """Коды тегов из заголовка CSV, без служебных колонок."""
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    return [
        column
        for column in header
        if column and column not in SERVICE_COLUMNS and not column.startswith(SERVICE_PREFIX)
    ]


def kip_codes(installation: str) -> list[str]:
    """Короткие коды листа «КИП» для установки — в порядке справочника."""
    frame = tag_map.build_tag_map()
    kip = frame[(frame["source_sheet"] == "КИП") & (frame["installation"] == installation)]
    return [code.split(":", 1)[1] for code in kip["code"]]


def compare_with_reference(installation: str) -> dict[str, list[str]]:
    """Расхождения между колонками телеметрии и справочником «КИП»."""
    columns = telemetry_columns(TELEMETRY_FILES[installation])
    codes = kip_codes(installation)
    return {
        "only_in_telemetry": sorted(set(columns) - set(codes)),
        "only_in_reference": sorted(set(codes) - set(columns)),
    }
