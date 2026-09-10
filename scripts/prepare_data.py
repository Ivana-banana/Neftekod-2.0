#!/usr/bin/env python3
"""Подготовительный пайплайн: от исходных выгрузок до long-format и отчётов.

Одна команда от нуля до готового датасета:

    python scripts/prepare_data.py

Результат в `build/`:
    lims_pak_long.csv    — исходные значения, нормализованные единицы,
                          время отбора/измерения и доступности с основаниями
    tag_map.csv          — КИП+ЛА+ПАК+ВАК, исправления и оригиналы формул
    coverage_report.csv  — покрытие/актуальность по каждому показателю
    coverage_report.md   — то же плюс разрывы охвата и конфликты единиц
    unit_conflicts.csv   — расхождения единиц (ADR-005)

Скрипт детерминирован: одни и те же входные файлы дают побайтово те же
выходные. Ничего не скачивает и не меняет исходники.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neftekod import config, coverage, parse_lims_pak, tag_map, telemetry  # noqa: E402

log = logging.getLogger("prepare_data")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config.BUILD_DIR.mkdir(exist_ok=True)

    missing = [path for path in (config.LIMS_XLSX, config.PAK_XLSX, config.TAGS_XLSX) if not path.exists()]
    if missing:
        log.error("нет исходных файлов: %s", ", ".join(str(path) for path in missing))
        return 1

    log.info("1/4 разбор ЛИМС и ПАК")
    long_format = parse_lims_pak.build_long_format()
    parse_lims_pak.write_long_format(long_format, config.LONG_FORMAT_CSV)
    log.info("  ЛИМС: доступность = отбор + 4ч (консервативно); ПАК: время измерения (допущение)")
    log.info(
        "  %d значений, %d показателей → %s",
        len(long_format),
        long_format["tag"].nunique(),
        config.LONG_FORMAT_CSV.name,
    )

    log.info("2/4 карта тегов")
    tags = tag_map.attach_units_from_exports(tag_map.build_tag_map(), long_format)
    tags.to_csv(config.TAG_MAP_CSV, index=False)
    log.info(
        "  ВАК: %d поправок организаторов; %d формул заблокированы из-за Pipeline",
        tags["formula_source"].eq("organizer_clarifications_2026-09-10").sum(),
        tags["formula_status"].eq("blocked_pipeline").sum(),
    )
    without_unit = tag_map.tags_without_unit(tags)
    log.info(
        "  %d записей → %s; без подтверждённой единицы: %d",
        len(tags),
        config.TAG_MAP_CSV.name,
        len(without_unit),
    )

    log.info("3/4 отчёт покрытия и актуальности")
    report = coverage.build_coverage_report(long_format)
    conflicts = parse_lims_pak.unit_conflicts(long_format)
    report.to_csv(config.COVERAGE_CSV, index=False)
    conflicts.to_csv(config.BUILD_DIR / "unit_conflicts.csv", index=False)
    config.COVERAGE_MD.write_text(coverage.to_markdown(report, conflicts), encoding="utf-8")
    late = report[report["lag_from_dataset_start_days"] > 30]
    log.info(
        "  %d показателей → %s; охват начинается позже начала выгрузки у %d; "
        "расхождений исходных единиц после нормализации (ADR-005): %d",
        len(report),
        config.COVERAGE_MD.name,
        len(late),
        conflicts["indicator"].nunique(),
    )

    log.info("4/4 сверка колонок телеметрии со справочником «КИП»")
    for installation, path in telemetry.TELEMETRY_FILES.items():
        if not path.exists():
            log.warning("  %s: нет %s — скачать по ссылке из README", installation, path.name)
            continue
        diff = telemetry.compare_with_reference(installation)
        if any(diff.values()):
            log.error("  %s: расхождение со справочником: %s", installation, diff)
            return 1
        log.info("  %s: %d колонок совпадают со справочником", installation, len(telemetry.kip_codes(installation)))

    log.info("готово: %s", config.BUILD_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
