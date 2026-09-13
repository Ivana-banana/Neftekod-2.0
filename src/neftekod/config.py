"""Пути и константы проекта. Единственное место, где зашиты имена файлов."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
BUILD_DIR = ROOT / "build"

AVT_TELEMETRY_CSV = DATA_DIR / "avt_tags.csv"
HT_TELEMETRY_CSV = DATA_DIR / "242000_tags.csv"

LIMS_XLSX = ROOT / "lims_export_2023.xlsx"
PAK_XLSX = ROOT / "pak_export_2023.xlsx"
TAGS_XLSX = ROOT / "docs/tz/tags_hackathon.xlsx"

LONG_FORMAT_CSV = BUILD_DIR / "lims_pak_long.csv"
TAG_MAP_CSV = BUILD_DIR / "tag_map.csv"
COVERAGE_CSV = BUILD_DIR / "coverage_report.csv"
COVERAGE_MD = BUILD_DIR / "coverage_report.md"
FORECAST_CSV = BUILD_DIR / "forecast_report.csv"
FORECAST_MD = BUILD_DIR / "forecast_report.md"

# Время ЛИМС — отбор пробы, публикация до 4 часов позже. При отсутствии
# фактического времени публикации используем консервативную верхнюю границу.
# См. ADR-006 и docs/tz/organizer-clarifications.md.
LIMS_PUBLICATION_DELAY_HOURS = 4

# Разбиение по времени — см. ADR-006. Границы зафиксированы до обучения
# моделей и не подбираются по результатам.
TRAIN_END = "2025-12-24 23:59:59"
TEST_START = "2026-01-01 00:00:00"
VALID_START = "2025-07-01 00:00:00"
EMBARGO_DAYS = 7
# Максимальное окно агрегирующего признака: определяет ширину зазора между
# train и test, иначе признак на первой точке теста заглянет в train.
FEATURE_MAX_WINDOW_DAYS = 7

# Жёсткое ограничение из ТЗ (§4): сера в товарном ДТ не более 10 мг/кг.
SULFUR_LIMIT_MG_KG = 10.0
# Доли компонентов блендинга суммируются в 100% (допуск на float-арифметику).
BLEND_FRACTION_SUM = 1.0
BLEND_FRACTION_TOL = 1e-6

# Обязательные показатели товарного ДТ — сера, T95 и цетановое число (ответы
# организаторов 2026-09-10, ADR-004). Предел серы задан организаторами;
# численные пределы T95 и цетанового числа они прямо оставили на усмотрение
# команды, поэтому это параметры сценария по ГОСТ 32511-2013 / ГОСТ Р 52368
# для К5, а не подтверждённый организаторами норматив. См. ADR-015.
T95_LIMIT_C = 360.0
CETANE_NUMBER_MIN = 51.0
# Пороги устаревания показателей спецификации. T95 отбирается той же суточной
# пробой, что и сера, поэтому порог тот же, что у ЛИМС в ADR-010. Цетановое
# число отбирается на порядок реже: p90 интервала между анализами на train —
# 41.8 суток, порог округлён до 42 суток по правилу ADR-013. См. ADR-015.
T95_MAX_AGE_HOURS = 48.0
CETANE_MAX_AGE_HOURS = 42 * 24.0
