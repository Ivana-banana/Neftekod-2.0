#!/usr/bin/env python3
"""Оценка baseline и телеметрической модели прогноза серы.

Запускать после scripts/prepare_data.py:

    python scripts/evaluate_forecast.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neftekod import config, features, forecast  # noqa: E402

log = logging.getLogger("evaluate_forecast")


def _markdown(report: pd.DataFrame) -> str:
    if report.empty:
        return "# Оценка прогноза серы\n\nНет доступных строк для оценки.\n"
    display = report.copy()
    for column in [
        "mae",
        "rmse",
        "r2",
        "lims_age_hours_median",
        "pak_age_hours_median",
        "P8_age_hours_median",
        "T11_age_hours_median",
        "F19_age_hours_median",
        "baseline_age_hours_median",
        "pak_missing_fraction",
        "P8_missing_fraction",
        "T11_missing_fraction",
        "F19_missing_fraction",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    return (
        "# Оценка прогноза серы\n\n"
        "Все строки имеют evidence_type: backtested: это проверка на истории, "
        "а не подтверждение эффекта управляющего действия.\n\n"
        "Лаг телеметрии и Ridge alpha выбираются на fit/valid; итоговая "
        "модель обучается на fit+valid и один раз проверяется на test.\n\n"
        + display.to_markdown(index=False)
        + "\n\n"
        "Возраст считается от момента выдачи прогноза: lims_age_hours_median "
        "и pak_age_hours_median — возраст доступных анализов, *_age_hours_median "
        "— возраст телеметрии. Все строки имеют evidence_type: backtested.\n"
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not config.LONG_FORMAT_CSV.exists():
        log.error("нет %s; сначала запустите scripts/prepare_data.py", config.LONG_FORMAT_CSV)
        return 1
    if not config.HT_TELEMETRY_CSV.exists():
        log.error(
            "нет %s; скачайте телеметрию 24-2000 по ссылке из README",
            config.HT_TELEMETRY_CSV,
        )
        return 1
    frame = pd.read_csv(
        config.LONG_FORMAT_CSV,
        parse_dates=["timestamp", "available_at", "first_seen"],
    )
    telemetry = features.load_telemetry(config.HT_TELEMETRY_CSV)
    log.info(
        "телеметрия 24-2000: %d строк, признаки %s; кандидаты лагов 0–3 ч",
        len(telemetry),
        ", ".join(features.TELEMETRY_TAGS),
    )
    report = forecast.evaluate_forecasts(
        frame,
        telemetry=telemetry,
        telemetry_lags=forecast.DEFAULT_TELEMETRY_LAGS,
    )
    config.BUILD_DIR.mkdir(exist_ok=True)
    report.to_csv(config.FORECAST_CSV, index=False, float_format="%.8f")
    config.FORECAST_MD.write_text(_markdown(report), encoding="utf-8")
    log.info("%d строк отчёта → %s и %s", len(report), config.FORECAST_CSV, config.FORECAST_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
