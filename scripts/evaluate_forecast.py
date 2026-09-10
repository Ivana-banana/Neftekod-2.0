#!/usr/bin/env python3
"""Оценка baseline и простой обучаемой модели прогноза серы.

Запускать после scripts/prepare_data.py:

    python scripts/evaluate_forecast.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neftekod import config, forecast  # noqa: E402

log = logging.getLogger("evaluate_forecast")


def _markdown(report: pd.DataFrame) -> str:
    if report.empty:
        return "# Оценка прогноза серы\n\nНет доступных строк для оценки.\n"
    display = report.copy()
    for column in ["mae", "rmse", "r2", "baseline_age_hours_median", "pak_missing_fraction"]:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    return (
        "# Оценка прогноза серы\n\n"
        "Все строки имеют evidence_type: backtested: это проверка на истории, "
        "а не подтверждение эффекта управляющего действия.\n\n"
        "Выбор Ridge alpha выполняется на valid; итоговая модель обучается на "
        "fit+valid и один раз проверяется на test.\n\n"
        + display.to_markdown(index=False)
        + "\n\n"
        "baseline_age_hours_median — возраст последнего доступного анализа от "
        "момента выдачи прогноза; pak_missing_fraction — доля строк без ПАК.\n"
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not config.LONG_FORMAT_CSV.exists():
        log.error("нет %s; сначала запустите scripts/prepare_data.py", config.LONG_FORMAT_CSV)
        return 1
    frame = pd.read_csv(
        config.LONG_FORMAT_CSV,
        parse_dates=["timestamp", "available_at", "first_seen"],
    )
    report = forecast.evaluate_forecasts(frame)
    config.BUILD_DIR.mkdir(exist_ok=True)
    report.to_csv(config.FORECAST_CSV, index=False, float_format="%.8f")
    config.FORECAST_MD.write_text(_markdown(report), encoding="utf-8")
    log.info("%d строк отчёта → %s и %s", len(report), config.FORECAST_CSV, config.FORECAST_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
