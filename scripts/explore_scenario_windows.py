#!/usr/bin/env python3
"""Разведка окон сценариев 2, 3 и 4 по свойствам данных.

Скрипт не выбирает модель и не трогает пороги: он печатает те числа, по
которым окна зафиксированы в ADR-011, чтобы критерий выбора можно было
перепроверить, а не принять на слово. Запускать после scripts/prepare_data.py
и scripts/evaluate_forecast.py:

    python scripts/explore_scenario_windows.py

Правила отбора (см. ADR-011) одинаковы для всех сценариев:

- момент выдачи — время публикации анализа ЛИМС (`available_at`): раньше
  оператор нового значения не видит;
- окно — 7 суток до момента выдачи, как в сценарии 1;
- «внутри полосы» — значение цели лежит в p05–p95 обучающей выборки;
  значения вне полосы отнесены к аномальным и не используются для сценариев
  2 и 4;
- «режим допустим» — вывод агента надёжности на этом окне.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neftekod import config, features  # noqa: E402
from neftekod.agents import context as context_module, quality, reliability  # noqa: E402

WINDOW_DAYS = 7
RISE_HOURS = 72

log = logging.getLogger("explore_scenario_windows")


def scan_test_period(long_format: pd.DataFrame, telemetry: pd.DataFrame) -> pd.DataFrame:
    """Выводы агентов качества и надёжности на каждый момент публикации ЛИМС."""
    # Диапазоны обучения и backtest считаются один раз: они не зависят от
    # момента выдачи, а пересчёт на каждый момент занимает минуты.
    stats = context_module.training_stats(long_format, telemetry)
    backtest = context_module.load_backtest()
    telemetry_end = pd.to_datetime(telemetry["timestamp"]).max()

    target = long_format[long_format["tag"].eq(features.TARGET_TAG)].copy()
    target["value"] = pd.to_numeric(target["value"], errors="coerce")
    target = target.dropna(subset=["value"])
    test = target[
        (pd.to_datetime(target["timestamp"]) >= pd.Timestamp(config.TEST_START))
        # Публикация позже конца телеметрии означает молчащий сигнал, а не режим.
        & (pd.to_datetime(target["available_at"]) <= telemetry_end)
    ].sort_values("available_at")

    rows = []
    for _, row in test.iterrows():
        issue = pd.Timestamp(row["available_at"])
        scenario_context = context_module.ScenarioContext(
            scenario="explore",
            issue_time=issue,
            window_start=issue - pd.Timedelta(days=WINDOW_DAYS),
            snapshot=context_module._snapshot(issue, long_format, telemetry),
            window_telemetry=telemetry[
                (pd.to_datetime(telemetry["timestamp"]) >= issue - pd.Timedelta(days=WINDOW_DAYS))
                & (pd.to_datetime(telemetry["timestamp"]) <= issue)
            ].copy(),
            window_lims=test[pd.to_datetime(test["available_at"]) <= issue],
            training_stats=stats,
            backtest=backtest,
            long_format=long_format,
            telemetry=telemetry,
        )
        quality_assessment = quality.assess(scenario_context)
        reliability_assessment = reliability.assess(scenario_context)
        forecast = quality_assessment.forecast[0] if quality_assessment.forecast else None
        rows.append(
            {
                "issue_time": issue,
                "sample_time": pd.Timestamp(row["timestamp"]),
                "sulfur_mg_kg": float(row["value"]),
                "quality_refusal": quality_assessment.refusal is not None,
                "risk_exceed_spec": None if forecast is None else forecast["risk_exceed_spec"],
                "margin_mg_kg": None if forecast is None else forecast["margin_mg_kg"],
                "severity_index": reliability_assessment.severity_index,
                "regime_admissible": reliability_assessment.regime_admissible,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for path in (config.LONG_FORMAT_CSV, config.FORECAST_CSV):
        if not path.exists():
            log.error("нет %s; сначала запустите scripts/prepare_data.py и scripts/evaluate_forecast.py", path)
            return 1
    long_format = pd.read_csv(
        config.LONG_FORMAT_CSV, parse_dates=["timestamp", "available_at", "first_seen"]
    )
    telemetry = features.load_telemetry()
    stats = context_module.training_stats(long_format, telemetry)["lims_value"]
    low, high = stats["p05"], stats["p95"]
    print(f"полоса p05–p95 обучающей выборки по цели: {low:.2f} … {high:.2f} мг/кг")

    scan = scan_test_period(long_format, telemetry)
    print(
        f"моментов публикации в test с живой телеметрией: {len(scan)}; "
        f"режим допустим: {int(scan['regime_admissible'].sum())}; "
        f"отказов агента качества: {int(scan['quality_refusal'].sum())}"
    )

    usable = scan[scan["regime_admissible"] & ~scan["quality_refusal"]]
    in_band = usable[usable["sulfur_mg_kg"].between(low, high)]

    print(f"\n=== Сценарий 4: наибольший запас по сере внутри полосы (top-5 из {len(in_band)})")
    print(
        in_band.nsmallest(5, "sulfur_mg_kg")[
            ["issue_time", "sulfur_mg_kg", "margin_mg_kg", "risk_exceed_spec", "severity_index"]
        ].to_string(index=False)
    )

    series = scan.set_index("issue_time")["sulfur_mg_kg"]
    rises = []
    for issue, value in series.items():
        previous = series[(series.index >= issue - pd.Timedelta(hours=RISE_HOURS)) & (series.index < issue)]
        if not previous.empty:
            rises.append({"issue_time": issue, "rise_mg_kg": value - previous.min()})
    rise_frame = in_band.merge(pd.DataFrame(rises), on="issue_time")
    print(f"\n=== Сценарий 2: наибольший рост серы за {RISE_HOURS} ч внутри полосы (top-5)")
    print(
        rise_frame.nlargest(5, "rise_mg_kg")[
            ["issue_time", "sulfur_mg_kg", "rise_mg_kg", "risk_exceed_spec", "severity_index"]
        ].to_string(index=False)
    )

    published = scan["issue_time"].reset_index(drop=True)
    gaps = pd.DataFrame(
        {
            "previous": published.shift(1),
            "next": published,
            "gap_hours": published.diff().dt.total_seconds() / 3600.0,
        }
    ).dropna()
    print("\n=== Сценарий 3: крупнейшие разрывы публикации ЛИМС в test (top-5)")
    print(gaps.nlargest(5, "gap_hours").to_string(index=False))
    widest = gaps.nlargest(1, "gap_hours").iloc[0]
    middle = widest["previous"] + (widest["next"] - widest["previous"]) / 2
    print(
        f"середина крупнейшего разрыва: {middle}; возраст последнего анализа там "
        f"{(middle - widest['previous']).total_seconds() / 3600.0:.1f} ч"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
