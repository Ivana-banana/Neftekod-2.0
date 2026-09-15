"""Вход сценария: снимок реальных данных на замороженный момент выдачи.

Контекст собирается один раз и раздаётся всем ролям, чтобы агенты смотрели
на одни и те же числа. Никакого «текущего времени» здесь нет: момент выдачи
задаётся явно, иначе прогон перестаёт быть воспроизводимым.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config, features, telemetry_defects

TRAINING_FEATURES = ("lims_value", "pak_value", "P8_value", "T11_value", "F19_value")


@dataclass(frozen=True, eq=False)
class ScenarioContext:
    scenario: str
    issue_time: pd.Timestamp
    window_start: pd.Timestamp
    snapshot: dict
    window_telemetry: pd.DataFrame
    window_lims: pd.DataFrame
    training_stats: dict
    backtest: dict
    long_format: pd.DataFrame
    telemetry: pd.DataFrame
    # Сколько значений маскировано как дефект на train и в окне, см. ADR-018.
    telemetry_defects: dict = field(default_factory=dict)


def _stats(values: pd.Series) -> dict:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
    if numeric.empty:
        raise ValueError("нет обучающих значений для расчёта диапазона")
    return {
        "n": int(len(numeric)),
        "min": float(numeric.min()),
        "p01": float(numeric.quantile(0.01)),
        "p05": float(numeric.quantile(0.05)),
        "p50": float(numeric.quantile(0.50)),
        "p95": float(numeric.quantile(0.95)),
        "p99": float(numeric.quantile(0.99)),
        "max": float(numeric.max()),
    }


def training_stats(long_format: pd.DataFrame, telemetry: pd.DataFrame) -> dict:
    """Диапазоны обучающей выборки — только train, до границы ADR-006.

    Возвращаются три разных среза, потому что на них висят разные вопросы:
    min/max — буквальный охват обучения (включая выбросы и остановы);
    p01–p99 — охват без выбросов, по нему решается «вне диапазона обучения»;
    p05–p95 — рутинная работа установки без хвостов и остановов, из неё
    агент надёжности строит консервативный модельный диапазон режима.
    Выбор трёх срезов и их назначение — см. ADR-010.
    """
    train_end = pd.Timestamp(config.TRAIN_END)
    quality = long_format[pd.to_datetime(long_format["timestamp"]) <= train_end]
    telemetry_train = telemetry[pd.to_datetime(telemetry["timestamp"]) <= train_end]
    stats = {
        "lims_value": _stats(quality.loc[quality["tag"].eq(features.TARGET_TAG), "value"]),
        "pak_value": _stats(quality.loc[quality["tag"].eq(features.PAK_SULFUR_TAG), "value"]),
    }
    for tag in features.TELEMETRY_TAGS:
        stats[f"{tag}_value"] = _stats(telemetry_train[tag])
    return stats


def load_backtest(path=None) -> dict:
    """Метрики backtest из отчёта, посчитанного scripts/evaluate_forecast.py.

    Используется только split=test: это единственная часть, которую модели не
    видели при подборе (ADR-006, ADR-008). Числа не пересчитываются здесь,
    чтобы сценарий не мог незаметно получить другую «доказанную» ошибку.
    """
    source = path or config.FORECAST_CSV
    if not source.exists():
        raise FileNotFoundError(f"нет отчёта backtest: {source}; запустите scripts/evaluate_forecast.py")
    report = pd.read_csv(source)
    test = report[report["split"].eq("test")]
    backtest: dict = {}
    for _, row in test.iterrows():
        horizon = float(row["horizon_hours"])
        backtest.setdefault(horizon, {})[str(row["model"])] = {
            "n": int(row["n"]),
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "r2": float(row["r2"]),
            "evidence_type": str(row["evidence_type"]),
            "medae": float(row["medae"]),
            # Параметры выбранного baseline: агент качества обязан уметь
            # воспроизвести прогноз, не переобучая ничего (ADR-012).
            "baseline_shrinkage": (
                None if pd.isna(row["baseline_shrinkage"]) else float(row["baseline_shrinkage"])
            ),
            "baseline_level_mg_kg": (
                None if pd.isna(row["baseline_level_mg_kg"]) else float(row["baseline_level_mg_kg"])
            ),
            # Гиперпараметры выбраны в evaluate_forecast на fit/valid; сценарий
            # их не переподбирает, а переиспользует (ADR-008, ADR-009).
            "alpha": None if pd.isna(row["alpha"]) else float(row["alpha"]),
            "telemetry_lag_hours": (
                None if pd.isna(row["telemetry_lag_hours"]) else float(row["telemetry_lag_hours"])
            ),
        }
    if not backtest:
        raise ValueError(f"в {source} нет строк split=test")
    return backtest


def _snapshot(issue_time: pd.Timestamp, long_format: pd.DataFrame, telemetry: pd.DataFrame) -> dict:
    row = features.synchronize_features([issue_time], long_format, telemetry).iloc[0]
    snapshot = {}
    for key, value in row.items():
        if isinstance(value, pd.Timestamp):
            snapshot[key] = value
        elif isinstance(value, (bool, np.bool_)):
            snapshot[key] = bool(value)
        elif value is pd.NaT:
            snapshot[key] = None
        else:
            snapshot[key] = float(value) if pd.notna(value) else None
    return snapshot


def load_context(
    scenario: str,
    issue_time,
    window_start,
    *,
    long_format: pd.DataFrame | None = None,
    telemetry: pd.DataFrame | None = None,
    backtest: dict | None = None,
) -> ScenarioContext:
    """Собрать вход сценария на замороженный момент выдачи."""
    issue = pd.Timestamp(issue_time)
    start = pd.Timestamp(window_start)
    if pd.isna(issue) or pd.isna(start):
        raise ValueError("момент выдачи и начало окна должны быть заданы")
    if start > issue:
        raise ValueError("окно сценария начинается позже момента выдачи")
    if long_format is None:
        long_format = pd.read_csv(
            config.LONG_FORMAT_CSV, parse_dates=["timestamp", "available_at", "first_seen"]
        )
    if telemetry is None:
        telemetry = features.load_raw_telemetry()
    raw_telemetry = features._parse_telemetry(telemetry, features.TELEMETRY_TAGS)
    telemetry = features._normalize_telemetry(telemetry, features.TELEMETRY_TAGS)

    stamps = pd.to_datetime(telemetry["timestamp"])
    in_window = (stamps >= start) & (stamps <= issue)
    window_telemetry = telemetry[in_window].copy()
    # Маски считаются по всему ряду и только потом режутся: залипание,
    # начавшееся до окна, должно быть видно и внутри окна.
    masks = telemetry_defects.defect_masks(raw_telemetry, features.TELEMETRY_TAGS)
    in_train = stamps <= pd.Timestamp(config.TRAIN_END)
    defects = {
        part: {
            tag: {code: int(masks[code][tag][rows].sum()) for code in telemetry_defects.DEFECT_CODES}
            for tag in features.TELEMETRY_TAGS
        }
        for part, rows in (("train", in_train), ("window", in_window))
    }
    quality_stamps = pd.to_datetime(long_format["timestamp"])
    available = pd.to_datetime(long_format["available_at"])
    # В окно попадают только опубликованные к моменту выдачи анализы: проба,
    # отобранная в окне, но ещё не выданная лабораторией, оператору неизвестна.
    window_lims = long_format[
        long_format["tag"].eq(features.TARGET_TAG)
        & (quality_stamps >= start)
        & (quality_stamps <= issue)
        & (available <= issue)
    ].copy()

    return ScenarioContext(
        scenario=scenario,
        issue_time=issue,
        window_start=start,
        snapshot=_snapshot(issue, long_format, telemetry),
        window_telemetry=window_telemetry,
        window_lims=window_lims,
        training_stats=training_stats(long_format, telemetry),
        backtest=backtest if backtest is not None else load_backtest(),
        long_format=long_format,
        telemetry=telemetry,
        telemetry_defects=defects,
    )
