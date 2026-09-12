"""Честная историческая оценка прогноза серы.

Модуль работает поверх long-format ЛИМС/ПАК. Для каждой лабораторной цели
строится момент выдачи прогноза, после чего признаки выбираются только среди
наблюдений, опубликованных к этому моменту. Метрики этого модуля имеют тип
backtested и не являются доказательством эффекта управляющего действия.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, features, splits

TARGET_TAG = "LIMS:Гидроочистка.2.Mg.Sulfur"
PAK_SULFUR_TAG = "24-2000:Mg.Sulfur"
FEATURE_NAMES = (
    "baseline_prediction",
    "pak_value",
    "baseline_age_hours",
    "pak_age_hours",
    "pak_missing",
)
TELEMETRY_FEATURE_NAMES = tuple(
    name
    for tag in features.TELEMETRY_TAGS
    for name in (f"{tag}_value", f"{tag}_age_hours", f"{tag}_missing")
)
DEFAULT_ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_TELEMETRY_LAGS = (0.0, 1.0, 2.0, 3.0)

# Семейство baseline: прогноз = shrinkage · последний анализ +
# (1 − shrinkage) · уровень обучающей выборки. shrinkage = 1 — это
# persistence, shrinkage = 0 — константа. Обе крайние точки входят в сетку,
# поэтому выбор между ними делается числом на valid, а не на словах.
# См. ADR-012.
BASELINE_SHRINKAGE_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
SHRUNK_BASELINE = "shrunk_persistence"


@dataclass(frozen=True)
class _IndexedObservations:
    frame: pd.DataFrame
    availability_ns: np.ndarray


def _validate_long_format(frame: pd.DataFrame) -> None:
    required = {"timestamp", "available_at", "source", "tag", "value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"для прогноза не хватает колонок: {sorted(missing)}")
    # См. ADR-008/ADR-006: ЛИМС/ПАК без корректной доступности нельзя считать
    # исторически известными.
    splits._times(frame, "timestamp")


def _indexed_observations(frame: pd.DataFrame, tag: str) -> _IndexedObservations:
    part = frame[frame["tag"].eq(tag)].copy()
    part["timestamp"] = pd.to_datetime(part["timestamp"])
    part["available_at"] = pd.to_datetime(part["available_at"])
    part["value"] = pd.to_numeric(part["value"], errors="coerce")
    part = part.dropna(subset=["timestamp", "available_at", "value"])
    finite = np.isfinite(part["value"].to_numpy(dtype=float))
    part = part.loc[finite].copy()
    if part.empty:
        return _IndexedObservations(part, np.array([], dtype=np.int64))
    if (part["available_at"] < part["timestamp"]).any():
        raise ValueError(f"для {tag} доступность раньше времени измерения")
    part = part.sort_values(["available_at", "timestamp"], kind="mergesort").reset_index(drop=True)
    availability_ns = part["available_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
    return _IndexedObservations(part, availability_ns)


def _latest(
    indexed: _IndexedObservations, as_of: pd.Timestamp
) -> tuple[float, pd.Timestamp, pd.Timestamp] | None:
    """Последнее числовое наблюдение, опубликованное к as_of."""
    if indexed.frame.empty:
        return None
    position = int(np.searchsorted(indexed.availability_ns, as_of.value, side="right") - 1)
    if position < 0:
        return None
    row = indexed.frame.iloc[position]
    return float(row["value"]), row["timestamp"], row["available_at"]


def build_prediction_frame(
    frame: pd.DataFrame,
    horizon_hours: float,
    *,
    target_tag: str = TARGET_TAG,
    pak_tag: str = PAK_SULFUR_TAG,
    telemetry: pd.DataFrame | None = None,
    telemetry_lag_hours: float = 0.0,
) -> pd.DataFrame:
    """Сформировать строки target / признаки as-of для заданного горизонта.

    Цель остаётся в моменте отбора. Например, для цели в 10:00 и горизонта
    1 час признаки берутся на 09:00. Анализ, отобранный в 00:00 и доступный в
    04:00, не используется для прогноза, выпущенного в 03:00.
    """
    if horizon_hours < 0:
        raise ValueError("горизонт прогноза не может быть отрицательным")
    _validate_long_format(frame)
    targets = _indexed_observations(frame, target_tag)
    pak = _indexed_observations(frame, pak_tag)
    rows: list[dict] = []
    horizon = pd.Timedelta(hours=float(horizon_hours))
    synchronized = None
    if telemetry is not None:
        issue_times = [target["timestamp"] - horizon for _, target in targets.frame.iterrows()]
        synchronized = features.synchronize_features(
            issue_times,
            frame,
            telemetry,
            telemetry_lag_hours=telemetry_lag_hours,
            target_tag=target_tag,
            pak_tag=pak_tag,
        )
    for target_position, (_, target) in enumerate(targets.frame.iterrows()):
        target_time = target["timestamp"]
        issue_time = target_time - horizon
        if synchronized is None:
            baseline = _latest(targets, issue_time)
            # См. ADR-008: без предыдущего доступного анализа persistence не определён.
            if baseline is None:
                continue
            pak_value = _latest(pak, issue_time)
            row = {
                "target_time": target_time,
                "issue_time": issue_time,
                "target_value": float(target["value"]),
                "target_available_at": target["available_at"],
                "baseline_prediction": baseline[0],
                "baseline_timestamp": baseline[1],
                "baseline_available_at": baseline[2],
                "baseline_age_hours": (issue_time - baseline[1]).total_seconds() / 3600.0,
                "pak_value": np.nan if pak_value is None else pak_value[0],
                "pak_timestamp": pd.NaT if pak_value is None else pak_value[1],
                "pak_available_at": pd.NaT if pak_value is None else pak_value[2],
                "pak_age_hours": np.nan
                if pak_value is None
                else (issue_time - pak_value[1]).total_seconds() / 3600.0,
                "pak_missing": pak_value is None,
            }
        else:
            synchronized_row = synchronized.iloc[target_position]
            # См. ADR-008: без предыдущего доступного анализа persistence не определён.
            if bool(synchronized_row["lims_missing"]):
                continue
            row = {
                "target_time": target_time,
                "issue_time": issue_time,
                "target_value": float(target["value"]),
                "target_available_at": target["available_at"],
                "baseline_prediction": synchronized_row["lims_value"],
                "baseline_timestamp": synchronized_row["lims_timestamp"],
                "baseline_available_at": synchronized_row["lims_available_at"],
                "baseline_age_hours": synchronized_row["lims_age_hours"],
                "pak_value": synchronized_row["pak_value"],
                "pak_timestamp": synchronized_row["pak_timestamp"],
                "pak_available_at": synchronized_row["pak_available_at"],
                "pak_age_hours": synchronized_row["pak_age_hours"],
                "pak_missing": synchronized_row["pak_missing"],
            }
            for tag in features.TELEMETRY_TAGS:
                for suffix in ("value", "timestamp", "age_hours", "missing"):
                    row[f"{tag}_{suffix}"] = synchronized_row[f"{tag}_{suffix}"]
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "target_time",
                "issue_time",
                "target_value",
                "target_available_at",
                *FEATURE_NAMES,
                *TELEMETRY_FEATURE_NAMES,
            ]
        )
    return result.sort_values("target_time", kind="mergesort").reset_index(drop=True)


def shrink_baseline(values, level: float, shrinkage: float):
    """Прогноз семейства baseline: сдвиг последнего анализа к уровню обучения."""
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage должен лежать в [0, 1]")
    return shrinkage * pd.to_numeric(values, errors="coerce") + (1.0 - shrinkage) * float(level)


def median_absolute_error(actual, predicted) -> float:
    """Устойчивая метрика отбора: в цели есть выбросы 120 и 2120 мг/кг."""
    y = np.asarray(pd.to_numeric(actual, errors="coerce"), dtype=float)
    y_hat = np.asarray(pd.to_numeric(predicted, errors="coerce"), dtype=float)
    mask = np.isfinite(y) & np.isfinite(y_hat)
    if not mask.any():
        raise ValueError("нет числовых пар для оценки")
    return float(np.median(np.abs(y_hat[mask] - y[mask])))


def select_shrinkage(
    fit: pd.DataFrame,
    valid: pd.DataFrame,
    grid: tuple[float, ...] = BASELINE_SHRINKAGE_GRID,
) -> float:
    """Выбрать shrinkage на valid по MedAE; при равенстве — большая доля анализа.

    Метрика отбора зафиксирована в ADR-012 до прогона test: MedAE, потому что
    RMSE на valid определяется единичными выбросами цели. Тай-брейк в пользу
    большей shrinkage тоже записан там: среди неразличимых вариантов система
    обязана остаться зависимой от свежего измерения, иначе она не может
    заметить выход за спецификацию.
    """
    if valid.empty:
        return grid[-1]
    level = float(fit["target_value"].median())
    scores = {
        shrinkage: median_absolute_error(
            valid["target_value"], shrink_baseline(valid["baseline_prediction"], level, shrinkage)
        )
        for shrinkage in grid
    }
    return min(scores, key=lambda shrinkage: (scores[shrinkage], -shrinkage))


def score_predictions(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    """MAE, RMSE и R² для пар числовых наблюдений."""
    y = np.asarray(pd.to_numeric(actual, errors="coerce"), dtype=float)
    y_hat = np.asarray(pd.to_numeric(predicted, errors="coerce"), dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError("actual и predicted должны иметь одинаковую длину")
    mask = np.isfinite(y) & np.isfinite(y_hat)
    if not mask.any():
        raise ValueError("нет числовых пар для оценки")
    y = y[mask]
    y_hat = y_hat[mask]
    residual = y_hat - y
    ss_tot = float(np.square(y - y.mean()).sum())
    r2 = float(1.0 - np.square(residual).sum() / ss_tot) if ss_tot else float("nan")
    return {
        "n": int(len(y)),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": r2,
    }


def _feature_names(frame: pd.DataFrame) -> tuple[str, ...]:
    if all(column in frame.columns for column in TELEMETRY_FEATURE_NAMES):
        return FEATURE_NAMES + TELEMETRY_FEATURE_NAMES
    return FEATURE_NAMES


def _feature_values(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
    imputation_values: np.ndarray | None = None,
) -> pd.DataFrame:
    values = frame.loc[:, feature_names].copy()
    values["pak_value"] = values["pak_value"].fillna(values["baseline_prediction"])
    values["pak_age_hours"] = values["pak_age_hours"].fillna(values["baseline_age_hours"])
    for column in feature_names:
        numeric = pd.to_numeric(values[column], errors="coerce")
        values[column] = numeric.where(np.isfinite(numeric), np.nan)
    values["pak_missing"] = values["pak_missing"].fillna(True).astype(float)
    if imputation_values is not None:
        fill_values = pd.Series(imputation_values, index=feature_names)
        values = values.fillna(fill_values)
    return values


def _features(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...] | None = None,
    imputation_values: np.ndarray | None = None,
) -> np.ndarray:
    names = feature_names or _feature_names(frame)
    values = _feature_values(frame, names, imputation_values)
    matrix = values.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("признаки содержат нечисловые значения")
    return matrix


@dataclass(frozen=True)
class RidgeModel:
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    imputation_values: np.ndarray | None = None


def fit_ridge(frame: pd.DataFrame, alpha: float = 1.0) -> RidgeModel:
    """Обучить малую ridge-модель; масштабирование вычисляется только на fit."""
    if alpha < 0:
        raise ValueError("alpha не может быть отрицательным")
    if len(frame) < 2:
        raise ValueError("для обучения нужно минимум две строки")
    # См. ADR-009: единицы/масштабы P8, T11 и F19 не подтверждены, поэтому
    # телеметрия используется как числовой сигнал и стандартизируется на fit.
    feature_names = _feature_names(frame)
    raw_values = _feature_values(frame, feature_names)
    imputation_values = raw_values.median(axis=0, skipna=True).to_numpy(
        dtype=float, copy=True
    )
    imputation_values[~np.isfinite(imputation_values)] = 0.0
    x = _features(frame, feature_names, imputation_values)
    y = frame["target_value"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("цель содержит нечисловые значения")
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0] = 1.0
    z = (x - means) / scales
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * float(np.sqrt(alpha))
    penalty[0, 0] = 0.0
    # Решение через расширенную матрицу не возводит число обусловленности в
    # квадрат, как нормальные уравнения; корректно работает и при collinear
    # признаках (см. ADR-008).
    augmented_design = np.vstack([design, penalty])
    augmented_target = np.concatenate([y, np.zeros(len(penalty))])
    coefficients = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)[0]
    return RidgeModel(means, scales, coefficients, feature_names, imputation_values)


def predict_ridge(model: RidgeModel, frame: pd.DataFrame) -> np.ndarray:
    x = (
        _features(frame, model.feature_names, model.imputation_values) - model.means
    ) / model.scales
    return np.column_stack([np.ones(len(x)), x]) @ model.coefficients


def _split_rows(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # См. ADR-006: границы train/valid/test фиксированы до обучения.
    target_time = pd.to_datetime(frame["target_time"])
    return {
        "fit": frame[target_time < pd.Timestamp(config.VALID_START)],
        "valid": frame[
            (target_time >= pd.Timestamp(config.VALID_START))
            & (target_time <= pd.Timestamp(config.TRAIN_END))
        ],
        "test": frame[target_time >= pd.Timestamp(config.TEST_START)],
    }


def _metric_row(
    *,
    horizon: float,
    split_name: str,
    model_name: str,
    rows: pd.DataFrame,
    predicted,
    alpha: float | None = None,
    telemetry_lag_hours: float | None = None,
    selected: bool | None = None,
    baseline_shrinkage: float | None = None,
    baseline_level_mg_kg: float | None = None,
) -> dict:
    metrics = score_predictions(rows["target_value"], predicted)
    telemetry_ages = {
        f"{tag}_age_hours_median": (
            float(rows[f"{tag}_age_hours"].median()) if f"{tag}_age_hours" in rows else np.nan
        )
        for tag in features.TELEMETRY_TAGS
    }
    telemetry_missing = {
        f"{tag}_missing_fraction": (
            float(rows[f"{tag}_missing"].mean()) if f"{tag}_missing" in rows else np.nan
        )
        for tag in features.TELEMETRY_TAGS
    }
    return {
        "horizon_hours": horizon,
        "split": split_name,
        "model": model_name,
        "alpha": alpha,
        "telemetry_lag_hours": telemetry_lag_hours,
        "selected": selected,
        "baseline_shrinkage": baseline_shrinkage,
        "baseline_level_mg_kg": baseline_level_mg_kg,
        "evidence_type": "backtested",
        **metrics,
        "medae": median_absolute_error(rows["target_value"], predicted),
        "lims_age_hours_median": float(rows["baseline_age_hours"].median()),
        "baseline_age_hours_median": float(rows["baseline_age_hours"].median()),
        "pak_age_hours_median": float(rows["pak_age_hours"].median()),
        "pak_missing_fraction": float(rows["pak_missing"].mean()),
        **telemetry_ages,
        **telemetry_missing,
    }


REPORT_COLUMNS = [
    "horizon_hours",
    "split",
    "model",
    "alpha",
    "telemetry_lag_hours",
    "selected",
    "baseline_shrinkage",
    "baseline_level_mg_kg",
    "evidence_type",
    "n",
    "mae",
    "rmse",
    "r2",
    "medae",
    "lims_age_hours_median",
    "baseline_age_hours_median",
    "pak_age_hours_median",
    "pak_missing_fraction",
    *(f"{tag}_age_hours_median" for tag in features.TELEMETRY_TAGS),
    *(f"{tag}_missing_fraction" for tag in features.TELEMETRY_TAGS),
]


def _shrunk_baseline_rows(horizon: float, parts: dict[str, pd.DataFrame]) -> list[dict]:
    """Строки семейства baseline: вся сетка на fit/valid, на test — выбранная.

    Уровень на test переоценивается на fit+valid — так же, как дообучается
    ridge (ADR-008). Сам shrinkage выбирается только на valid (ADR-012).
    """
    fit, valid, test = parts["fit"], parts["valid"], parts["test"]
    if fit.empty:
        return []
    level_fit = float(fit["target_value"].median())
    selected = select_shrinkage(fit, valid)
    rows: list[dict] = []
    for split_name in ("fit", "valid"):
        frame = parts[split_name]
        if frame.empty:
            continue
        for shrinkage in BASELINE_SHRINKAGE_GRID:
            rows.append(
                _metric_row(
                    horizon=horizon,
                    split_name=split_name,
                    model_name=SHRUNK_BASELINE,
                    rows=frame,
                    predicted=shrink_baseline(frame["baseline_prediction"], level_fit, shrinkage),
                    selected=shrinkage == selected,
                    baseline_shrinkage=shrinkage,
                    baseline_level_mg_kg=level_fit,
                )
            )
    if not test.empty:
        level_train = float(pd.concat([fit, valid], ignore_index=True)["target_value"].median())
        rows.append(
            _metric_row(
                horizon=horizon,
                split_name="test",
                model_name=SHRUNK_BASELINE,
                rows=test,
                predicted=shrink_baseline(test["baseline_prediction"], level_train, selected),
                selected=True,
                baseline_shrinkage=selected,
                baseline_level_mg_kg=level_train,
            )
        )
    return rows


def _evaluate_without_telemetry(
    frame: pd.DataFrame,
    horizons: tuple[float, ...] = (0, 1, 2, 3),
    *,
    target_tag: str = TARGET_TAG,
    pak_tag: str = PAK_SULFUR_TAG,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> list[dict]:
    report: list[dict] = []
    for horizon in horizons:
        supervised = build_prediction_frame(
            frame, horizon, target_tag=target_tag, pak_tag=pak_tag
        )
        parts = _split_rows(supervised)
        for split_name, rows in parts.items():
            if rows.empty:
                continue
            report.append(
                _metric_row(
                    horizon=horizon,
                    split_name=split_name,
                    model_name="persistence",
                    rows=rows,
                    predicted=rows["baseline_prediction"],
                )
            )
        report.extend(_shrunk_baseline_rows(horizon, parts))

        fit_rows = parts["fit"]
        if len(fit_rows) < 2:
            continue
        # См. ADR-008: alpha выбирается только на valid, test не участвует.
        fit_model_by_alpha = {alpha: fit_ridge(fit_rows, alpha) for alpha in alphas}
        if parts["valid"].empty:
            selected_alpha = alphas[0]
        else:
            scores = {
                alpha: score_predictions(
                    parts["valid"]["target_value"], predict_ridge(model, parts["valid"])
                )["rmse"]
                for alpha, model in fit_model_by_alpha.items()
            }
            selected_alpha = min(scores, key=lambda alpha: (scores[alpha], alpha))
        fit_model = fit_model_by_alpha[selected_alpha]
        for split_name in ("fit", "valid"):
            rows = parts[split_name]
            if rows.empty:
                continue
            report.append(
                _metric_row(
                    horizon=horizon,
                    split_name=split_name,
                    model_name="ridge",
                    rows=rows,
                    predicted=predict_ridge(fit_model, rows),
                    alpha=selected_alpha,
                )
            )
        if not parts["test"].empty:
            # См. ADR-008: после выбора alpha разрешено дообучение на fit+valid.
            final_train = pd.concat([parts["fit"], parts["valid"]], ignore_index=True)
            final_model = fit_ridge(final_train, selected_alpha)
            report.append(
                _metric_row(
                    horizon=horizon,
                    split_name="test",
                    model_name="ridge",
                    rows=parts["test"],
                    predicted=predict_ridge(final_model, parts["test"]),
                    alpha=selected_alpha,
                )
            )
    return report


def _normalise_lags(telemetry_lags) -> tuple[float, ...]:
    if not telemetry_lags:
        raise ValueError("список лагов телеметрии пуст")
    lags = tuple(float(lag) for lag in telemetry_lags)
    if any(not np.isfinite(lag) or lag < 0 for lag in lags):
        raise ValueError("лаг телеметрии должен быть конечным и неотрицательным")
    return tuple(sorted(set(lags)))


def _evaluate_with_telemetry(
    frame: pd.DataFrame,
    telemetry: pd.DataFrame,
    horizons: tuple[float, ...],
    target_tag: str,
    pak_tag: str,
    alphas: tuple[float, ...],
    telemetry_lags: tuple[float, ...],
) -> list[dict]:
    """Оценить телеметрию; test получает только lag, выбранный на valid."""
    report: list[dict] = []
    for horizon in horizons:
        by_lag = {
            lag: _split_rows(
                build_prediction_frame(
                    frame,
                    horizon,
                    target_tag=target_tag,
                    pak_tag=pak_tag,
                    telemetry=telemetry,
                    telemetry_lag_hours=lag,
                )
            )
            for lag in telemetry_lags
        }
        baseline_parts = by_lag[telemetry_lags[0]]
        for split_name, rows in baseline_parts.items():
            if rows.empty:
                continue
            report.append(
                _metric_row(
                    horizon=horizon,
                    split_name=split_name,
                    model_name="persistence",
                    rows=rows,
                    predicted=rows["baseline_prediction"],
                )
            )
        report.extend(_shrunk_baseline_rows(horizon, baseline_parts))

        candidates = []
        for lag in telemetry_lags:
            parts = by_lag[lag]
            fit_rows = parts["fit"]
            if len(fit_rows) < 2:
                continue
            models = {alpha: fit_ridge(fit_rows, alpha) for alpha in alphas}
            if parts["valid"].empty:
                selected_alpha = alphas[0]
                valid_rmse = float("inf")
            else:
                scores = {
                    alpha: score_predictions(
                        parts["valid"]["target_value"],
                        predict_ridge(model, parts["valid"]),
                    )["rmse"]
                    for alpha, model in models.items()
                }
                selected_alpha = min(scores, key=lambda alpha: (scores[alpha], alpha))
                valid_rmse = scores[selected_alpha]
            candidates.append((valid_rmse, lag, selected_alpha, models[selected_alpha]))

        if not candidates:
            continue
        # См. ADR-009: lag выбирается только по fit/valid, test здесь не участвует.
        _, selected_lag, selected_alpha, _ = min(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        for _, lag, alpha, fit_model in candidates:
            parts = by_lag[lag]
            for split_name in ("fit", "valid"):
                rows = parts[split_name]
                if rows.empty:
                    continue
                report.append(
                    _metric_row(
                        horizon=horizon,
                        split_name=split_name,
                        model_name="ridge_telemetry",
                        rows=rows,
                        predicted=predict_ridge(fit_model, rows),
                        alpha=alpha,
                        telemetry_lag_hours=lag,
                        selected=lag == selected_lag,
                    )
                )

        selected_parts = by_lag[selected_lag]
        if not selected_parts["test"].empty:
            final_train = pd.concat(
                [selected_parts["fit"], selected_parts["valid"]], ignore_index=True
            )
            final_model = fit_ridge(final_train, selected_alpha)
            report.append(
                _metric_row(
                    horizon=horizon,
                    split_name="test",
                    model_name="ridge_telemetry",
                    rows=selected_parts["test"],
                    predicted=predict_ridge(final_model, selected_parts["test"]),
                    alpha=selected_alpha,
                    telemetry_lag_hours=selected_lag,
                    selected=True,
                )
            )
    return report


def evaluate_forecasts(
    frame: pd.DataFrame,
    horizons: tuple[float, ...] = (0, 1, 2, 3),
    *,
    target_tag: str = TARGET_TAG,
    pak_tag: str = PAK_SULFUR_TAG,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    telemetry: pd.DataFrame | None = None,
    telemetry_lags: tuple[float, ...] = DEFAULT_TELEMETRY_LAGS,
) -> pd.DataFrame:
    """Оценить persistence и Ridge по fit/valid/test.

    Если передана телеметрия, lag и alpha выбираются на fit/valid, а test
    оценивает только выбранную комбинацию. Без телеметрии сохраняется
    существующий baseline-режим для обратной совместимости.
    """
    if not horizons:
        raise ValueError("нужно задать хотя бы один горизонт")
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise ValueError("список alpha пуст или содержит отрицательное значение")
    if telemetry is None:
        report = _evaluate_without_telemetry(
            frame,
            horizons,
            target_tag=target_tag,
            pak_tag=pak_tag,
            alphas=alphas,
        )
    else:
        report = _evaluate_with_telemetry(
            frame,
            telemetry,
            horizons,
            target_tag,
            pak_tag,
            alphas,
            _normalise_lags(telemetry_lags),
        )
    return pd.DataFrame(report, columns=REPORT_COLUMNS)
