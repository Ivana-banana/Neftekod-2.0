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

from . import config, splits

TARGET_TAG = "LIMS:Гидроочистка.2.Mg.Sulfur"
PAK_SULFUR_TAG = "24-2000:Mg.Sulfur"
FEATURE_NAMES = (
    "baseline_prediction",
    "pak_value",
    "baseline_age_hours",
    "pak_age_hours",
    "pak_missing",
)
DEFAULT_ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)


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
    for _, target in targets.frame.iterrows():
        target_time = target["timestamp"]
        issue_time = target_time - horizon
        baseline = _latest(targets, issue_time)
        # См. ADR-008: без предыдущего доступного анализа persistence не определён.
        if baseline is None:
            continue
        pak_value = _latest(pak, issue_time)
        rows.append(
            {
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
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "target_time",
                "issue_time",
                "target_value",
                "target_available_at",
                *FEATURE_NAMES,
            ]
        )
    return result.sort_values("target_time", kind="mergesort").reset_index(drop=True)


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


def _features(frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[:, FEATURE_NAMES].copy()
    values["pak_value"] = values["pak_value"].fillna(values["baseline_prediction"])
    values["pak_age_hours"] = values["pak_age_hours"].fillna(values["baseline_age_hours"])
    values["pak_missing"] = values["pak_missing"].astype(float)
    matrix = values.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("признаки содержат нечисловые значения")
    return matrix


@dataclass(frozen=True)
class RidgeModel:
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray


def fit_ridge(frame: pd.DataFrame, alpha: float = 1.0) -> RidgeModel:
    """Обучить малую ridge-модель; масштабирование вычисляется только на fit."""
    if alpha < 0:
        raise ValueError("alpha не может быть отрицательным")
    if len(frame) < 2:
        raise ValueError("для обучения нужно минимум две строки")
    x = _features(frame)
    y = frame["target_value"].to_numpy(dtype=float)
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
    return RidgeModel(means, scales, coefficients)


def predict_ridge(model: RidgeModel, frame: pd.DataFrame) -> np.ndarray:
    x = (_features(frame) - model.means) / model.scales
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
) -> dict:
    metrics = score_predictions(rows["target_value"], predicted)
    return {
        "horizon_hours": horizon,
        "split": split_name,
        "model": model_name,
        "alpha": alpha,
        "evidence_type": "backtested",
        **metrics,
        "baseline_age_hours_median": float(rows["baseline_age_hours"].median()),
        "pak_missing_fraction": float(rows["pak_missing"].mean()),
    }


def evaluate_forecasts(
    frame: pd.DataFrame,
    horizons: tuple[float, ...] = (0, 1, 2, 3),
    *,
    target_tag: str = TARGET_TAG,
    pak_tag: str = PAK_SULFUR_TAG,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> pd.DataFrame:
    """Оценить persistence и Ridge на fit/valid/test по каждому горизонту."""
    if not horizons:
        raise ValueError("нужно задать хотя бы один горизонт")
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise ValueError("список alpha пуст или содержит отрицательное значение")
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
    columns = [
        "horizon_hours",
        "split",
        "model",
        "alpha",
        "evidence_type",
        "n",
        "mae",
        "rmse",
        "r2",
        "baseline_age_hours_median",
        "pak_missing_fraction",
    ]
    return pd.DataFrame(report, columns=columns)
