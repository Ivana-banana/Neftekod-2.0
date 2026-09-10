"""Синхронизация ЛИМС, ПАК и телеметрии по моменту выдачи прогноза.

Каждая строка результата строится as-of для одного issue_time. Для ЛИМС
и ПАК одновременно проверяются время измерения и время доступности. Для
телеметрии используется последнее конечное измерение не позже заданного
момента; пропуски не интерполируются, а их возраст сохраняется явно.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, splits

TARGET_TAG = "LIMS:Гидроочистка.2.Mg.Sulfur"
PAK_SULFUR_TAG = "24-2000:Mg.Sulfur"
TELEMETRY_TAGS = ("P8", "T11", "F19")


@dataclass(frozen=True)
class _IndexedSeries:
    frame: pd.DataFrame
    timestamp_ns: np.ndarray
    available_ns: np.ndarray | None
    value_column: str

    def latest(self, as_of: pd.Timestamp) -> tuple[float, pd.Timestamp, pd.Timestamp | None] | None:
        """Вернуть последнее конечное наблюдение, допустимое на as_of."""
        if self.frame.empty:
            return None
        position = int(np.searchsorted(self.timestamp_ns, as_of.value, side="right") - 1)
        while position >= 0:
            row = self.frame.iloc[position]
            if self.available_ns is None or self.available_ns[position] <= as_of.value:
                return (
                    float(row[self.value_column]),
                    row["timestamp"],
                    None if self.available_ns is None else row["available_at"],
                )
            position -= 1
        return None


def load_telemetry(
    path: Path | None = None,
    tags: tuple[str, ...] = TELEMETRY_TAGS,
) -> pd.DataFrame:
    """Загрузить только нужные колонки телеметрии 24-2000.

    В CSV временной ключ называется date. Большие исходные файлы не
    изменяются; чтение ограничено выбранными управляющими сигналами.
    """
    source = path or config.HT_TELEMETRY_CSV
    if not source.exists():
        raise FileNotFoundError(f"нет файла телеметрии: {source}")
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = set(tags) - set(header)
    if "date" not in header:
        raise ValueError(f"в телеметрии нет временного ключа date: {source}")
    if missing:
        raise ValueError(f"в телеметрии нет тегов: {sorted(missing)}")
    frame = pd.read_csv(source, usecols=["date", *tags])
    return _normalize_telemetry(frame, tags)


def _normalize_telemetry(frame: pd.DataFrame, tags: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    if "timestamp" not in result.columns:
        if "date" not in result.columns:
            raise ValueError("телеметрия должна содержать date или timestamp")
        result = result.rename(columns={"date": "timestamp"})
    missing = set(tags) - set(result.columns)
    if missing:
        raise ValueError(f"в телеметрии нет тегов: {sorted(missing)}")
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    if result["timestamp"].isna().any():
        raise ValueError("телеметрия содержит неизвестное время измерения")
    for tag in tags:
        values = pd.to_numeric(result[tag], errors="coerce")
        result[tag] = values.where(np.isfinite(values), np.nan)
    return result.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _quality_index(long_format: pd.DataFrame, tag: str) -> _IndexedSeries:
    part = long_format[long_format["tag"].eq(tag)].copy()
    part["timestamp"] = pd.to_datetime(part["timestamp"], errors="coerce")
    part["available_at"] = pd.to_datetime(part["available_at"], errors="coerce")
    part["value"] = pd.to_numeric(part["value"], errors="coerce")
    part = part.dropna(subset=["timestamp", "value"])
    part = part.loc[np.isfinite(part["value"].to_numpy(dtype=float))].copy()
    part = part.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if not part.empty and (part["available_at"] < part["timestamp"]).any():
        raise ValueError(f"для {tag} доступность раньше времени измерения")
    return _IndexedSeries(
        frame=part,
        timestamp_ns=part["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64"),
        available_ns=(
            part["available_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
            if not part.empty
            else np.array([], dtype=np.int64)
        ),
        value_column="value",
    )


def _telemetry_indexes(frame: pd.DataFrame, tags: tuple[str, ...]) -> dict[str, _IndexedSeries]:
    # Задержка публикации телеметрии не предоставлена; timestamp измерения
    # используется как доступность по явному допущению, см. ADR-009.
    indexes = {}
    for tag in tags:
        part = frame[["timestamp", tag]].copy()
        part[tag] = pd.to_numeric(part[tag], errors="coerce")
        part = part.loc[part[tag].notna() & np.isfinite(part[tag].to_numpy(dtype=float))]
        part = part.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        indexes[tag] = _IndexedSeries(
            frame=part,
            timestamp_ns=part["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64"),
            available_ns=None,
            value_column=tag,
        )
    return indexes


def _validate_issue_times(issue_times) -> list[pd.Timestamp]:
    values = pd.to_datetime(pd.Index(issue_times), errors="coerce")
    if values.isna().any():
        raise ValueError("неизвестное время выдачи прогноза")
    return list(values)


def _validate_quality_availability(long_format: pd.DataFrame) -> None:
    required = {"timestamp", "available_at", "source", "tag", "value"}
    missing = required - set(long_format.columns)
    if missing:
        raise ValueError(f"для синхронизации не хватает колонок: {sorted(missing)}")
    # Для ЛИМС/ПАК отсутствие available_at не трактуется как нулевая задержка.
    splits._times(long_format, "timestamp")


def _source_columns(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}_value",
        f"{prefix}_timestamp",
        f"{prefix}_available_at",
        f"{prefix}_age_hours",
        f"{prefix}_missing",
    )


def synchronize_features(
    issue_times,
    long_format: pd.DataFrame,
    telemetry: pd.DataFrame,
    *,
    telemetry_lag_hours: float = 0.0,
    target_tag: str = TARGET_TAG,
    pak_tag: str = PAK_SULFUR_TAG,
    telemetry_tags: tuple[str, ...] = TELEMETRY_TAGS,
) -> pd.DataFrame:
    """Сформировать доступные признаки для заданных моментов выдачи.

    telemetry_lag_hours задаёт момент чтения телеметрии как
    issue_time - lag. Возраст всё равно считается от issue_time — это
    возраст фактически использованного измерения на момент выдачи.
    """
    if telemetry_lag_hours < 0:
        raise ValueError("лаг телеметрии не может быть отрицательным")
    issues = _validate_issue_times(issue_times)
    _validate_quality_availability(long_format)
    telemetry_frame = _normalize_telemetry(telemetry, telemetry_tags)
    lims = _quality_index(long_format, target_tag)
    pak = _quality_index(long_format, pak_tag)
    telemetry_indexes = _telemetry_indexes(telemetry_frame, telemetry_tags)
    lag = pd.Timedelta(hours=float(telemetry_lag_hours))

    rows: list[dict] = []
    for issue_time in issues:
        row = {
            "issue_time": issue_time,
            "telemetry_reference_time": issue_time - lag,
        }
        for prefix, index in (("lims", lims), ("pak", pak)):
            observation = index.latest(issue_time)
            row[f"{prefix}_value"] = np.nan if observation is None else observation[0]
            row[f"{prefix}_timestamp"] = pd.NaT if observation is None else observation[1]
            row[f"{prefix}_available_at"] = (
                pd.NaT if observation is None else observation[2]
            )
            row[f"{prefix}_age_hours"] = (
                np.nan
                if observation is None
                else (issue_time - observation[1]).total_seconds() / 3600.0
            )
            row[f"{prefix}_missing"] = observation is None

        # Пропуски не заполняются интерполяцией; as-of оставляет возраст старого
        # измерения видимым для модели, см. ADR-009.
        reference_time = issue_time - lag
        for tag, index in telemetry_indexes.items():
            observation = index.latest(reference_time)
            row[f"{tag}_value"] = np.nan if observation is None else observation[0]
            row[f"{tag}_timestamp"] = pd.NaT if observation is None else observation[1]
            row[f"{tag}_age_hours"] = (
                np.nan
                if observation is None
                else (issue_time - observation[1]).total_seconds() / 3600.0
            )
            row[f"{tag}_missing"] = observation is None
        rows.append(row)

    columns = ["issue_time", "telemetry_reference_time"]
    columns += list(_source_columns("lims")) + list(_source_columns("pak"))
    for tag in telemetry_tags:
        columns += [
            f"{tag}_value",
            f"{tag}_timestamp",
            f"{tag}_age_hours",
            f"{tag}_missing",
        ]
    result = pd.DataFrame(rows, columns=columns)
    for column in ["lims_missing", "pak_missing", *(f"{tag}_missing" for tag in telemetry_tags)]:
        if column in result:
            result[column] = result[column].astype(object)
    return result
