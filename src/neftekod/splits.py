"""Разбиение по времени. Единственный допустимый способ делить эти данные.

Границы и обоснование — ADR-006. Здесь нет и не должно появиться варианта со
случайным перемешиванием: для 10-минутного ряда оно даёт утечку из будущего
(ТЗ, §6).
"""

from __future__ import annotations

import pandas as pd

from . import config

TRAIN_END = pd.Timestamp(config.TRAIN_END)
TEST_START = pd.Timestamp(config.TEST_START)
VALID_START = pd.Timestamp(config.VALID_START)
EMBARGO = pd.Timedelta(days=config.EMBARGO_DAYS)


def split(frame: pd.DataFrame, column: str = "timestamp") -> dict[str, pd.DataFrame]:
    """train / test по времени. Строки зазора не попадают никуда — см. ADR-006."""
    stamps = pd.to_datetime(frame[column])
    return {
        "train": frame[stamps <= TRAIN_END],
        "test": frame[stamps >= TEST_START],
    }


def train_valid_split(frame: pd.DataFrame, column: str = "timestamp") -> dict[str, pd.DataFrame]:
    """Внутренняя валидация внутри train — тест на подбор не смотрит."""
    train = split(frame, column)["train"]
    stamps = pd.to_datetime(train[column])
    return {
        "fit": train[stamps < VALID_START],
        "valid": train[stamps >= VALID_START],
    }


def _times(frame: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    """Времена события и доступности; неизвестное время нельзя считать прошлым."""
    stamps = pd.to_datetime(frame[column])
    if stamps.isna().any():
        raise ValueError("неизвестное время измерения/отбора")
    if "available_at" in frame:
        available = pd.to_datetime(frame["available_at"])
    elif "source" in frame and frame["source"].isin(["lims", "pak"]).any():
        raise ValueError("нет времени доступности available_at для ЛИМС/ПАК; пересоберите данные")
    else:
        # Для телеметрии без отдельного времени доступности используется
        # время измерения. Для ЛИМС/ПАК это допущение задаёт только парсер.
        available = stamps
    if available.isna().any() or (available < stamps).any():
        raise ValueError("время доступности неизвестно или раньше измерения/отбора")
    return stamps, available


def _cutoff(prediction_time) -> pd.Timestamp:
    cutoff = pd.Timestamp(prediction_time)
    if pd.isna(cutoff):
        raise ValueError("неизвестное время прогноза")
    return cutoff


def available_history(
    frame: pd.DataFrame, prediction_time, column: str = "timestamp"
) -> pd.DataFrame:
    """История, известная к моменту прогноза; без интерполяции и подглядывания.

    См. ADR-006: `timestamp` — время цели/отбора, `available_at` — допуск
    значения в признаки. На точной границе доступности значение уже известно.
    Отсутствие available_at у ЛИМС/ПАК — ошибка схемы, не нулевая задержка.
    """
    cutoff = _cutoff(prediction_time)
    stamps, available = _times(frame, column)
    return frame[(stamps <= cutoff) & (available <= cutoff)].copy()


def assert_no_leakage(frame: pd.DataFrame, prediction_time, column: str = "timestamp") -> None:
    """Ни одно значение в признаке не измерено и не опубликовано в будущем.

    Вызывается на входе построителя признаков: сплит по времени сам по себе
    не ловит утечку внутри признака (CLAUDE.md, «Тесты и доказательство
    результата»).
    """
    cutoff = _cutoff(prediction_time)
    stamps, available = _times(frame, column)
    future = frame[(stamps > cutoff) | (available > cutoff)]
    if len(future):
        raise ValueError(
            f"утечка из будущего: {len(future)} значений позже {prediction_time}, "
            f"максимум времени доступности {available.max()}"
        )
