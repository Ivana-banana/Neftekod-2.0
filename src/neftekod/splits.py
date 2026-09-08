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


def assert_no_leakage(frame: pd.DataFrame, prediction_time, column: str = "timestamp") -> None:
    """Ни одно значение, попавшее в признак, не относится к будущему.

    Вызывается на входе построителя признаков: сплит по времени сам по себе
    не ловит утечку внутри признака (CLAUDE.md, «Тесты и доказательство
    результата»).
    """
    stamps = pd.to_datetime(frame[column])
    future = frame[stamps > pd.Timestamp(prediction_time)]
    if len(future):
        raise ValueError(
            f"утечка из будущего: {len(future)} значений позже {prediction_time}, "
            f"максимум {stamps.max()}"
        )
