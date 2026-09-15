"""Дефекты телеметрии: маркер 307.0 и залипший сигнал.

Выгрузка не содержит NaN: неисправный датчик выглядит как обычное число.
Здесь такие значения находятся и превращаются в пропуск до синхронизации
признаков, чтобы as-of не выдавал показание мёртвого датчика за свежее
измерение. Правила и пороги — ADR-018.
"""

from __future__ import annotations

import pandas as pd

# Машиночитаемые коды дефекта, см. ADR-018.
DEFECT_CODES = ("sentinel", "frozen")
# Ровно 307.0 на физически разных тегах — маркер, а не измерение (ADR-007).
SENTINEL_VALUE = 307.0
# Серия из 6 одинаковых строк = 1 час при шаге 10 минут. У настоящих сигналов
# P8/T11/F19 на train серия длиннее 5 строк не встречается. См. ADR-018.
FROZEN_MIN_ROWS = 6


def _frozen(values: pd.Series) -> pd.Series:
    # Позиция строки внутри серии одинаковых значений, считая только прошлое:
    # помечается строка, у которой уже есть 5 таких же предшественников.
    run_id = values.ne(values.shift()).cumsum()
    position = values.groupby(run_id).cumcount() + 1
    return (position >= FROZEN_MIN_ROWS) & values.notna()


def defect_masks(frame: pd.DataFrame, tags: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Булевы маски {код дефекта: DataFrame по тегам}.

    Строки должны идти по времени без пропусков шага: залипание считается по
    подряд идущим строкам.
    """
    values = frame[list(tags)]
    return {
        "sentinel": values.eq(SENTINEL_VALUE),
        "frozen": values.apply(_frozen),
    }


def mask_defects(frame: pd.DataFrame, tags: tuple[str, ...]) -> pd.DataFrame:
    """Копия frame, в которой дефектные значения тегов заменены на NaN."""
    masks = defect_masks(frame, tags)
    bad = masks["sentinel"] | masks["frozen"]
    result = frame.copy()
    result[list(tags)] = result[list(tags)].mask(bad)
    return result

