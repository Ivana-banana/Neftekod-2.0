"""Детерминированная запись выводов агентов.

Побайтовая воспроизводимость прогона — требование демонстрации, а не
эстетика: без неё «тот же сценарий» на сцене нельзя отличить от другого.
Поэтому здесь нет времени запуска, случайных ключей и свободного
форматирования float.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

FLOAT_DIGITS = 6


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return round(number, FLOAT_DIGITS)
    return value


def to_json_text(payload) -> str:
    return json.dumps(_plain(payload), ensure_ascii=False, indent=2) + "\n"


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json_text(payload), encoding="utf-8")
    return path
