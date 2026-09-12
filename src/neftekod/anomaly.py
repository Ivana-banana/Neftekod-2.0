"""Контроль качества входа: явная проверка аномальности лабораторного значения.

До этого модуля аномалии отсекались побочно — порогами устаревания и
перцентильными полосами внутри агентов. Это не одно и то же: выброс ЛИМС
2120 мг/кг внутри обучающей выборки не был помечен ничем, он просто не
попадал в полосы. Здесь проверка называется своим именем и её результат
попадает в вывод агента, а не растворяется в других правилах.

Порог и правила — ADR-014. Модуль ничего не решает за агентов: он возвращает
флаги, а что с ними делать, написано в правилах агента оптимизации.
"""

from __future__ import annotations

import math

# Машиночитаемые типы аномалии: разбирать русскую строку подстрокой — не
# способ классификации ни для теста, ни для интерфейса оператора.
ANOMALY_CODES = ("nonphysical", "outside_training_band")


def _is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def assess_lab_value(value, stats: dict) -> dict:
    """Оценить лабораторное значение против обучающей выборки.

    `stats` — срез `training_stats` по этому показателю (нужны p01 и p99).
    Полоса p01–p99 обучающей выборки взята той же, по которой агент качества
    решает «вне диапазона обучения»: заводить второй порог для той же величины
    значило бы иметь два разных ответа на один вопрос.
    """
    low, high = float(stats["p01"]), float(stats["p99"])
    report = {
        "value": None if not _is_number(value) else float(value),
        "training_band": [low, high],
        "band_basis": "p01–p99 обучающей выборки по границе ADR-006 (см. ADR-014)",
        "flags": [],
        "anomalous": False,
    }
    if not _is_number(value):
        return report
    number = float(value)
    flags = []
    # Неположительная сера физически невозможна: это дефект измерения или
    # выгрузки, а не чистый продукт.
    if number <= 0:
        flags.append(
            {
                "code": "nonphysical",
                "detail": f"неположительное значение {number:g} мг/кг физически невозможно",
            }
        )
    elif number < low or number > high:
        flags.append(
            {
                "code": "outside_training_band",
                "detail": (
                    f"значение {number:g} вне полосы p01–p99 обучающей выборки "
                    f"[{low:g}, {high:g}]: как основание рекомендации не используется"
                ),
            }
        )
    report["flags"] = flags
    report["anomalous"] = bool(flags)
    return report
