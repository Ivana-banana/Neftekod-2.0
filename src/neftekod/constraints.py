"""Жёсткие ограничения. Проверяются всегда, вне зависимости от метрик.

Это не «валидация входа» и не рекомендательная эвристика: нарушение любого
из этих правил означает, что рекомендация не выдаётся (ТЗ, §4). Улучшение
выпуска, энергозатрат или любой другой метрики не является основанием
ослабить проверку.

Модуль намеренно не знает ничего о моделях и агентах: агент оптимизации и
оркестратор вызывают `validate_recommendation()` и получают список
нарушений. Пустой список — единственный способ считать рекомендацию
допустимой.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import config

# CLAUDE.md, «Верификация и честность»: каждый вывод агента оптимизации
# обязан различать проверенный историей прогноз и модельную оценку эффекта
# ещё не случавшегося действия.
EVIDENCE_TYPES = frozenset({"backtested", "simulated_unvalidated"})


@dataclass(frozen=True)
class Violation:
    rule: str
    detail: str


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(value)


def check_sulfur(value_mg_kg) -> list[Violation]:
    """Сера в товарном ДТ ≤ 10 мг/кг (ТЗ, §4).

    Отсутствующее или нечисловое значение — тоже нарушение: при нехватке
    данных система обязана отказаться от рекомендации, а не считать
    неизвестное допустимым.
    """
    if not _is_number(value_mg_kg):
        return [Violation("sulfur", f"нет числового значения серы: {value_mg_kg!r}")]
    if value_mg_kg > config.SULFUR_LIMIT_MG_KG:
        return [
            Violation(
                "sulfur",
                f"сера {value_mg_kg} мг/кг > предела {config.SULFUR_LIMIT_MG_KG} мг/кг",
            )
        ]
    return []


def check_blend_fractions(fractions) -> list[Violation]:
    """Доли компонентов блендинга суммируются в 100% и неотрицательны."""
    if not fractions:
        return [Violation("blend_fractions", "состав блендинга не задан")]
    values = list(fractions.values()) if isinstance(fractions, dict) else list(fractions)
    if not all(_is_number(value) for value in values):
        return [Violation("blend_fractions", f"нечисловые доли: {values!r}")]
    violations = []
    negative = [value for value in values if value < 0]
    if negative:
        violations.append(Violation("blend_fractions", f"отрицательные доли: {negative}"))
    total = math.fsum(values)
    if abs(total - config.BLEND_FRACTION_SUM) > config.BLEND_FRACTION_TOL:
        violations.append(
            Violation("blend_fractions", f"сумма долей {total!r} ≠ {config.BLEND_FRACTION_SUM}")
        )
    return violations


def check_within_ranges(setpoints, allowed_ranges) -> list[Violation]:
    """Рекомендация не выходит за явно заданные диапазоны.

    Диапазон должен быть задан для каждого изменяемого параметра. Параметр
    без диапазона — нарушение, а не «разрешено по умолчанию»: исторический
    min/max не является промышленным пределом (ТЗ, §4).
    """
    violations = []
    for name, value in (setpoints or {}).items():
        bounds = (allowed_ranges or {}).get(name)
        if bounds is None:
            violations.append(Violation("range", f"для параметра {name} не задан диапазон"))
            continue
        low, high = bounds
        if not _is_number(value):
            violations.append(Violation("range", f"нечисловая уставка {name}: {value!r}"))
        elif not low <= value <= high:
            violations.append(Violation("range", f"{name}={value} вне диапазона [{low}, {high}]"))
    return violations


def check_evidence_type(evidence_type) -> list[Violation]:
    """Поле `evidence_type` обязательно и имеет одно из двух значений."""
    if evidence_type not in EVIDENCE_TYPES:
        return [
            Violation(
                "evidence_type",
                f"ожидалось одно из {sorted(EVIDENCE_TYPES)}, получено {evidence_type!r}",
            )
        ]
    return []


def validate_recommendation(recommendation: dict) -> list[Violation]:
    """Все жёсткие проверки разом. Пустой список — рекомендация допустима.

    Проверки не складываются в «в целом приемлемо»: любое нарушение
    отклоняет рекомендацию целиком.
    """
    violations: list[Violation] = []
    violations += check_sulfur(recommendation.get("predicted_sulfur_mg_kg"))
    violations += check_evidence_type(recommendation.get("evidence_type"))
    if "blend_fractions" in recommendation:
        violations += check_blend_fractions(recommendation["blend_fractions"])
    if "setpoints" in recommendation:
        violations += check_within_ranges(
            recommendation["setpoints"], recommendation.get("allowed_ranges")
        )
    return violations
