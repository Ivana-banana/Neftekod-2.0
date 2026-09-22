"""Жёсткие ограничения. Проверяются всегда, вне зависимости от метрик.

Это не «валидация входа» и не рекомендательная эвристика: нарушение любого
из этих правил означает, что рекомендация не выдаётся (ТЗ, §4). Улучшение
выпуска, энергозатрат или любой другой метрики не является основанием
ослабить проверку.

Модуль намеренно не знает ничего о моделях и агентах: агент оптимизации и
оркестратор вызывают `validate_recommendation()` и получают список
нарушений, а `blocking()` отбирает из него те, что отклоняют рекомендацию.
Единственное различие — неподтверждённый показатель не блокирует сохранение
текущего режима (ADR-017); для любого изменения пустой список остаётся
единственным способом считать рекомендацию допустимой.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from . import config

# CLAUDE.md, «Верификация и честность»: каждый вывод агента оптимизации
# обязан различать проверенный историей прогноз и модельную оценку эффекта
# ещё не случавшегося действия.
EVIDENCE_TYPES = frozenset({"backtested", "simulated_unvalidated"})

# «Нарушен» и «не подтверждён» — разные случаи (ADR-017). Нарушение — число вне
# предела или мусор в поле: отклоняется всё, включая сохранение режима.
# Неподтверждённый показатель — числа нет, потому что анализ просрочен,
# отсутствует или нефизичен: отклоняется любое изменение, статус-кво допустим
# с явной пометкой.
VIOLATED = "violated"
UNVERIFIED = "unverified"
# Текущая уставка вне модельного диапазона p05–p95 (ADR-010): для статус-кво
# это не «рекомендация выйти за предел» — режим уже там, и его допустимость
# оценивает агент надёжности через severity_index. Для любого изменения
# уставок диапазон остаётся жёстким. См. ADR-019.
MODEL_RANGE = "model_range"


@dataclass(frozen=True)
class Violation:
    rule: str
    detail: str
    kind: str = VIOLATED


def _is_number(value) -> bool:
    """Конечное вещественное число: не bool, не NaN, не ±inf, numpy-скаляры допустимы."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value))


def _at_resolution(value: float, step: float) -> float:
    """Округлить до шага разрешения входа, половина — вверх (ADR-017).

    Через Decimal от repr, а не через `round()`: встроенное округление
    банковское и работает с двоичным представлением, из-за чего 50.95 могло бы
    оказаться то 51.0, то 50.9.
    """
    quantum = Decimal(repr(float(step)))
    steps = (Decimal(repr(float(value))) / quantum).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return float(steps * quantum)


def _unverified(rule: str, title: str) -> list[Violation]:
    return [
        Violation(rule, f"показатель «{title}» не подтверждён анализом: числа нет", kind=UNVERIFIED)
    ]


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
                f"сера {value_mg_kg:.6g} мг/кг > предела {config.SULFUR_LIMIT_MG_KG:g} мг/кг",
            )
        ]
    return []


def check_t95(value_c) -> list[Violation]:
    """T95 товарного ДТ ≤ предела (обязательный показатель, ADR-004, ADR-015).

    Отсутствующее значение (`None`) — неподтверждённый показатель, нечисловое —
    нарушение: «не измерили» не означает «в пределах нормы». С пределом
    сравнивается значение в разрешении входа — целых °C (ADR-017).
    """
    if value_c is None:
        return _unverified("t95", "T95")
    if not _is_number(value_c):
        return [Violation("t95", f"нет числового значения T95: {value_c!r}")]
    compared = _at_resolution(value_c, config.T95_RESOLUTION_C)
    if compared > config.T95_LIMIT_C:
        return [
            Violation(
                "t95",
                f"T95 {value_c:.6g} °C (в разрешении {config.T95_RESOLUTION_C:g} °C: "
                f"{compared:g}) > предела {config.T95_LIMIT_C} °C",
            )
        ]
    return []


def check_cetane_number(value) -> list[Violation]:
    """Цетановое число товарного ДТ ≥ минимума (ADR-004, ADR-015).

    Единственный показатель проекта с ограничением снизу: чем меньше, тем
    хуже. Отсутствие значения — неподтверждённый показатель; сравнение — в
    разрешении 0.1 (ADR-017).
    """
    if value is None:
        return _unverified("cetane_number", "цетановое число")
    if not _is_number(value):
        return [Violation("cetane_number", f"нет числового значения цетанового числа: {value!r}")]
    compared = _at_resolution(value, config.CETANE_NUMBER_RESOLUTION)
    if compared < config.CETANE_NUMBER_MIN:
        return [
            Violation(
                "cetane_number",
                f"цетановое число {value:.6g} (в разрешении {config.CETANE_NUMBER_RESOLUTION:g}: "
                f"{compared:g}) < минимума {config.CETANE_NUMBER_MIN}",
            )
        ]
    return []


def check_density(value_kg_m3) -> list[Violation]:
    """Плотность товарного ДТ при 15 °C внутри диапазона ГОСТ (ADR-016).

    Единственная проверка проекта с ограничением **с двух сторон**: и слишком
    лёгкое, и слишком тяжёлое топливо не соответствует спецификации. Как и у
    остальных обязательных показателей, отсутствующее значение — неподтверждённый
    показатель; сравнение — в разрешении 0.1 кг/м³ (ADR-017).
    """
    if value_kg_m3 is None:
        return _unverified("density", "плотность")
    if not _is_number(value_kg_m3):
        return [Violation("density", f"нет числового значения плотности: {value_kg_m3!r}")]
    compared = _at_resolution(value_kg_m3, config.DENSITY_RESOLUTION_KG_M3)
    if compared < config.DENSITY_MIN_KG_M3 or compared > config.DENSITY_MAX_KG_M3:
        return [
            Violation(
                "density",
                f"плотность {value_kg_m3:.7g} кг/м³ (в разрешении "
                f"{config.DENSITY_RESOLUTION_KG_M3:g}: {compared:g}) вне диапазона "
                f"{config.DENSITY_MIN_KG_M3}–{config.DENSITY_MAX_KG_M3} кг/м³",
            )
        ]
    return []


def check_blend_fractions(fractions) -> list[Violation]:
    """Доли компонентов блендинга суммируются в 100% и неотрицательны."""
    if not fractions:
        return [Violation("blend_fractions", "состав блендинга не задан")]
    if not isinstance(fractions, (dict, list, tuple)):
        return [Violation("blend_fractions", f"состав блендинга не является словарём: {fractions!r}")]
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
    if setpoints is not None and not isinstance(setpoints, dict):
        return [Violation("range", f"уставки должны быть словарём, получено {type(setpoints).__name__}")]
    if allowed_ranges is not None and not isinstance(allowed_ranges, dict):
        return [Violation("range", f"диапазоны должны быть словарём, получено {type(allowed_ranges).__name__}")]
    for name, value in (setpoints or {}).items():
        bounds = (allowed_ranges or {}).get(name)
        if bounds is None:
            violations.append(Violation("range", f"для параметра {name} не задан диапазон"))
            continue
        # Мусорный диапазон — нарушение, а не traceback у оркестратора (ADR-019).
        try:
            low, high = bounds
        except (TypeError, ValueError):
            violations.append(Violation("range", f"диапазон {name} не является парой границ: {bounds!r}"))
            continue
        if not (_is_number(low) and _is_number(high)) or low > high:
            violations.append(Violation("range", f"диапазон {name} задан некорректно: {bounds!r}"))
            continue
        if not _is_number(value):
            violations.append(Violation("range", f"нечисловая уставка {name}: {value!r}"))
        elif not low <= value <= high:
            violations.append(Violation("range", f"{name}={value:.6g} вне диапазона [{low:.6g}, {high:.6g}]"))
    return violations


def check_evidence_type(evidence_type) -> list[Violation]:
    """Поле `evidence_type` обязательно и имеет одно из двух значений."""
    if not isinstance(evidence_type, str) or evidence_type not in EVIDENCE_TYPES:
        return [
            Violation(
                "evidence_type",
                f"ожидалось одно из {sorted(EVIDENCE_TYPES)}, получено {evidence_type!r}",
            )
        ]
    return []


def check_measured_product(indicators, last_sulfur_mg_kg=None) -> list[Violation]:
    """Измеренные показатели продукта против пределов (ADR-019).

    Прогноз серы сдвинут к уровню обучения (ADR-012): анализ 10.5 мг/кг даёт
    прогноз 9.2, и проверка одного прогноза пропустила бы доказанное
    нарушение. Лабораторный результат — контрольный факт (ТЗ, §2), поэтому с
    пределом сравнивается и сам последний анализ, и измеренные T95, цетановое
    число и плотность продукта. Показатель без числа здесь не нарушение —
    его отсутствие ловит `validate_recommendation` как «не подтверждён».
    """
    violations: list[Violation] = []
    if last_sulfur_mg_kg is not None and _is_number(last_sulfur_mg_kg):
        if last_sulfur_mg_kg > config.SULFUR_LIMIT_MG_KG:
            violations.append(
                Violation(
                    "sulfur",
                    f"измеренная сера продукта {last_sulfur_mg_kg:g} мг/кг > предела "
                    f"{config.SULFUR_LIMIT_MG_KG:g} мг/кг: доказанное нарушение спецификации",
                )
            )
    for key, check in (
        ("t95", check_t95),
        ("cetane_number", check_cetane_number),
        ("density", check_density),
    ):
        value = ((indicators or {}).get(key) or {}).get("value")
        if value is None:
            continue
        for violation in check(value):
            violations.append(Violation(violation.rule, f"измерено на продукте: {violation.detail}"))
    return violations


def validate_recommendation(recommendation: dict) -> list[Violation]:
    """Все жёсткие проверки разом. Пустой список — рекомендация допустима.

    Проверки не складываются в «в целом приемлемо»: любое нарушение
    отклоняет рекомендацию целиком. Исключения выносятся в `blocking()`:
    неподтверждённый показатель (ADR-017) и текущая уставка вне модельного
    диапазона (ADR-019) не блокируют статус-кво.
    """
    violations: list[Violation] = []
    violations += check_sulfur(recommendation.get("predicted_sulfur_mg_kg"))
    # Показатели спецификации проверяются безусловно, а не «если поле есть»:
    # серу, T95 и цетановое число назвали обязательными организаторы (ADR-004),
    # плотность добавлена как нормируемый ГОСТ показатель, измеряемый на той же
    # точке (ADR-016). Рекомендация без любого из них не является проверенной.
    violations += check_t95(recommendation.get("predicted_t95_c"))
    violations += check_cetane_number(recommendation.get("predicted_cetane_number"))
    violations += check_density(recommendation.get("predicted_density_kg_m3"))
    violations += check_evidence_type(recommendation.get("evidence_type"))
    # Состав и уставки обязательны: без них пересчёт смеси и проверка
    # диапазонов пропускались бы, и рекомендацию можно было бы протащить,
    # просто не положив поле в пакет (ADR-019).
    if "blend_fractions" not in recommendation:
        violations.append(Violation("blend_fractions", "в рекомендации нет состава блендинга"))
    else:
        violations += check_blend_fractions(recommendation["blend_fractions"])
    if "setpoints" not in recommendation:
        violations.append(Violation("range", "в рекомендации нет уставок"))
    else:
        violations += [
            Violation(v.rule, v.detail, kind=MODEL_RANGE)
            for v in check_within_ranges(
                recommendation["setpoints"], recommendation.get("allowed_ranges")
            )
        ]
    return violations


def blocking(violations: list[Violation], *, status_quo: bool) -> list[Violation]:
    """Нарушения, которые отклоняют рекомендацию (ADR-017, ADR-019).

    Для изменения режима или состава блокирует всё. Для статус-кво не
    блокируют два вида: неподтверждённый показатель («ничего не менять» не
    требует сертифицировать то, чего лаборатория ещё не выдала) и текущая
    уставка вне модельного диапазона p05–p95 (режим уже там; исторический
    диапазон — не промышленный предел, ТЗ §4, а допустимость режима оценивает
    агент надёжности). Доказанное нарушение блокирует и статус-кво. Что
    считать статус-кво, решает вызывающий — по фактическим уставкам и
    составу, а не по ярлыку `action`.
    """
    return [v for v in violations if not (status_quo and v.kind in (UNVERIFIED, MODEL_RANGE))]


def out_of_model_range(violations: list[Violation]) -> list[str]:
    """Тексты выходов текущих уставок за модельный диапазон — для предупреждения."""
    return [v.detail for v in violations if v.kind == MODEL_RANGE]


def unverified_indicators(violations: list[Violation]) -> list[str]:
    """Имена неподтверждённых показателей — для пометки в рекомендации."""
    return sorted({v.rule for v in violations if v.kind == UNVERIFIED})
