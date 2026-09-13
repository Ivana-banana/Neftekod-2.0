"""Показатели спецификации товарного ДТ помимо серы: T95 и цетановое число.

Организаторы назвали обязательными три показателя — серу, T95 и цетановое
число (`docs/tz/organizer-clarifications.md`, ADR-004). Сера была реализована
первой, потому что её предел задан организаторами; численные пределы двух
остальных они прямо оставили на усмотрение команды, и это единственная
причина, по которой они появились позже, а не признак их второстепенности.

Модуль отвечает за два разных вопроса и намеренно их не смешивает:

1. **Что измерено.** Оба показателя есть в ЛИМС на той же точке отбора
   «Гидроочистка.2», что и сера товарного продукта: `95%.T` — суточно,
   `CetaneNumber` — примерно раз в месяц. Блокировка ADR-003 касается
   ссылки `Pipeline` внутри формулы ВАК, а не прямых лабораторных измерений,
   поэтому здесь используются измерения, а не расчёт по формуле.
2. **Что домыслено.** Цетановое число дешёвого компонента в пакете не
   измеряется вовсе — это явное допущение сценария (см. `CUT_CETANE_NUMBER`).
   Модели смешения тоже допущения: историей они не проверялись.

Возраст значения важнее его наличия: `features.synchronize_features()` по
построению отдаёт последнее измерение любой давности (ADR-009), поэтому у
каждого показателя свой порог устаревания, и просроченное значение помечается
`verified: false` вместо того, чтобы молча попасть в проверку ограничений.
См. ADR-015.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config, splits

# Точка отбора та же, что у серы товарного продукта (features.TARGET_TAG).
BASE_T95_TAG = "LIMS:Гидроочистка.2.95%.T"
BASE_CETANE_TAG = "LIMS:Гидроочистка.2.CetaneNumber"
# Дешёвый компонент блендинга — сырьё гидроочистки, та же точка, из которой
# берётся его сера в ADR-013.
CUT_T95_TAG = "LIMS:Гидроочистка.1.95%.T"

# Цетановое число дешёвого компонента не измеряется ни в одной точке пакета.
# Взята нижняя граница ориентира для straight-run diesel из
# `docs/research/2026-09-blending-recipes.md` (~45–60): нижняя, чтобы
# допущение не льстило смеси. Это допущение, а не измерение.
CUT_CETANE_NUMBER = 45.0
CUT_CETANE_BASIS = (
    "допущение: цетановое число дешёвого компонента в пакете не измеряется; "
    "взята нижняя граница ориентира ~45–60 для straight-run diesel из "
    "docs/research/2026-09-blending-recipes.md, чтобы допущение не льстило смеси"
)

# Цетановое число смешивается через blend index CN_b = (Σ wᵢ·CNᵢ^n)^(1/n)
# (ADR-004: «нелинейный blending index», не линейная интерполяция). Показатель
# n = 1.5 взят из демонстрационного примера того же отчёта, где он прямо назван
# условным и подлежащим калибровке на своих данных. На долях дешёвого
# компонента порядка 1e-4 он отличается от линейного смешения меньше, чем на
# 1e-3 единицы ЦЧ — числа приведены в ADR-015.
CETANE_BLEND_INDEX_EXPONENT = 1.5

# Физически невозможные значения температуры: в `LIMS:Гидроочистка.1.95%.T`
# минимум по всей выгрузке равен 0 °C. Такое значение не является измерением
# и не используется, иначе смесь получает несуществующий запас по T95.
MIN_PHYSICAL_T95_C = 100.0


@dataclass(frozen=True)
class Measured:
    """Измеренное значение показателя на момент выдачи с его возрастом."""

    tag: str
    value: float | None
    unit: str | None
    sample_time: str | None
    available_at: str | None
    age_hours: float | None
    max_age_hours: float
    verified: bool
    detail: str

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "value": self.value,
            "unit": self.unit,
            "sample_time": self.sample_time,
            "available_at": self.available_at,
            "age_hours": self.age_hours,
            "max_age_hours": self.max_age_hours,
            "verified": self.verified,
            "detail": self.detail,
        }


def resolve_measured(
    long_format: pd.DataFrame,
    issue_time,
    tag: str,
    *,
    max_age_hours: float,
    min_physical: float | None = None,
) -> Measured:
    """Последний опубликованный к моменту выдачи анализ показателя.

    Берётся только то, что лаборатория успела выдать (`available_at`), и только
    если оно не старше порога. Просроченное или физически невозможное значение
    возвращается с `verified: false` и без числа: неизвестный показатель не
    считается допустимым (ТЗ, §4).
    """
    issue = pd.Timestamp(issue_time)
    part = long_format[long_format["tag"].eq(tag)].copy()
    part["value"] = pd.to_numeric(part["value"], errors="coerce")
    part = part.dropna(subset=["value"])
    published = splits.available_history(part, issue)
    if published.empty:
        return Measured(
            tag=tag,
            value=None,
            unit=None,
            sample_time=None,
            available_at=None,
            age_hours=None,
            max_age_hours=max_age_hours,
            verified=False,
            detail=f"нет опубликованного анализа {tag} на момент выдачи",
        )
    last = published.sort_values("timestamp").iloc[-1]
    sample_time = pd.Timestamp(last["timestamp"])
    age_hours = (issue - sample_time).total_seconds() / 3600.0
    value = float(last["value"])
    unit = None if pd.isna(last.get("unit")) else str(last.get("unit"))
    common = {
        "tag": tag,
        "unit": unit,
        "sample_time": str(sample_time),
        "available_at": str(pd.Timestamp(last["available_at"])),
        "age_hours": age_hours,
        "max_age_hours": max_age_hours,
    }
    if min_physical is not None and value < min_physical:
        return Measured(
            **common,
            value=None,
            verified=False,
            detail=(
                f"последнее значение {tag} равно {value:g} и физически невозможно "
                f"(ниже {min_physical:g}): как измерение не используется"
            ),
        )
    if age_hours > max_age_hours:
        return Measured(
            **common,
            value=None,
            verified=False,
            detail=(
                f"анализ {tag} старше {max_age_hours / 24:g} суток "
                f"(возраст {age_hours / 24:.1f} суток): показатель не подтверждён"
            ),
        )
    return Measured(
        **common,
        value=value,
        verified=True,
        detail=f"анализ {tag} от {sample_time}, возраст {age_hours:.1f} ч",
    )


def blend_t95(fractions: dict, component_t95: dict) -> float | None:
    """T95 смеси массовым балансом: T95 = Σ wᵢ·T95ᵢ.

    Заявленное допущение, а не проверенная модель. Внешний ориентир
    (ADR-004) рекомендует для дистилляции модель кумулятивной кривой по
    псевдокомпонентам, для которой нужны полные кривые разгонки компонентов.
    На долях дешёвого компонента порядка 1e-4 и разнице T95 компонентов в
    единицы °C любая разумная модель смешения даёт сдвиг меньше 1e-3 °C, то
    есть на четыре порядка ниже разрешения лабораторного значения (1 °C).
    См. ADR-015.
    """
    total = 0.0
    for name, weight in (fractions or {}).items():
        value = component_t95.get(name)
        if value is None:
            return None
        total += float(weight) * float(value)
    return total


def cetane_blend_index_form(fractions: dict, component_cetane: dict) -> float | None:
    """Цетановое число смеси через blend index: CN = (Σ wᵢ·CNᵢ^n)^(1/n).

    Нелинейная форма взята по ADR-004; показатель n назван в источнике
    условным и подлежащим калибровке, поэтому число выведено в константу
    `CETANE_BLEND_INDEX_EXPONENT`, а не вписано в формулу.
    """
    exponent = CETANE_BLEND_INDEX_EXPONENT
    total = 0.0
    for name, weight in (fractions or {}).items():
        value = component_cetane.get(name)
        if value is None or float(value) <= 0.0:
            return None
        total += float(weight) * float(value) ** exponent
    if total <= 0.0:
        return None
    return total ** (1.0 / exponent)


def cetane_linear_form(fractions: dict, component_cetane: dict) -> float | None:
    """Цетановое число смеси линейным массовым балансом: CN = Σ wᵢ·CNᵢ."""
    total = 0.0
    for name, weight in (fractions or {}).items():
        value = component_cetane.get(name)
        if value is None:
            return None
        total += float(weight) * float(value)
    return total


def blend_cetane_number(fractions: dict, component_cetane: dict) -> float | None:
    """Цетановое число смеси — худшая из двух рассматриваемых моделей смешения.

    Формы функции смешения мы не знаем: ADR-004 предписывает нелинейный blend
    index, но показатель степени в источнике прямо назван условным и
    подлежащим калибровке, которой у нас нет. Измерено при реализации: на смеси
    55 и 45 единиц пополам индекс даёт 50.125 против 50.0 у линейного баланса,
    то есть **индекс выше** линейного среднего и сам по себе не консервативен.

    Ставить защитную проверку на оптимистичную из двух оценок нельзя, поэтому
    берётся минимум. Это заявленная политика, а не подгонка: она не зависит от
    того, какая форма окажется верной, и озвучивается вместе с результатом.
    На долях дешёвого компонента порядка 1e-4 обе формы расходятся меньше чем
    на 1e-4 единицы ЦЧ, так что на текущих данных выбор ничего не решает —
    но перестанет быть безразличным, если доля вырастет. См. ADR-015.
    """
    index = cetane_blend_index_form(fractions, component_cetane)
    linear = cetane_linear_form(fractions, component_cetane)
    if index is None or linear is None:
        return None
    return min(index, linear)


@dataclass(frozen=True)
class SpecIndicators:
    """Обязательные показатели продукта, кроме серы, на момент выдачи."""

    t95: Measured
    cetane_number: Measured

    @property
    def verified(self) -> bool:
        return self.t95.verified and self.cetane_number.verified

    def unverified_reasons(self) -> list[str]:
        return [
            item.detail
            for item in (self.t95, self.cetane_number)
            if not item.verified
        ]

    def to_dict(self) -> dict:
        return {
            "t95": self.t95.to_dict()
            | {
                "limit_c": config.T95_LIMIT_C,
                "direction": "max",
                "margin_c": (
                    None if self.t95.value is None else config.T95_LIMIT_C - self.t95.value
                ),
            },
            "cetane_number": self.cetane_number.to_dict()
            | {
                "minimum": config.CETANE_NUMBER_MIN,
                "direction": "min",
                "margin": (
                    None
                    if self.cetane_number.value is None
                    else self.cetane_number.value - config.CETANE_NUMBER_MIN
                ),
            },
            "limits_basis": (
                "пределы T95 ≤ 360 °C и ЦЧ ≥ 51 приняты по ГОСТ 32511-2013 / "
                "ГОСТ Р 52368 для К5 как параметры сценария: организаторы оставили "
                "их на усмотрение команды (ADR-004, ADR-015)"
            ),
        }


def resolve_product_indicators(long_format: pd.DataFrame, issue_time) -> SpecIndicators:
    """Измеренные T95 и цетановое число товарного продукта на момент выдачи."""
    return SpecIndicators(
        t95=resolve_measured(
            long_format,
            issue_time,
            BASE_T95_TAG,
            max_age_hours=config.T95_MAX_AGE_HOURS,
            min_physical=MIN_PHYSICAL_T95_C,
        ),
        cetane_number=resolve_measured(
            long_format,
            issue_time,
            BASE_CETANE_TAG,
            max_age_hours=config.CETANE_MAX_AGE_HOURS,
        ),
    )


def resolve_cut_t95(long_format: pd.DataFrame, issue_time) -> Measured:
    """Измеренный T95 дешёвого компонента блендинга на момент выдачи."""
    return resolve_measured(
        long_format,
        issue_time,
        CUT_T95_TAG,
        max_age_hours=config.T95_MAX_AGE_HOURS,
        min_physical=MIN_PHYSICAL_T95_C,
    )
