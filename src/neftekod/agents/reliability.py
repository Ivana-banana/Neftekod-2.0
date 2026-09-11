"""Агент надёжности: тяжесть режима по прокси и модельные границы для оптимизации.

Роль описана в `.claude/agents/reliability.md` и `docs/tz/tz_neftekod.md`, §3.
Прямой разметки отказов, деградации катализатора или ремонтов в пакете нет,
поэтому оценка строится на прокси. Допущение прокси перечисляется в выводе
рядом с числом, а не остаётся в комментарии к коду.

Ключевое ограничение честности: физическая интерпретация P8, T11 и F19
организаторами не подтверждена, а наблюдаемые значения не похожи на
заявленные величины (ADR-007). Поэтому прокси **не** использует смысл
«температура», «расход» или «давление» и не знает, в какую сторону сигнал
«тяжелее». Он оценивает только, насколько режим необычен относительно
истории обучения и насколько он спокоен внутри окна.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import features
from .context import ScenarioContext

SEVERITY_THRESHOLDS = ((0.2, "low"), (0.5, "elevated"))
ADMISSIBLE_SEVERITY_MAX = 0.5
# Телеметрия 10-минутная, шаг рекомендаций 15–60 минут. Значение старше часа
# на момент выдачи означает молчащий сигнал, а не спокойный режим: as-of
# подставляет последнее конечное измерение, каким бы старым оно ни было
# (ADR-009), поэтому «не пропущено» само по себе ничего не гарантирует.
MAX_SIGNAL_AGE_HOURS = 1.0


@dataclass(frozen=True)
class ReliabilityAssessment:
    issue_time: str
    severity_index: float | None
    severity_class: str | None
    regime_admissible: bool
    signals: dict
    risk_factors: list
    allowed_ranges: dict
    allowed_ranges_basis: str
    assumptions: list
    refusal: dict | None = None
    agent: str = field(default="reliability", init=False)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "issue_time": self.issue_time,
            "refusal": self.refusal,
            "severity_index": self.severity_index,
            "severity_class": self.severity_class,
            "regime_admissible": self.regime_admissible,
            "risk_factors": self.risk_factors,
            "signals": self.signals,
            "allowed_ranges": self.allowed_ranges,
            "allowed_ranges_basis": self.allowed_ranges_basis,
            "assumptions": self.assumptions,
        }


def _severity_class(index: float) -> str:
    for threshold, name in SEVERITY_THRESHOLDS:
        if index < threshold:
            return name
    return "high"


def _rates_per_hour(frame: pd.DataFrame, tag: str) -> tuple[float | None, float | None]:
    part = frame[["timestamp", tag]].dropna()
    if len(part) < 2:
        return None, None
    hours = np.diff(part["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")) / 3.6e12
    steps = np.abs(np.diff(part[tag].to_numpy(dtype=float)))
    usable = hours > 0
    if not usable.any():
        return None, None
    rates = steps[usable] / hours[usable]
    return float(rates.max()), float(np.quantile(rates, 0.95))


def assess(context: ScenarioContext) -> ReliabilityAssessment:
    """Оценить тяжесть режима и выдать модельные границы для оптимизации."""
    assumptions = [
        "Прямой разметки отказов/деградации в пакете нет — индекс тяжести "
        "является прокси, а не измеренным риском.",
        "Прокси не использует физический смысл P8, T11, F19: справочные "
        "величины не согласуются с наблюдаемыми значениями (ADR-007). "
        "Направление «тяжелее/легче» по каждому сигналу неизвестно.",
        "Необычность режима считается как выход за полосу p05–p95 обучающей "
        "выборки, нормированный на ширину полосы; нестабильность — как "
        "стандартное отклонение внутри окна в тех же долях ширины полосы.",
        f"Сигнал старше {MAX_SIGNAL_AGE_HOURS:g} ч на момент выдачи считается "
        "молчащим: as-of подставляет последнее конечное измерение любой "
        "давности, поэтому «значение есть» не означает «сигнал живой».",
        "Границы allowed_ranges — консервативный модельный диапазон по "
        "истории обучения. Это допущение для эксперимента, а не промышленный "
        "предел (ТЗ, §4, «Правило границ»).",
    ]
    unusable = []
    for tag in features.TELEMETRY_TAGS:
        if context.snapshot.get(f"{tag}_missing"):
            unusable.append(f"{tag} — нет значения")
            continue
        age = context.snapshot.get(f"{tag}_age_hours")
        if age is None or age > MAX_SIGNAL_AGE_HOURS:
            unusable.append(f"{tag} — возраст {age!r} ч > {MAX_SIGNAL_AGE_HOURS:g} ч")
    allowed_ranges = {
        tag: [
            context.training_stats[f"{tag}_value"]["p05"],
            context.training_stats[f"{tag}_value"]["p95"],
        ]
        for tag in features.TELEMETRY_TAGS
    }
    basis = (
        "p05–p95 обучающей выборки по границе ADR-006: рутинная работа установки "
        "без хвостов и остановов. Консервативный модельный диапазон, допущение "
        "для эксперимента, а не подтверждённый промышленный предел"
    )
    if unusable:
        return ReliabilityAssessment(
            issue_time=str(context.issue_time),
            severity_index=None,
            severity_class=None,
            regime_admissible=False,
            signals={},
            risk_factors=[],
            allowed_ranges=allowed_ranges,
            allowed_ranges_basis=basis,
            assumptions=assumptions,
            refusal={
                "reason": f"нет актуальных значений сигналов: {'; '.join(unusable)}",
                "rule": "риск-скор «на глаз» при нехватке данных не выдаётся",
            },
        )

    signals: dict = {}
    risk_factors: list = []
    excursions: list[float] = []
    instabilities: list[float] = []
    for tag in features.TELEMETRY_TAGS:
        stats = context.training_stats[f"{tag}_value"]
        value = context.snapshot[f"{tag}_value"]
        width = stats["p95"] - stats["p05"]
        if width <= 0:
            raise ValueError(f"вырожденная обучающая полоса для {tag}")
        outside = max(0.0, stats["p05"] - value, value - stats["p95"]) / width
        window_values = pd.to_numeric(context.window_telemetry[tag], errors="coerce").dropna()
        window_std = float(window_values.std(ddof=0)) if len(window_values) > 1 else 0.0
        instability = window_std / width
        max_rate, p95_rate = _rates_per_hour(context.window_telemetry, tag)
        signals[tag] = {
            "value": value,
            "age_hours": context.snapshot[f"{tag}_age_hours"],
            "train_p05": stats["p05"],
            "train_p95": stats["p95"],
            "relative_position": (value - stats["p05"]) / width,
            "outside_band_fraction": outside,
            "window_n": int(len(window_values)),
            "window_std": window_std,
            "window_instability": instability,
            "max_abs_rate_per_hour": max_rate,
            "p95_abs_rate_per_hour": p95_rate,
        }
        excursions.append(outside)
        instabilities.append(instability)
        if outside > 0:
            risk_factors.append(f"{tag} вне полосы p05–p95 обучающей выборки на {outside:.3f} ширины")
        if instability > 0.1:
            risk_factors.append(f"{tag} нестабилен внутри окна: std = {instability:.3f} ширины полосы")

    severity = min(1.0, max(excursions) + max(instabilities))
    return ReliabilityAssessment(
        issue_time=str(context.issue_time),
        severity_index=severity,
        severity_class=_severity_class(severity),
        regime_admissible=severity < ADMISSIBLE_SEVERITY_MAX,
        signals=signals,
        risk_factors=risk_factors,
        allowed_ranges=allowed_ranges,
        allowed_ranges_basis=basis,
        assumptions=assumptions,
    )
