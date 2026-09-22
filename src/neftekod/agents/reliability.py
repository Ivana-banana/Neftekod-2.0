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
# См. ADR-010.
MAX_SIGNAL_AGE_HOURS = 1.0
# Скорость изменения сигнала, при которой он проходит всю рутинную полосу
# быстрее чем за час. Порождает фактор риска, но НЕ входит в severity_index:
# ограничений скорости изменения в задаче не задано (ответы организаторов),
# и придумывать за них предел нельзя. См. ADR-010.
FAST_EXCURSION_BANDS_PER_HOUR = 1.0
# Останов установки: T11 (массовый расход сырья по справочнику КИП) ниже 50
# при обычных ~360 — 7073 строки train, названные простоем в ADR-018. Единицы
# сигнала не подтверждены (ADR-007), поэтому это признак «сырьё не идёт», а не
# число в т/ч. Такой момент — не «тяжёлый режим», а отсутствие режима, и
# оператору это говорится прямо (ADR-019).
SHUTDOWN_T11_MAX = 50.0


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
    telemetry_defects: dict = field(default_factory=dict)
    plant_stopped: bool = False
    agent: str = field(default="reliability", init=False)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "issue_time": self.issue_time,
            "refusal": self.refusal,
            "severity_index": self.severity_index,
            "severity_class": self.severity_class,
            "regime_admissible": self.regime_admissible,
            "plant_stopped": self.plant_stopped,
            "risk_factors": self.risk_factors,
            "signals": self.signals,
            "allowed_ranges": self.allowed_ranges,
            "allowed_ranges_basis": self.allowed_ranges_basis,
            "assumptions": self.assumptions,
            "telemetry_defects": self.telemetry_defects,
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
        f"Скорость изменения свыше {FAST_EXCURSION_BANDS_PER_HOUR:g} ширины полосы в час "
        "даёт фактор риска, но в severity_index не входит: ограничений скорости "
        "изменения организаторы не задали, и придумывать предел за них нельзя.",
        f"Граница допустимости режима severity_index < {ADMISSIBLE_SEVERITY_MAX:g} — "
        "модельный порог, выбранный до прогона сценариев и не подобранный по "
        "результату. Внешнего обоснования у него нет.",
        "Границы allowed_ranges — консервативный модельный диапазон по "
        "истории обучения. Это допущение для эксперимента, а не промышленный "
        "предел (ТЗ, §4, «Правило границ»).",
        "Значение ровно 307.0 и сигнал, повторивший одно число 6 строк подряд, "
        "считаются дефектом датчика и не используются как измерение (ADR-018). "
        "Отрицательные значения дефектом не считаются: они совпадают с простоем.",
        f"T11 ниже {SHUTDOWN_T11_MAX:g} при обычных ~360 трактуется как останов "
        "установки (простой по ADR-018): тогда индекс тяжести описывает не "
        "режим, а его отсутствие, и любое изменение уставок недопустимо (ADR-019).",
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
            telemetry_defects=context.telemetry_defects,
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
            "max_rate_in_bands_per_hour": None if max_rate is None else max_rate / width,
        }
        excursions.append(outside)
        instabilities.append(instability)
        if outside > 0:
            risk_factors.append(f"{tag} вне полосы p05–p95 обучающей выборки на {outside:.3f} ширины")
        if instability > 0.1:
            risk_factors.append(f"{tag} нестабилен внутри окна: std = {instability:.3f} ширины полосы")
        # Спокойный std за окно прячет короткий скачок: без этой проверки
        # период с броском на несколько ширин полосы за час назывался бы
        # устойчивым. Найдено прогоном роли на сценарии 1.
        if max_rate is not None and max_rate / width > FAST_EXCURSION_BANDS_PER_HOUR:
            risk_factors.append(
                f"{tag}: скорость изменения до {max_rate / width:.2f} ширины полосы в час "
                "внутри окна — короткий бросок, не видимый в std за всё окно"
            )

    for tag, counts in context.telemetry_defects.get("window", {}).items():
        if any(counts.values()):
            risk_factors.append(
                f"{tag}: в окне {counts} значений маскировано как дефект датчика (ADR-018)"
            )

    severity = min(1.0, max(excursions) + max(instabilities))
    plant_stopped = bool(context.snapshot["T11_value"] < SHUTDOWN_T11_MAX)
    if plant_stopped:
        risk_factors.insert(
            0,
            f"установка остановлена: T11 = {context.snapshot['T11_value']:.4g} при пороге "
            f"останова {SHUTDOWN_T11_MAX:g} (обычно ~360, ADR-018) — сырьё не идёт, "
            "режима нет; рекомендации по режиму невозможны",
        )
    return ReliabilityAssessment(
        issue_time=str(context.issue_time),
        severity_index=severity,
        severity_class=_severity_class(severity),
        regime_admissible=severity < ADMISSIBLE_SEVERITY_MAX and not plant_stopped,
        plant_stopped=plant_stopped,
        signals=signals,
        risk_factors=risk_factors,
        allowed_ranges=allowed_ranges,
        allowed_ranges_basis=basis,
        assumptions=assumptions,
        telemetry_defects=context.telemetry_defects,
    )
