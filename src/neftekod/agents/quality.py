"""Агент качества: прогноз серы, риск выхода за спецификацию, доверие к входу.

Роль описана в `.claude/agents/quality.md` и `docs/tz/tz_neftekod.md`, §3.
Агент не принимает решений об управляющих воздействиях и не смягчает порог
серы: он выдаёт прогноз, его проверенную историей ошибку и риск нарушения.

Прогноз — persistence (последний опубликованный анализ). Это не «за
неимением лучшего»: на отложенной по времени истории persistence выигрывает
у телеметрической Ridge (test RMSE 2.62 против 168.2, см. build/forecast_report.md
и ADR-008), поэтому именно он несёт evidence_type: backtested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import config
from .context import TRAINING_FEATURES, ScenarioContext

BASELINE_MODEL = "persistence"
# Отбор ЛИМС суточный (10:00), публикация +4 ч. Анализ старше 48 часов
# означает, что как минимум одна проба пропущена: persistence на таком
# основании не выдаётся. Модельное допущение сценария, не норматив.
MAX_BASELINE_AGE_HOURS = 48.0


@dataclass(frozen=True)
class QualityAssessment:
    issue_time: str
    forecast: list
    in_training_range: bool
    training_range: dict
    data_confidence: dict
    assumptions: list
    refusal: dict | None = None
    agent: str = field(default="quality", init=False)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "issue_time": self.issue_time,
            "refusal": self.refusal,
            "forecast": self.forecast,
            "in_training_range": self.in_training_range,
            "training_range": self.training_range,
            "data_confidence": self.data_confidence,
            "assumptions": self.assumptions,
        }


def _risk(margin: float, rmse: float) -> str:
    """Риск нарушения спецификации в единицах проверенной историей ошибки."""
    if margin < 0:
        return "spec_violation_predicted"
    if margin < rmse:
        return "high"
    if margin < 2 * rmse:
        return "moderate"
    return "low"


def _training_range(context: ScenarioContext) -> dict:
    report = {}
    for name in TRAINING_FEATURES:
        stats = context.training_stats[name]
        value = context.snapshot.get(name)
        report[name] = {
            "value": value,
            "train_min": stats["min"],
            "train_p01": stats["p01"],
            "train_p99": stats["p99"],
            "train_max": stats["max"],
            "in_training_range": (
                value is not None and stats["p01"] <= value <= stats["p99"]
            ),
        }
    return report


def _data_confidence(context: ScenarioContext) -> dict:
    snapshot = context.snapshot
    sources = {
        "lims_sulfur": ("lims_value", "lims_age_hours", "lims_missing"),
        "pak_sulfur": ("pak_value", "pak_age_hours", "pak_missing"),
    }
    for tag in ("P8", "T11", "F19"):
        sources[tag] = (f"{tag}_value", f"{tag}_age_hours", f"{tag}_missing")
    return {
        name: {
            "value": snapshot.get(value_key),
            "age_hours": snapshot.get(age_key),
            "missing": bool(snapshot.get(missing_key)),
        }
        for name, (value_key, age_key, missing_key) in sources.items()
    }


def assess(context: ScenarioContext) -> QualityAssessment:
    """Оценить качество товарного ДТ на момент выдачи.

    При отсутствии опубликованного анализа или при слишком старом анализе
    возвращается явный отказ: неизвестная сера не считается допустимой
    (ТЗ, §4; `constraints.check_sulfur`).
    """
    snapshot = context.snapshot
    training_range = _training_range(context)
    confidence = _data_confidence(context)
    assumptions = [
        "Прогноз серы — persistence: последний опубликованный анализ ЛИМС "
        "«Гидроочистка.2». Выбран по backtest (ADR-008), не по удобству.",
        f"Анализ старше {MAX_BASELINE_AGE_HOURS:g} ч не используется как основание "
        "прогноза: суточный отбор ЛИМС + публикация 4 ч. Допущение сценария.",
        "«Вне диапазона обучения» решается по p01–p99 обучающей выборки: "
        "буквальные min/max содержат остановы и выбросы и ничего не ограничивают.",
        "ПАК приводится как независимый свежий сигнал, но в прогноз не входит: "
        "задержка публикации ПАК организаторами не подтверждена (ADR-009).",
    ]
    in_range = all(item["in_training_range"] for item in training_range.values())

    baseline = snapshot.get("lims_value")
    baseline_age = snapshot.get("lims_age_hours")
    if snapshot.get("lims_missing") or baseline is None:
        refusal = {
            "reason": "нет опубликованного анализа ЛИМС на момент выдачи",
            "rule": "неизвестная сера не считается допустимой (ТЗ, §4)",
        }
        return QualityAssessment(
            issue_time=str(context.issue_time),
            forecast=[],
            in_training_range=in_range,
            training_range=training_range,
            data_confidence=confidence,
            assumptions=assumptions,
            refusal=refusal,
        )
    if baseline_age is not None and baseline_age > MAX_BASELINE_AGE_HOURS:
        refusal = {
            "reason": (
                f"последний анализ ЛИМС старше {MAX_BASELINE_AGE_HOURS:g} ч "
                f"(возраст {baseline_age:.1f} ч)"
            ),
            "rule": "persistence на устаревшем анализе не является обоснованным прогнозом",
        }
        return QualityAssessment(
            issue_time=str(context.issue_time),
            forecast=[],
            in_training_range=in_range,
            training_range=training_range,
            data_confidence=confidence,
            assumptions=assumptions,
            refusal=refusal,
        )

    forecast = []
    for horizon in sorted(context.backtest):
        metrics = context.backtest[horizon].get(BASELINE_MODEL)
        if metrics is None:
            continue
        margin = config.SULFUR_LIMIT_MG_KG - baseline
        forecast.append(
            {
                "horizon_hours": horizon,
                "model": BASELINE_MODEL,
                "predicted_sulfur_mg_kg": baseline,
                "spec_limit_mg_kg": config.SULFUR_LIMIT_MG_KG,
                "margin_mg_kg": margin,
                "backtest_rmse_mg_kg": metrics["rmse"],
                "backtest_mae_mg_kg": metrics["mae"],
                "backtest_n": metrics["n"],
                "margin_in_rmse": margin / metrics["rmse"],
                "risk_exceed_spec": _risk(margin, metrics["rmse"]),
                "in_training_range": in_range,
                "evidence_type": metrics["evidence_type"],
            }
        )
    if not forecast:
        raise ValueError("в backtest нет строк persistence: пересоберите build/forecast_report.csv")
    return QualityAssessment(
        issue_time=str(context.issue_time),
        forecast=forecast,
        in_training_range=in_range,
        training_range=training_range,
        data_confidence=confidence,
        assumptions=assumptions,
    )
