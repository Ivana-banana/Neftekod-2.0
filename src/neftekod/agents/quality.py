"""Агент качества: прогноз серы, риск выхода за спецификацию, доверие к входу.

Роль описана в `.claude/agents/quality.md` и `docs/tz/tz_neftekod.md`, §3.
Агент не принимает решений об управляющих воздействиях и не смягчает порог
серы: он выдаёт прогноз, его проверенную историей ошибку и риск нарушения.

Прогноз — сдвинутый persistence: shrinkage · последний опубликованный анализ
ЛИМС + (1 − shrinkage) · уровень обучающей выборки. Доля shrinkage выбрана
только на valid (ADR-012); persistence (shrinkage = 1) и константа
(shrinkage = 0) входят в то же семейство и проиграли ей на отложенной
истории (test RMSE 2.296 против 2.620 и 2.483 при h = 0). Именно этот прогноз
несёт evidence_type: backtested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .. import anomaly, config, constraints, features, spec, splits
from .context import TRAINING_FEATURES, ScenarioContext

BASELINE_MODEL = "shrunk_persistence"
# Отбор ЛИМС суточный (10:00), публикация +4 ч. Анализ старше 48 часов
# означает, что как минимум одна проба пропущена: прогноз на таком
# основании не выдаётся. Модельное допущение сценария — см. ADR-010.
MAX_BASELINE_AGE_HOURS = 48.0
# ПАК, повторяющий одно и то же число дольше этого срока, считается залипшим
# сигналом, а не свежим измерением. Порог тот же, что у телеметрии в ADR-018
# (6 строк по 10 минут = 1 час), см. ADR-019.
PAK_FROZEN_MIN_HOURS = 1.0
# Измеренное нарушение спецификации — отдельный класс риска: это не прогноз,
# а факт, и он блокирует даже сохранение режима (ADR-019).
RISK_MEASURED_VIOLATION = "spec_violation_measured"


@dataclass(frozen=True)
class QualityAssessment:
    issue_time: str
    forecast: list
    in_training_range: bool
    training_range: dict
    data_confidence: dict
    assumptions: list
    input_anomaly: dict = field(default_factory=dict)
    spec_indicators: dict = field(default_factory=dict)
    # Доказанные нарушения по измеренным показателям продукта (ADR-019):
    # пустой список — утверждение «измеренных нарушений нет».
    measured_violations: list = field(default_factory=list)
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
            "input_anomaly": self.input_anomaly,
            "spec_indicators": self.spec_indicators,
            "measured_violations": self.measured_violations,
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
        width = stats["p99"] - stats["p01"]
        report[name] = {
            "value": value,
            "train_min": stats["min"],
            "train_p01": stats["p01"],
            "train_p99": stats["p99"],
            "train_max": stats["max"],
            # Бинарного флага мало: значение на 0.97 ширины формально «внутри»,
            # но любое повышение выводит его наружу. Оптимизатор должен видеть
            # запас до границы заранее, а не постфактум.
            "relative_position": None if value is None or width <= 0 else (value - stats["p01"]) / width,
            "in_training_range": (
                value is not None and stats["p01"] <= value <= stats["p99"]
            ),
        }
    return report


def _pak_frozen_hours(context: ScenarioContext) -> float | None:
    """Сколько часов ПАК повторяет одно и то же число к моменту выдачи.

    Залипший поточный анализатор выглядит как свежее измерение с возрастом
    0 ч; в выгрузке NaN нет, поэтому дефект находится по серии одинаковых
    значений, как у телеметрии (ADR-018). Считается по опубликованным к моменту
    выдачи значениям — заглядывать в будущее серия не может.
    """
    frame = context.long_format
    part = frame[frame["tag"].eq(features.PAK_SULFUR_TAG)].copy()
    part["value"] = pd.to_numeric(part["value"], errors="coerce")
    part = part.dropna(subset=["value"])
    published = splits.available_history(part, context.issue_time)
    if published.empty:
        return None
    published = published.sort_values("timestamp", kind="mergesort")
    values = published["value"].to_numpy(dtype=float)
    stamps = pd.to_datetime(published["timestamp"]).to_numpy()
    start = len(values) - 1
    while start > 0 and values[start - 1] == values[-1]:
        start -= 1
    if start == len(values) - 1:
        return 0.0
    return float((context.issue_time - pd.Timestamp(stamps[start])).total_seconds() / 3600.0)


def _data_confidence(context: ScenarioContext) -> dict:
    snapshot = context.snapshot
    sources = {
        "lims_sulfur": ("lims_value", "lims_age_hours", "lims_missing"),
        "pak_sulfur": ("pak_value", "pak_age_hours", "pak_missing"),
    }
    for tag in ("P8", "T11", "F19"):
        sources[tag] = (f"{tag}_value", f"{tag}_age_hours", f"{tag}_missing")
    confidence = {
        name: {
            "value": snapshot.get(value_key),
            "age_hours": snapshot.get(age_key),
            "missing": bool(snapshot.get(missing_key)),
            # Свежесть сигнала не означает, что он понят: единицы и масштаб
            # P8/T11/F19 организаторами не подтверждены (ADR-007). Без этой
            # пометки «age 0, missing false» читается как полное доверие.
            "units_confirmed": name in ("lims_sulfur", "pak_sulfur"),
        }
        for name, (value_key, age_key, missing_key) in sources.items()
    }
    frozen_hours = None if confidence["pak_sulfur"]["missing"] else _pak_frozen_hours(context)
    confidence["pak_sulfur"]["frozen_hours"] = frozen_hours
    confidence["pak_sulfur"]["frozen"] = (
        frozen_hours is not None and frozen_hours >= PAK_FROZEN_MIN_HOURS
    )
    return confidence


def assess(context: ScenarioContext) -> QualityAssessment:
    """Оценить качество товарного ДТ на момент выдачи.

    При отсутствии опубликованного анализа или при слишком старом анализе
    возвращается явный отказ: неизвестная сера не считается допустимой
    (ТЗ, §4; `constraints.check_sulfur`).
    """
    snapshot = context.snapshot
    training_range = _training_range(context)
    confidence = _data_confidence(context)
    # T95 и цетановое число — обязательные показатели наравне с серой (ADR-004).
    # Считаются до всех ветвлений: даже при отказе от прогноза серы оператор
    # должен видеть, подтверждены ли они (ADR-015).
    spec_indicators = spec.resolve_product_indicators(
        context.long_format, context.issue_time
    ).to_dict()
    assumptions = [
        "Прогноз серы — сдвинутый persistence: shrinkage · последний "
        "опубликованный анализ ЛИМС «Гидроочистка.2» + (1 − shrinkage) · уровень "
        "обучающей выборки. Доля shrinkage отобрана на valid (ADR-012), не по "
        "удобству; persistence и константа входят в то же семейство и проиграли.",
        f"Анализ старше {MAX_BASELINE_AGE_HOURS:g} ч не используется как основание "
        "прогноза: суточный отбор ЛИМС + публикация 4 ч. Допущение сценария.",
        "«Вне диапазона обучения» решается по p01–p99 обучающей выборки: "
        "буквальные min/max содержат остановы и выбросы и ничего не ограничивают.",
        "ПАК приводится как независимый свежий сигнал, но в прогноз не входит: "
        "задержка публикации ПАК организаторами не подтверждена (ADR-009).",
        "Классы риска заданы в единицах backtest-RMSE: запас меньше одной RMSE — "
        "high, меньше двух — moderate. Внешнего обоснования у этих границ нет, "
        "это модельный порог сценария (ADR-010).",
        "Аномальность последнего анализа проверяется отдельно (ADR-014): "
        "неположительное значение или выход за полосу p01–p99 обучающей выборки "
        "помечается флагом, и агент оптимизации не строит на нём действий.",
        "Единицы P8/T11/F19 не подтверждены (ADR-007): в data_confidence они "
        "помечены units_confirmed = false, свежесть сигнала доверия к ним не даёт.",
        f"Пределы T95 ≤ {config.T95_LIMIT_C:g} °C и цетанового числа "
        f"≥ {config.CETANE_NUMBER_MIN:g} — параметры сценария по ГОСТ 32511-2013 / "
        "ГОСТ Р 52368 для К5: организаторы назвали эти показатели обязательными, "
        "но численные пределы оставили на усмотрение команды (ADR-004, ADR-015).",
        f"T95 и цетановое число берутся измерением из ЛИМС «Гидроочистка.2», а не "
        f"расчётом по формуле ВАК: ссылка Pipeline в формуле T95 не резолвится "
        f"(ADR-003). Порог устаревания у них разный — {config.T95_MAX_AGE_HOURS:g} ч "
        f"для T95 (суточный отбор) и {config.CETANE_MAX_AGE_HOURS / 24:g} суток для "
        "цетанового числа (p90 интервала между анализами на train — 41.8 суток).",
    ]
    in_range = all(item["in_training_range"] for item in training_range.values())

    baseline = snapshot.get("lims_value")
    baseline_age = snapshot.get("lims_age_hours")
    if snapshot.get("lims_missing"):
        baseline = None
    # Сера — четвёртый обязательный показатель в той же таблице, что T95, ЦЧ и
    # плотность: измеренное значение сравнивается с пределом как факт, а не
    # только его сдвинутый прогноз (ADR-019).
    sulfur_fresh = baseline is not None and (
        baseline_age is None or baseline_age <= MAX_BASELINE_AGE_HOURS
    )
    spec_indicators["sulfur"] = {
        "tag": features.TARGET_TAG,
        "value": baseline,
        "unit": "мг/кг",
        "age_hours": baseline_age,
        "max_age_hours": MAX_BASELINE_AGE_HOURS,
        "verified": bool(sulfur_fresh),
        "limit_mg_kg": config.SULFUR_LIMIT_MG_KG,
        "direction": "max",
        "margin_mg_kg": None if baseline is None else config.SULFUR_LIMIT_MG_KG - baseline,
        "detail": (
            "нет опубликованного анализа на момент выдачи"
            if baseline is None
            else f"анализ {features.TARGET_TAG}, возраст {baseline_age:.1f} ч"
            if sulfur_fresh
            else f"анализ старше {MAX_BASELINE_AGE_HOURS:g} ч (возраст {baseline_age:.1f} ч)"
        ),
    }
    measured_violations = [
        {"rule": item.rule, "detail": item.detail, "kind": item.kind}
        for item in constraints.check_measured_product(spec_indicators, baseline)
    ]
    # Контроль качества входа — отдельная названная проверка, а не побочный
    # эффект перцентильных полос (ADR-014).
    input_anomaly = anomaly.assess_lab_value(baseline, context.training_stats["lims_value"])
    if baseline is None:
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
            input_anomaly=input_anomaly,
            spec_indicators=spec_indicators,
            measured_violations=measured_violations,
            refusal=refusal,
        )
    if input_anomaly.get("anomalous") and any(
        flag.get("code") == "nonphysical" for flag in input_anomaly.get("flags", [])
    ):
        # Нефизичное измерение (≤ 0 или не число) — не основание для прогноза:
        # 0.3·(−5) + 0.7·8.6 = 4.5 «мг/кг» с риском low — это не прогноз, а
        # артефакт формулы (ADR-019).
        refusal = {
            "reason": (
                f"последний анализ ЛИМС нефизичен ({baseline!r} мг/кг): "
                + "; ".join(flag["detail"] for flag in input_anomaly.get("flags", []))
            ),
            "rule": "прогноз на нефизичном измерении не является обоснованным прогнозом (ADR-014)",
        }
        return QualityAssessment(
            issue_time=str(context.issue_time),
            forecast=[],
            in_training_range=in_range,
            training_range=training_range,
            data_confidence=confidence,
            assumptions=assumptions,
            input_anomaly=input_anomaly,
            spec_indicators=spec_indicators,
            measured_violations=measured_violations,
            refusal=refusal,
        )
    if baseline_age is not None and baseline_age > MAX_BASELINE_AGE_HOURS:
        refusal = {
            "reason": (
                f"последний анализ ЛИМС старше {MAX_BASELINE_AGE_HOURS:g} ч "
                f"(возраст {baseline_age:.1f} ч)"
            ),
            "rule": "прогноз на устаревшем анализе не является обоснованным прогнозом",
        }
        return QualityAssessment(
            issue_time=str(context.issue_time),
            forecast=[],
            in_training_range=in_range,
            training_range=training_range,
            data_confidence=confidence,
            assumptions=assumptions,
            input_anomaly=input_anomaly,
            spec_indicators=spec_indicators,
            measured_violations=measured_violations,
            refusal=refusal,
        )

    forecast = []
    for horizon in sorted(context.backtest):
        metrics = context.backtest[horizon].get(BASELINE_MODEL)
        if metrics is None:
            continue
        shrinkage = metrics.get("baseline_shrinkage")
        level = metrics.get("baseline_level_mg_kg")
        if shrinkage is None or level is None:
            raise ValueError(
                f"в backtest нет параметров baseline {BASELINE_MODEL}: "
                "пересоберите build/forecast_report.csv"
            )
        # Прогноз воспроизводится по параметрам, выбранным в backtest, и не
        # подбирается заново внутри сценария (ADR-012).
        predicted = shrinkage * baseline + (1.0 - shrinkage) * level
        margin = config.SULFUR_LIMIT_MG_KG - predicted
        # Измеренное нарушение серы перекрывает прогноз: прогноз 9.2 при
        # анализе 10.5 — не «риск high», а факт нарушения (ADR-019).
        sulfur_violated = any(item["rule"] == "sulfur" for item in measured_violations)
        forecast.append(
            {
                "horizon_hours": horizon,
                "model": BASELINE_MODEL,
                "baseline_shrinkage": shrinkage,
                "baseline_level_mg_kg": level,
                "last_published_sulfur_mg_kg": baseline,
                "predicted_sulfur_mg_kg": predicted,
                "spec_limit_mg_kg": config.SULFUR_LIMIT_MG_KG,
                "margin_mg_kg": margin,
                "backtest_rmse_mg_kg": metrics["rmse"],
                "backtest_mae_mg_kg": metrics["mae"],
                "backtest_medae_mg_kg": metrics["medae"],
                "backtest_n": metrics["n"],
                "margin_in_rmse": margin / metrics["rmse"],
                "risk_exceed_spec": (
                    RISK_MEASURED_VIOLATION if sulfur_violated else _risk(margin, metrics["rmse"])
                ),
                "in_training_range": in_range,
                "evidence_type": metrics["evidence_type"],
            }
        )
    if not forecast:
        raise ValueError(
            f"в backtest нет строк {BASELINE_MODEL}: пересоберите build/forecast_report.csv"
        )
    return QualityAssessment(
        issue_time=str(context.issue_time),
        forecast=forecast,
        in_training_range=in_range,
        training_range=training_range,
        data_confidence=confidence,
        assumptions=assumptions,
        input_anomaly=input_anomaly,
        spec_indicators=spec_indicators,
        measured_violations=measured_violations,
    )
