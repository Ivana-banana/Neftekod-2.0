"""Агент оптимизации: кандидатные воздействия и выбор среди допустимых.

Роль описана в `.claude/agents/optimization.md` и `docs/tz/tz_neftekod.md`, §3–§4.

Каждый кандидат несёт машиночитаемое поле `evidence_type`:

- `backtested` — на отложенной по времени истории проверена **ошибка
  прогноза**, а не последствие действия. Такую подпись имеет право нести
  только сохранение текущего режима: там прогноз и есть весь вывод, ничего
  сверх измеренного не утверждается. История при этом не была чистым режимом
  «ничего не меняем», так что даже здесь backtest говорит о происхождении
  числа, а не о безопасности режима.
- `simulated_unvalidated` — модельная оценка эффекта ещё не случавшегося
  воздействия. Симулятора установки нет, историей такая оценка не
  подтверждается по построению.

Смешивать эти два числа нельзя (CLAUDE.md, «Верификация и честность»).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .. import config, constraints, features, forecast
from .context import ScenarioContext
from . import quality as quality_module
from .quality import QualityAssessment
from .reliability import ReliabilityAssessment

# Шаг уставки как доля ширины диапазона, выданного агентом надёжности.
# Абсолютный шаг в «градусах» или «т/ч» задать нельзя: единицы сигналов не
# подтверждены (ADR-007). См. ADR-010.
SETPOINT_STEP_FRACTION = 0.05

# Машиночитаемые причины отклонения: разбирать русскую строку подстрокой —
# не способ классификации ни для теста, ни для интерфейса оператора.
REJECTION_CODES = (
    "hard_constraint",
    "insufficient_margin",
    "unvalidated_effect_model",
    "regime_inadmissible",
)


@dataclass(frozen=True)
class BlendingScenario:
    """Модельная постановка блендинга — данных по рецептурам нет (ADR-004).

    Все поля ниже являются допущениями сценария и печатаются рядом с
    результатом. Жёсткая проверка «сумма долей = 100%» действует для них
    наравне с реальными данными.
    """

    base_component: str = "hydrotreated_diesel"
    cut_component: str = "straight_run_240_350"
    cut_sulfur_mg_kg: float = 4000.0
    cut_relative_price: float = 0.85
    fractions: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)

    def mock_notice(self) -> str:
        return (
            f"[MOCKED: блендинг модельный — рецептур и спецификации в пакете нет (ADR-004); "
            f"сера компонента {self.cut_component} принята {self.cut_sulfur_mg_kg:g} мг/кг, "
            f"его относительная цена {self.cut_relative_price:g} от цены товарного ДТ]"
        )


@dataclass(frozen=True)
class EffectModel:
    """Линейная чувствительность серы к уставкам и её собственный backtest."""

    name: str
    horizon_hours: float
    alpha: float
    telemetry_lag_hours: float
    sensitivity: dict
    test_rmse: float
    baseline_model: str
    baseline_test_rmse: float

    @property
    def beats_baseline(self) -> bool:
        """Выигрывает ли модель эффекта у выбранного baseline (ADR-010, §3).

        Сравнение идёт с тем baseline, который отобран на valid и используется
        агентом качества (ADR-012), а не с persistence: иначе действие могло бы
        пройти, проигрывая прогнозу, на котором стоит вся оценка риска.
        """
        return self.test_rmse < self.baseline_test_rmse

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "horizon_hours": self.horizon_hours,
            "alpha": self.alpha,
            "telemetry_lag_hours": self.telemetry_lag_hours,
            "sensitivity_mg_kg_per_unit": self.sensitivity,
            "test_rmse": self.test_rmse,
            "baseline_model": self.baseline_model,
            "baseline_test_rmse": self.baseline_test_rmse,
            "beats_baseline": self.beats_baseline,
            "evidence_type": "simulated_unvalidated",
        }


def fit_effect_model(context: ScenarioContext, horizon_hours: float = 0.0) -> EffectModel:
    """Обучить модель эффекта уставок на train и приложить её backtest.

    Гиперпараметры не подбираются заново: берутся те, что выбраны на
    fit/valid в scripts/evaluate_forecast.py. Модель обучается только на
    данных до границы ADR-006 — сценарий стоит в test-периоде.
    """
    metrics = context.backtest[horizon_hours]["ridge_telemetry"]
    baseline = context.backtest[horizon_hours][quality_module.BASELINE_MODEL]
    frame = forecast.build_prediction_frame(
        context.long_format,
        horizon_hours,
        telemetry=context.telemetry,
        telemetry_lag_hours=metrics["telemetry_lag_hours"],
    )
    train = frame[pd.to_datetime(frame["target_time"]) <= pd.Timestamp(config.TRAIN_END)]
    model = forecast.fit_ridge(train, metrics["alpha"])
    names = list(model.feature_names)
    sensitivity = {}
    for tag in features.TELEMETRY_TAGS:
        position = names.index(f"{tag}_value")
        # coefficients[0] — свободный член; признаки стандартизованы на fit,
        # поэтому чувствительность в исходных единицах сигнала = coef / scale.
        sensitivity[tag] = float(model.coefficients[position + 1] / model.scales[position])
    return EffectModel(
        name="ridge_telemetry",
        horizon_hours=horizon_hours,
        alpha=float(metrics["alpha"]),
        telemetry_lag_hours=float(metrics["telemetry_lag_hours"]),
        sensitivity=sensitivity,
        test_rmse=float(metrics["rmse"]),
        baseline_model=quality_module.BASELINE_MODEL,
        baseline_test_rmse=float(baseline["rmse"]),
    )


@dataclass(frozen=True)
class OptimizationProposal:
    issue_time: str
    candidates: list
    recommended_id: str | None
    decision_rules: list
    mock_notices: list
    refusal: dict | None = None
    agent: str = field(default="optimization", init=False)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "issue_time": self.issue_time,
            "refusal": self.refusal,
            "recommended_id": self.recommended_id,
            "decision_rules": self.decision_rules,
            "mock_notices": self.mock_notices,
            "candidates": self.candidates,
        }

    def recommended(self) -> dict | None:
        for candidate in self.candidates:
            if candidate["id"] == self.recommended_id:
                return candidate
        return None


def _candidate(
    *,
    identifier: str,
    kind: str,
    description: str,
    recommendation: dict,
    objective: dict,
    effect_model: dict | None,
) -> dict:
    violations = constraints.validate_recommendation(recommendation)
    return {
        "id": identifier,
        "kind": kind,
        "description": description,
        "recommendation": recommendation,
        "objective": objective,
        "effect_model": effect_model,
        "hard_constraint_violations": [
            {"rule": violation.rule, "detail": violation.detail} for violation in violations
        ],
        "admissible": not violations,
        "rejection_reasons": [
            f"жёсткое ограничение {violation.rule}: {violation.detail}" for violation in violations
        ],
        "rejection_codes": ["hard_constraint"] * len(violations),
    }


def propose(
    context: ScenarioContext,
    quality: QualityAssessment,
    reliability: ReliabilityAssessment,
    *,
    blending: BlendingScenario | None = None,
    effect_model: EffectModel | None = None,
) -> OptimizationProposal:
    """Сформировать кандидатов и выбрать среди допустимых.

    Агент не выдаёт результат, если агент качества отказался от прогноза или
    агент надёжности отказался от оценки: оптимизировать без прогноза серы и
    без границ режима — это и есть «рискованная догадка» из ТЗ, §4.
    """
    blending = blending or BlendingScenario()
    issue_time = str(context.issue_time)
    mock_notices = [blending.mock_notice()]
    decision_rules = [
        "Жёсткие ограничения проверяются у каждого кандидата через "
        "constraints.validate_recommendation; нарушение отклоняет кандидата целиком.",
        "Кандидат, меняющий режим, дополнительно обязан оставить запас по сере "
        "не меньше backtest-RMSE прогноза на худшем из горизонтов действия "
        "(0–3 ч), а не на ближайшем: иначе выигрыш лежит внутри ошибки. "
        "Сохранение текущего режима от этого требования освобождено — это не "
        "действие, а статус-кво.",
        "Кандидат, эффект которого предсказан моделью, не выигравшей у "
        "выбранного baseline прогноза на отложенной истории, отклоняется: его "
        "прогноз серы не является основанием для действия.",
        "Среди допустимых выбирается минимальный стоимостный прокси; при "
        "равенстве остаётся сохранение текущего режима.",
    ]

    if quality.refusal is not None or reliability.refusal is not None:
        reasons = [
            item["reason"]
            for item in (quality.refusal, reliability.refusal)
            if item is not None
        ]
        return OptimizationProposal(
            issue_time=issue_time,
            candidates=[],
            recommended_id=None,
            decision_rules=decision_rules,
            mock_notices=mock_notices,
            refusal={
                "reason": "; ".join(reasons),
                "rule": "без прогноза качества или без границ режима кандидаты не строятся",
            },
        )

    horizon = min(context.backtest)
    baseline_forecast = next(item for item in quality.forecast if item["horizon_hours"] == horizon)
    baseline_sulfur = baseline_forecast["predicted_sulfur_mg_kg"]
    # Порог действия берётся по ХУДШЕМУ горизонту, а не по ближайшему:
    # рекомендация действует на 0–3 часа (ответы организаторов), и настраивать
    # «гейт» на самую выгодную из измеренных ошибок значит пропускать впритык
    # то, что на реальном горизонте действия не проходит. См. ADR-010.
    worst = max(quality.forecast, key=lambda item: item["backtest_rmse_mg_kg"])
    forecast_rmse = worst["backtest_rmse_mg_kg"]
    forecast_rmse_horizon = worst["horizon_hours"]
    current_setpoints = {
        tag: context.snapshot[f"{tag}_value"] for tag in features.TELEMETRY_TAGS
    }
    allowed_ranges = reliability.allowed_ranges

    candidates = [
        _candidate(
            identifier="hold",
            kind="no_action",
            description="Сохранить текущий режим и текущий состав блендинга",
            recommendation={
                "action": "hold",
                "predicted_sulfur_mg_kg": baseline_sulfur,
                "evidence_type": "backtested",
                "blend_fractions": {blending.base_component: 1.0, blending.cut_component: 0.0},
                "setpoints": dict(current_setpoints),
                "allowed_ranges": allowed_ranges,
            },
            objective={
                "cut_component_share": 0.0,
                "cost_proxy_rel": 1.0,
                "severity_index": reliability.severity_index,
                "sulfur_margin_mg_kg": config.SULFUR_LIMIT_MG_KG - baseline_sulfur,
            },
            effect_model=None,
        )
    ]

    for fraction in blending.fractions:
        if fraction <= 0:
            continue
        # Массовый баланс серы: S = Σ wᵢSᵢ (ADR-004, внешний ориентир).
        # Это заявленная модель смешения, а не подогнанная под историю регрессия.
        blended = (1.0 - fraction) * baseline_sulfur + fraction * blending.cut_sulfur_mg_kg
        candidates.append(
            _candidate(
                identifier=f"blend_{blending.cut_component}_{fraction:g}",
                kind="blend",
                description=(
                    f"Вовлечь {fraction:.0%} компонента {blending.cut_component} "
                    "в товарный ДТ (модельный блендинг)"
                ),
                recommendation={
                    "action": "adjust_blend",
                    "predicted_sulfur_mg_kg": blended,
                    "evidence_type": "simulated_unvalidated",
                    "blend_fractions": {
                        blending.base_component: 1.0 - fraction,
                        blending.cut_component: fraction,
                    },
                    "setpoints": dict(current_setpoints),
                    "allowed_ranges": allowed_ranges,
                },
                objective={
                    "cut_component_share": fraction,
                    "cost_proxy_rel": (1.0 - fraction) + fraction * blending.cut_relative_price,
                    "severity_index": reliability.severity_index,
                    "sulfur_margin_mg_kg": config.SULFUR_LIMIT_MG_KG - blended,
                },
                effect_model={
                    "name": "sulfur_mass_balance",
                    "formula": "S_blend = (1-w)·S_base + w·S_cut",
                    "evidence_type": "simulated_unvalidated",
                    "note": "балансная модель смешения; исторически не проверялась",
                },
            )
        )

    if effect_model is not None:
        for tag in features.TELEMETRY_TAGS:
            low, high = allowed_ranges[tag]
            step = SETPOINT_STEP_FRACTION * (high - low)
            for direction in (-1.0, 1.0):
                delta = direction * step
                setpoints = dict(current_setpoints)
                setpoints[tag] = current_setpoints[tag] + delta
                predicted = baseline_sulfur + effect_model.sensitivity[tag] * delta
                candidates.append(
                    _candidate(
                        identifier=f"setpoint_{tag}_{'plus' if direction > 0 else 'minus'}",
                        kind="setpoint",
                        description=(
                            f"Изменить уставку {tag} на {delta:+.4g} "
                            f"({SETPOINT_STEP_FRACTION:.0%} ширины допустимого диапазона)"
                        ),
                        recommendation={
                            "action": "adjust_setpoint",
                            "predicted_sulfur_mg_kg": predicted,
                            "evidence_type": "simulated_unvalidated",
                            "blend_fractions": {
                                blending.base_component: 1.0,
                                blending.cut_component: 0.0,
                            },
                            "setpoints": setpoints,
                            "allowed_ranges": allowed_ranges,
                        },
                        objective={
                            "cut_component_share": 0.0,
                            "cost_proxy_rel": 1.0,
                            "severity_index": reliability.severity_index,
                            "sulfur_margin_mg_kg": config.SULFUR_LIMIT_MG_KG - predicted,
                        },
                        effect_model=effect_model.to_dict(),
                    )
                )

    for candidate in candidates:
        reasons = list(candidate["rejection_reasons"])
        codes = list(candidate["rejection_codes"])
        if candidate["kind"] != "no_action":
            margin = candidate["objective"]["sulfur_margin_mg_kg"]
            if margin < forecast_rmse:
                reasons.append(
                    f"запас по сере {margin:.3g} мг/кг меньше backtest-RMSE прогноза "
                    f"{forecast_rmse:.3g} мг/кг на худшем горизонте {forecast_rmse_horizon:g} ч: "
                    "выигрыш неотличим от ошибки"
                )
                codes.append("insufficient_margin")
            model = candidate["effect_model"]
            if model is not None and model.get("name") == "ridge_telemetry" and not model["beats_baseline"]:
                reasons.append(
                    f"модель эффекта {model['name']} проиграла baseline "
                    f"{model['baseline_model']} на test "
                    f"(RMSE {model['test_rmse']:.4g} против {model['baseline_test_rmse']:.4g}): "
                    "её прогноз серы не является основанием для действия"
                )
                codes.append("unvalidated_effect_model")
            if not reliability.regime_admissible:
                reasons.append(
                    f"агент надёжности не считает режим допустимым "
                    f"(severity_index {reliability.severity_index:.3g})"
                )
                codes.append("regime_inadmissible")
        candidate["rejection_reasons"] = reasons
        candidate["rejection_codes"] = codes
        candidate["admissible"] = not reasons

    admissible = [item for item in candidates if item["admissible"]]
    if not admissible:
        return OptimizationProposal(
            issue_time=issue_time,
            candidates=candidates,
            recommended_id=None,
            decision_rules=decision_rules,
            mock_notices=mock_notices,
            refusal={
                "reason": "ни один кандидат не прошёл проверки",
                "rule": "при отсутствии допустимого варианта рекомендация не выдаётся (ТЗ, §4)",
            },
        )
    best = min(
        admissible,
        key=lambda item: (item["objective"]["cost_proxy_rel"], item["kind"] != "no_action", item["id"]),
    )
    return OptimizationProposal(
        issue_time=issue_time,
        candidates=candidates,
        recommended_id=best["id"],
        decision_rules=decision_rules,
        mock_notices=mock_notices,
    )


def build_submission(
    context: ScenarioContext,
    quality: QualityAssessment,
    reliability: ReliabilityAssessment,
    proposal: OptimizationProposal,
    *,
    blending: BlendingScenario | None = None,
) -> dict:
    """Сформировать пакет для оркестратора: вход, рекомендация, ограничения.

    Сознательно не передаются reasoning агентов, перебор кандидатов и их
    обоснования: оркестратор обязан воспроизвести допустимость сам, а не
    поверить исполнителю (CLAUDE.md, «Мультиагентность»).
    """
    blending = blending or BlendingScenario()
    recommended = proposal.recommended()
    horizon = min(context.backtest)
    baseline = (
        next((item for item in quality.forecast if item["horizon_hours"] == horizon), None)
        if quality.forecast
        else None
    )
    # Оркестратор получает тот же порог, каким пользовался оптимизатор:
    # худший горизонт действия, а не ближайший (см. ADR-010).
    worst = (
        max(quality.forecast, key=lambda item: item["backtest_rmse_mg_kg"])
        if quality.forecast
        else None
    )
    return {
        "input": {
            "scenario": context.scenario,
            "issue_time": str(context.issue_time),
            "components": {
                blending.base_component: {
                    "sulfur_mg_kg": None if baseline is None else baseline["predicted_sulfur_mg_kg"],
                    "source": "прогноз агента качества (сдвинутый persistence по ЛИМС «Гидроочистка.2»)",
                    "evidence_type": None if baseline is None else baseline["evidence_type"],
                },
                blending.cut_component: {
                    "sulfur_mg_kg": blending.cut_sulfur_mg_kg,
                    "source": blending.mock_notice(),
                    "evidence_type": "simulated_unvalidated",
                },
            },
            "current_setpoints": {
                tag: context.snapshot[f"{tag}_value"] for tag in features.TELEMETRY_TAGS
            },
            # Текущий состав — модельный статус-кво (ADR-004). Оркестратор
            # определяет по нему «ничего не меняем» по числам, а не по ярлыку.
            "current_blend_fractions": {
                blending.base_component: 1.0,
                blending.cut_component: 0.0,
            },
        },
        "recommendation": None if recommended is None else recommended["recommendation"],
        "applied_constraints": {
            "sulfur_limit_mg_kg": config.SULFUR_LIMIT_MG_KG,
            "blend_fraction_sum": config.BLEND_FRACTION_SUM,
            "allowed_ranges": reliability.allowed_ranges,
            "allowed_ranges_basis": reliability.allowed_ranges_basis,
            "forecast_backtest_rmse_mg_kg": None if worst is None else worst["backtest_rmse_mg_kg"],
            "forecast_backtest_rmse_horizon_hours": None if worst is None else worst["horizon_hours"],
            "quality_risk_exceed_spec": None if baseline is None else baseline["risk_exceed_spec"],
            "regime_admissible": reliability.regime_admissible,
        },
        "refusals": {
            "quality": quality.refusal,
            "reliability": reliability.refusal,
            "optimization": proposal.refusal,
        },
    }
