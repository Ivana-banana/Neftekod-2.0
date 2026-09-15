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

from .. import config, constraints, features, forecast, spec, splits
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
    "anomalous_input",
    "unmodelled_spec_effect",
    "unverified_spec_indicator",
)


@dataclass(frozen=True)
class BlendingScenario:
    """Постановка блендинга: доли модельные (ADR-004), сера компонента — измеренная.

    Рецептур и спецификации в пакете нет, поэтому состав и цена остаются
    допущениями и печатаются рядом с результатом. Сера дешёвого компонента
    берётся из реального анализа сырья гидроочистки как-of момента выдачи —
    это первое звено цепочки, названной самими организаторами (ADR-013).
    Жёсткая проверка «сумма долей = 100%» действует наравне с реальными данными.
    """

    base_component: str = "hydrotreated_diesel"
    cut_component: str = "straight_run_240_350"
    # Сера дешёвого компонента — реальный анализ, а не придуманное число.
    cut_sulfur_tag: str = "LIMS:Гидроочистка.1.Mass.Sulfur"
    cut_sulfur_unit_factor: float = 10000.0  # % масс. → мг/кг
    # Отбор сырья идёт медианно раз в 3 суток, p90 интервала — 31 сутки.
    # Анализ старше 31 суток не используется: см. ADR-013.
    max_cut_sulfur_age_hours: float = 31 * 24.0
    cut_relative_price: float = 0.85
    # Цетановое число дешёвого компонента не измеряется нигде в пакете —
    # единственный показатель спецификации, целиком взятый из допущения
    # (spec.CUT_CETANE_NUMBER, ADR-015).
    cut_cetane_number: float = spec.CUT_CETANE_NUMBER
    # Модельный текущий состав: половина доли, физически допустимой по сере
    # при типовой (медианной по train) сере базового компонента. См. ADR-013.
    current_cut_fraction: float = 7.4e-5
    # Сетка кандидатов задаётся в долях физически допустимого максимума:
    # абсолютный шаг в процентах бессмысленен, когда максимум измеряется
    # сотыми долями процента (ADR-013).
    candidate_fractions_of_max: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

    def mock_notice(self) -> str:
        return (
            f"[MOCKED: состав блендинга модельный — рецептур и спецификации в пакете нет "
            f"(ADR-004); текущая доля {self.cut_component} принята "
            f"{self.current_cut_fraction:.4%}, его относительная цена "
            f"{self.cut_relative_price:g} от цены товарного ДТ. Сера этого компонента "
            f"не выдумана: берётся из анализа {self.cut_sulfur_tag}, T95 — из "
            f"{spec.CUT_T95_TAG}. Цетановое число этого компонента в пакете не "
            f"измеряется и принято {self.cut_cetane_number:g} — допущение]"
        )


def resolve_cut_sulfur(context: ScenarioContext, blending: BlendingScenario) -> dict:
    """Сера дешёвого компонента по реальному анализу сырья на момент выдачи.

    Используется только анализ, опубликованный к моменту выдачи. Слишком
    старый анализ не берётся: неизвестная сера компонента не считается
    допустимой (ТЗ, §4), а состав смеси без неё не определён.
    """
    frame = context.long_format
    part = frame[frame["tag"].eq(blending.cut_sulfur_tag)].copy()
    part["value"] = pd.to_numeric(part["value"], errors="coerce")
    part = part.dropna(subset=["value"])
    published = splits.available_history(part, context.issue_time)
    if published.empty:
        return {
            "value_mg_kg": None,
            "refusal": {
                "reason": f"нет опубликованного анализа {blending.cut_sulfur_tag} на момент выдачи",
                "rule": "состав смеси без серы компонента не определён (ТЗ, §4)",
            },
        }
    last = published.sort_values("timestamp").iloc[-1]
    age_hours = (context.issue_time - pd.Timestamp(last["timestamp"])).total_seconds() / 3600.0
    resolved = {
        "tag": blending.cut_sulfur_tag,
        "value_mg_kg": float(last["value"]) * blending.cut_sulfur_unit_factor,
        "value_declared": float(last["value"]),
        "unit_declared": str(last.get("unit_declared")),
        "sample_time": str(pd.Timestamp(last["timestamp"])),
        "available_at": str(pd.Timestamp(last["available_at"])),
        "age_hours": age_hours,
        "max_age_hours": blending.max_cut_sulfur_age_hours,
        "evidence_type": "backtested",
        "refusal": None,
    }
    if age_hours > blending.max_cut_sulfur_age_hours:
        resolved["value_mg_kg"] = None
        resolved["refusal"] = {
            "reason": (
                f"анализ сырья {blending.cut_sulfur_tag} старше "
                f"{blending.max_cut_sulfur_age_hours / 24:g} суток (возраст {age_hours / 24:.1f} суток)"
            ),
            "rule": "состав смеси на устаревшем анализе компонента не определён",
        }
    return resolved


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
    blending_inputs: dict | None = None
    spec_inputs: dict | None = None
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
            "blending_inputs": self.blending_inputs,
            "spec_inputs": self.spec_inputs,
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
    # Неподтверждённый показатель не блокирует статус-кво, но блокирует любое
    # изменение; доказанное нарушение блокирует всё. Пометка кладётся в саму
    # рекомендацию у каждого кандидата: пустой список — это утверждение «всё
    # подтверждено», а не пропущенное поле. См. ADR-017.
    blocking = constraints.blocking(violations, status_quo=kind == "no_action")
    recommendation["unverified_spec_indicators"] = constraints.unverified_indicators(violations)
    return {
        "id": identifier,
        "kind": kind,
        "description": description,
        "recommendation": recommendation,
        "objective": objective,
        "effect_model": effect_model,
        "hard_constraint_violations": [
            {"rule": violation.rule, "detail": violation.detail, "kind": violation.kind}
            for violation in violations
        ],
        "admissible": not blocking,
        "rejection_reasons": [
            (
                f"обязательный показатель {violation.rule} не подтверждён: изменение "
                f"режима или состава не сертифицируется ({violation.detail})"
                if violation.kind == constraints.UNVERIFIED
                else f"жёсткое ограничение {violation.rule}: {violation.detail}"
            )
            for violation in blocking
        ],
        "rejection_codes": [
            "unverified_spec_indicator"
            if violation.kind == constraints.UNVERIFIED
            else "hard_constraint"
            for violation in blocking
        ],
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
        "Если контроль качества входа пометил последний анализ как аномальный "
        "(ADR-014), действия не рекомендуются вовсе: сохранение режима остаётся "
        "допустимым, а менять режим по подозрительному измерению нельзя.",
        "Кандидат, эффект которого предсказан моделью, не выигравшей у "
        "выбранного baseline прогноза на отложенной истории, отклоняется: его "
        "прогноз серы не является основанием для действия.",
        "Доли дешёвого компонента перебираются в долях физически допустимого "
        "по сере максимума w_max = (S_предел − S_база) / (S_компонент − S_база); "
        "сера компонента берётся из реального анализа сырья (ADR-013).",
        "Обязательных показателей товарного ДТ три, а не один (ADR-004): сера "
        f"≤ {config.SULFUR_LIMIT_MG_KG:g} мг/кг, T95 ≤ {config.T95_LIMIT_C:g} °C и "
        f"цетановое число ≥ {config.CETANE_NUMBER_MIN:g}; к ним добавлена плотность "
        f"{config.DENSITY_MIN_KG_M3:g}–{config.DENSITY_MAX_KG_M3:g} кг/м³ — "
        "нормируемый ГОСТ показатель на той же точке отбора (ADR-016). Показатель, не "
        "подтверждённый свежим анализом, не считается выполненным: любое изменение "
        "режима или состава отклоняется кодом unverified_spec_indicator, а сохранение "
        "текущего режима остаётся допустимым с явной пометкой "
        "unverified_spec_indicators (ADR-015, ADR-017). Доказанное нарушение "
        "отклоняет и сохранение режима.",
        f"Прогноз показателя сравнивается с пределом в разрешении лабораторного "
        f"входа: T95 — {config.T95_RESOLUTION_C:g} °C, цетановое число — "
        f"{config.CETANE_NUMBER_RESOLUTION:g}, плотность — "
        f"{config.DENSITY_RESOLUTION_KG_M3:g} кг/м³; превышение меньше половины шага "
        "данные не различают (ADR-017).",
        "Влияние изменения уставок P8/T11/F19 на T95 и цетановое число не "
        "моделируется: модели нет, а подставлять текущее измерение как прогноз "
        "после действия нельзя. Поэтому кандидат, меняющий уставки, отклоняется "
        "кодом unmodelled_spec_effect независимо от результата по сере (ADR-015).",
        "Среди допустимых выбирается минимальный стоимостный прокси; при "
        "равенстве остаётся сохранение текущего режима. Экономика решает только "
        "между уже допустимыми вариантами и никогда не делает вариант допустимым.",
    ]

    cut = resolve_cut_sulfur(context, blending)
    # T95 дешёвого компонента — тоже измерение, а не допущение (ADR-015).
    cut_t95 = spec.resolve_cut_t95(context.long_format, context.issue_time)
    # Плотность дешёвого компонента тоже измеряется (ADR-016); он плотнее
    # продукта, поэтому его доля двигает плотность смеси к верхней границе.
    cut_density = spec.resolve_cut_density(context.long_format, context.issue_time)
    if quality.refusal is not None or reliability.refusal is not None or cut["refusal"] is not None:
        reasons = [
            item["reason"]
            for item in (quality.refusal, reliability.refusal, cut["refusal"])
            if item is not None
        ]
        return OptimizationProposal(
            issue_time=issue_time,
            candidates=[],
            recommended_id=None,
            decision_rules=decision_rules,
            mock_notices=mock_notices,
            blending_inputs=cut,
            spec_inputs={"cut_t95": cut_t95.to_dict(), "cut_density": cut_density.to_dict()},
            refusal={
                "reason": "; ".join(reasons),
                "rule": (
                    "без прогноза качества, без границ режима или без серы компонента "
                    "смеси кандидаты не строятся"
                ),
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

    cut_sulfur = cut["value_mg_kg"]
    current_fraction = blending.current_cut_fraction
    # Измеренные показатели продукта приходят от агента качества, а не
    # добываются здесь заново: оркестратор сверяет пакет именно с его выводом
    # (ADR-010, §6а). Неподтверждённый показатель приходит со value = None.
    product_indicators = quality.spec_indicators or {}

    # Массовый баланс серы: S = Σ wᵢSᵢ (ADR-004, внешний ориентир) — заявленная
    # модель смешения, а не подогнанная под историю регрессия.
    def blend_sulfur(fraction: float) -> float:
        return (1.0 - fraction) * baseline_sulfur + fraction * cut_sulfur

    # Показатели спецификации смешиваются по своим моделям (ADR-015):
    # T95 — массовым балансом, цетановое число — нелинейным blend index.
    # Значение компонента, не подтверждённое анализом, равно None, и тогда
    # показатель смеси тоже None — жёсткая проверка отклонит такого кандидата.
    component_t95 = {
        blending.base_component: (product_indicators.get("t95") or {}).get("value"),
        blending.cut_component: cut_t95.value,
    }
    component_cetane = {
        blending.base_component: (product_indicators.get("cetane_number") or {}).get("value"),
        blending.cut_component: blending.cut_cetane_number,
    }
    component_density = {
        blending.base_component: (product_indicators.get("density") or {}).get("value"),
        blending.cut_component: cut_density.value,
    }

    def blend_t95(fraction: float):
        return spec.blend_t95(fractions_of(fraction), component_t95)

    def blend_cetane(fraction: float):
        return spec.blend_cetane_number(fractions_of(fraction), component_cetane)

    def blend_density(fraction: float):
        return spec.blend_density(fractions_of(fraction), component_density)

    def spec_margins(fraction: float) -> dict:
        t95 = blend_t95(fraction)
        cetane = blend_cetane(fraction)
        density = blend_density(fraction)
        return {
            "t95_c": t95,
            "t95_margin_c": None if t95 is None else config.T95_LIMIT_C - t95,
            "cetane_number": cetane,
            "cetane_margin": None if cetane is None else cetane - config.CETANE_NUMBER_MIN,
            "density_kg_m3": density,
            # Диапазон двусторонний, поэтому запас — до ближайшей границы.
            "density_margin_kg_m3": (
                None
                if density is None
                else min(
                    config.DENSITY_MAX_KG_M3 - density,
                    density - config.DENSITY_MIN_KG_M3,
                )
            ),
        }

    def cost_proxy(fraction: float) -> float:
        return (1.0 - fraction) + fraction * blending.cut_relative_price

    def fractions_of(fraction: float) -> dict:
        return {blending.base_component: 1.0 - fraction, blending.cut_component: fraction}

    # Физический максимум доли дешёвого компонента по сере:
    # w_max = (S_предел − S_база) / (S_компонент − S_база). Сетка кандидатов
    # задаётся в долях этого максимума: абсолютный шаг в процентах бессмыслен,
    # когда максимум измеряется сотыми долями процента (ADR-013).
    max_fraction = (
        min(1.0, max(0.0, (config.SULFUR_LIMIT_MG_KG - baseline_sulfur) / (cut_sulfur - baseline_sulfur)))
        if cut_sulfur > baseline_sulfur
        else 0.0
    )

    candidates = [
        _candidate(
            identifier="hold",
            kind="no_action",
            description=(
                f"Сохранить текущий режим и текущий состав блендинга "
                f"(доля {blending.cut_component} {current_fraction:.4%})"
            ),
            recommendation={
                "action": "hold",
                "predicted_sulfur_mg_kg": blend_sulfur(current_fraction),
                "predicted_t95_c": blend_t95(current_fraction),
                "predicted_cetane_number": blend_cetane(current_fraction),
                "predicted_density_kg_m3": blend_density(current_fraction),
                # Даже «ничего не менять» здесь — не чисто измеренное число:
                # прогноз базового компонента backtested, но текущая доля
                # дешёвого компонента модельная (ADR-004), поэтому сера смеси
                # получена моделью смешения. См. ADR-013.
                "evidence_type": (
                    "backtested" if current_fraction == 0.0 else "simulated_unvalidated"
                ),
                "blend_fractions": fractions_of(current_fraction),
                "setpoints": dict(current_setpoints),
                "allowed_ranges": allowed_ranges,
            },
            objective={
                "cut_component_share": current_fraction,
                "cost_proxy_rel": cost_proxy(current_fraction),
                "severity_index": reliability.severity_index,
                "sulfur_margin_mg_kg": config.SULFUR_LIMIT_MG_KG - blend_sulfur(current_fraction),
                **spec_margins(current_fraction),
            },
            effect_model=None,
        )
    ]

    for share in blending.candidate_fractions_of_max:
        fraction = share * max_fraction
        if abs(fraction - current_fraction) <= 1e-12:
            continue  # это статус-кво: он уже представлен кандидатом hold
        candidates.append(
            _candidate(
                identifier=f"blend_{blending.cut_component}_f{share:g}",
                kind="blend",
                description=(
                    f"{'Повысить' if fraction > current_fraction else 'Снизить'} долю "
                    f"{blending.cut_component} с {current_fraction:.4%} до {fraction:.4%} "
                    f"({share:.0%} физически допустимого по сере максимума {max_fraction:.4%})"
                ),
                recommendation={
                    "action": "adjust_blend",
                    "predicted_sulfur_mg_kg": blend_sulfur(fraction),
                    "predicted_t95_c": blend_t95(fraction),
                    "predicted_cetane_number": blend_cetane(fraction),
                    "predicted_density_kg_m3": blend_density(fraction),
                    "evidence_type": "simulated_unvalidated",
                    "blend_fractions": fractions_of(fraction),
                    "setpoints": dict(current_setpoints),
                    "allowed_ranges": allowed_ranges,
                },
                objective={
                    "cut_component_share": fraction,
                    "cost_proxy_rel": cost_proxy(fraction),
                    "severity_index": reliability.severity_index,
                    "sulfur_margin_mg_kg": config.SULFUR_LIMIT_MG_KG - blend_sulfur(fraction),
                    **spec_margins(fraction),
                },
                effect_model={
                    "name": "sulfur_mass_balance",
                    "formula": "S_blend = (1-w)·S_base + w·S_cut",
                    "cut_sulfur_mg_kg": cut_sulfur,
                    "cut_sulfur_source": cut["tag"],
                    "cut_sulfur_age_hours": cut["age_hours"],
                    "max_fraction_by_sulfur": max_fraction,
                    "t95_model": "T95_blend = Σ wᵢ·T95ᵢ (массовый баланс, ADR-015)",
                    "cut_t95_c": cut_t95.value,
                    "cut_t95_source": cut_t95.tag,
                    "cut_t95_age_hours": cut_t95.age_hours,
                    "cetane_model": (
                        f"CN_blend = (Σ wᵢ·CNᵢ^{spec.CETANE_BLEND_INDEX_EXPONENT:g})"
                        f"^(1/{spec.CETANE_BLEND_INDEX_EXPONENT:g}) (blend index, ADR-004)"
                    ),
                    "cut_cetane_number": blending.cut_cetane_number,
                    "cut_cetane_basis": spec.CUT_CETANE_BASIS,
                    "density_model": "1/ρ_blend = Σ wᵢ/ρᵢ (аддитивность объёмов, ADR-016)",
                    "cut_density_kg_m3": cut_density.value,
                    "cut_density_source": cut_density.tag,
                    "cut_density_age_hours": cut_density.age_hours,
                    "evidence_type": "simulated_unvalidated",
                    "note": (
                        "балансная модель смешения на измеренной сере компонента; "
                        "сама доля исторически не проверялась"
                    ),
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
                # Изменение уставки двигает серу базового компонента; состав
                # смеси при этом остаётся текущим, поэтому сера товарного ДТ
                # считается тем же массовым балансом.
                shifted_base = baseline_sulfur + effect_model.sensitivity[tag] * delta
                predicted = (
                    (1.0 - current_fraction) * shifted_base + current_fraction * cut_sulfur
                )
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
                            # Модели влияния уставок на T95 и ЦЧ нет: сюда
                            # кладётся состояние до действия, а сам кандидат
                            # ниже отклоняется кодом unmodelled_spec_effect.
                            "predicted_t95_c": blend_t95(current_fraction),
                            "predicted_cetane_number": blend_cetane(current_fraction),
                            "predicted_density_kg_m3": blend_density(current_fraction),
                            "evidence_type": "simulated_unvalidated",
                            "blend_fractions": fractions_of(current_fraction),
                            "setpoints": setpoints,
                            "allowed_ranges": allowed_ranges,
                        },
                        objective={
                            "cut_component_share": current_fraction,
                            "cost_proxy_rel": cost_proxy(current_fraction),
                            "severity_index": reliability.severity_index,
                            "sulfur_margin_mg_kg": config.SULFUR_LIMIT_MG_KG - predicted,
                            **spec_margins(current_fraction),
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
            if candidate["recommendation"].get("action") == "adjust_setpoint":
                reasons.append(
                    "влияние изменения уставки на T95 и цетановое число не "
                    "моделируется: обязательные показатели после действия "
                    "неизвестны, поэтому действие не сертифицируется (ADR-015)"
                )
                codes.append("unmodelled_spec_effect")
            if (quality.input_anomaly or {}).get("anomalous"):
                detail = "; ".join(
                    item["detail"] for item in quality.input_anomaly.get("flags", [])
                )
                reasons.append(
                    f"контроль качества входа пометил последний анализ как аномальный: {detail}"
                )
                codes.append("anomalous_input")
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
            blending_inputs=cut,
            spec_inputs={"cut_t95": cut_t95.to_dict(), "cut_density": cut_density.to_dict()},
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
        blending_inputs=cut,
        spec_inputs={"cut_t95": cut_t95.to_dict(), "cut_density": cut_density.to_dict()},
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
    cut = proposal.blending_inputs or resolve_cut_sulfur(context, blending)
    cut_t95 = ((proposal.spec_inputs or {}).get("cut_t95")) or spec.resolve_cut_t95(
        context.long_format, context.issue_time
    ).to_dict()
    cut_density = ((proposal.spec_inputs or {}).get("cut_density")) or spec.resolve_cut_density(
        context.long_format, context.issue_time
    ).to_dict()
    product = quality.spec_indicators or {}
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
                    # T95 и ЦЧ базового компонента — измерения того же агента
                    # качества; оркестратор пересчитывает по ним смесь сам.
                    "t95_c": (product.get("t95") or {}).get("value"),
                    "cetane_number": (product.get("cetane_number") or {}).get("value"),
                    "density_kg_m3": (product.get("density") or {}).get("value"),
                },
                blending.cut_component: {
                    "sulfur_mg_kg": cut["value_mg_kg"],
                    "source": (
                        f"анализ {blending.cut_sulfur_tag}, отбор {cut.get('sample_time')}, "
                        f"возраст {cut.get('age_hours')} ч"
                    ),
                    "evidence_type": "backtested",
                    "t95_c": cut_t95.get("value"),
                    "cetane_number": blending.cut_cetane_number,
                    "density_kg_m3": cut_density.get("value"),
                },
            },
            "current_setpoints": {
                tag: context.snapshot[f"{tag}_value"] for tag in features.TELEMETRY_TAGS
            },
            # Текущий состав — модельный статус-кво (ADR-004). Оркестратор
            # определяет по нему «ничего не меняем» по числам, а не по ярлыку.
            "current_blend_fractions": {
                blending.base_component: 1.0 - blending.current_cut_fraction,
                blending.cut_component: blending.current_cut_fraction,
            },
        },
        "recommendation": None if recommended is None else recommended["recommendation"],
        "applied_constraints": {
            "sulfur_limit_mg_kg": config.SULFUR_LIMIT_MG_KG,
            "t95_limit_c": config.T95_LIMIT_C,
            "cetane_number_min": config.CETANE_NUMBER_MIN,
            "density_min_kg_m3": config.DENSITY_MIN_KG_M3,
            "density_max_kg_m3": config.DENSITY_MAX_KG_M3,
            "blend_fraction_sum": config.BLEND_FRACTION_SUM,
            "allowed_ranges": reliability.allowed_ranges,
            "allowed_ranges_basis": reliability.allowed_ranges_basis,
            "forecast_backtest_rmse_mg_kg": None if worst is None else worst["backtest_rmse_mg_kg"],
            "forecast_backtest_rmse_horizon_hours": None if worst is None else worst["horizon_hours"],
            # Класс риска берётся с того же худшего горизонта, что и порог:
            # смешивать «риск с ближайшего» и «ошибку с худшего» нельзя —
            # пакет тогда описывает два разных момента. Дефект найден
            # оркестратором на сценарии 4 (ADR-010, §3б).
            "quality_risk_exceed_spec": None if worst is None else worst["risk_exceed_spec"],
            "regime_admissible": reliability.regime_admissible,
        },
        "refusals": {
            "quality": quality.refusal,
            "reliability": reliability.refusal,
            "optimization": proposal.refusal,
        },
    }
