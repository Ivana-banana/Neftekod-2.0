"""Оркестратор: разрешение конфликта целей и adversarial-проверка результата.

Роль описана в `.claude/agents/orchestrator.md` и `docs/tz/tz_neftekod.md`, §3.

Оркестратор намеренно не получает reasoning исполнителей, перебор кандидатов
и их обоснования — только вход, саму рекомендацию и применённые ограничения.
Слова агента оптимизации «ограничения соблюдены» здесь не имеют силы: пока
оркестратор не воспроизвёл допустимость своими руками, она отклоняется
(CLAUDE.md, «Мультиагентность»).

Ключевой урок первого прогона роли: пакет для проверки собирает сам агент
оптимизации, поэтому сверка двух полей внутри пакета ничего не доказывает —
подделать можно оба сразу. Найдены и закрыты четыре такие дыры: согласованно
расширенные диапазоны, заниженная сера компонента во входе, «hold», меняющий
уставки, и снятое предупреждение о риске качества. Поэтому `review()`
обязательно получает `sources` — выводы агентов качества и надёжности и
снимок входа, сделанные независимо от агента оптимизации, — и сверяет пакет
с ними, а не пакет с самим собой. См. ADR-010.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .. import config, constraints, spec

SUBMISSION_KEYS = frozenset({"input", "recommendation", "applied_constraints", "refusals"})
# Вложенные allowlist: плоской проверки верхнего уровня мало — обоснование
# исполнителя спокойно проходило внутри `recommendation`. См. ADR-010.
INPUT_KEYS = frozenset(
    {"scenario", "issue_time", "components", "current_setpoints", "current_blend_fractions"}
)
RECOMMENDATION_KEYS = frozenset(
    {
        "action",
        "predicted_sulfur_mg_kg",
        "predicted_t95_c",
        "predicted_cetane_number",
        "predicted_density_kg_m3",
        "evidence_type",
        "blend_fractions",
        "setpoints",
        "allowed_ranges",
        "unverified_spec_indicators",
    }
)
APPLIED_CONSTRAINT_KEYS = frozenset(
    {
        "sulfur_limit_mg_kg",
        "t95_limit_c",
        "cetane_number_min",
        "density_min_kg_m3",
        "density_max_kg_m3",
        "blend_fraction_sum",
        "allowed_ranges",
        "allowed_ranges_basis",
        "forecast_backtest_rmse_mg_kg",
        "forecast_backtest_rmse_horizon_hours",
        "quality_risk_exceed_spec",
        "regime_admissible",
    }
)
# Независимые источники: выводы ролей, снимок входа, показатели компонентов
# смеси, разрешённые из данных без участия агента оптимизации, и модельный
# текущий состав из постановки блендинга (ADR-019: до этого показатели
# дешёвого компонента и «текущие доли» брались из самого пакета).
SOURCE_KEYS = ("quality", "reliability", "snapshot", "components", "current_blend_fractions")
BLOCKING_QUALITY_RISKS = frozenset({"high", "spec_violation_predicted", "spec_violation_measured"})
COMPONENT_FIELDS = ("sulfur_mg_kg", "t95_c", "cetane_number", "density_kg_m3")
SETPOINT_TOLERANCE = 1e-9
# Показатель спецификации: поле компонента во входе → ключ вывода агента качества.
SPEC_COMPONENT_KEYS = {"t95_c": "t95", "cetane_number": "cetane_number", "density_kg_m3": "density"}
SPEC_TITLES = {"t95": "T95", "cetane_number": "цетановое число", "density": "плотность"}

# Что эта проверка не доказывает. Не устная оговорка на демонстрации, а часть
# машиночитаемого вывода: см. ADR-010 и `.claude/agents/orchestrator.md`.
REVIEW_LIMITS = (
    "Сам прогноз серы оркестратор перепроверить не может: другого источника, "
    "кроме агента качества, для него нет. Сверяется, что число в пакете — "
    "действительно то, которое выдал агент качества, а не то, что удобно.",
    "Состав блендинга модельный (ADR-004): «текущие доли» не сверяются с "
    "данными, потому что данных по рецептурам в пакете нет.",
    "Проверка подтверждает допустимость режима по заявленным ограничениям, "
    "а не безопасность режима в цехе.",
    "T95, цетановое число и плотность смеси пересчитываются по моделям "
    "смешения (массовый баланс, blend index и аддитивность объёмов — "
    "ADR-015, ADR-016), а сами модели историей не проверялись: воспроизводимо "
    "— арифметика, а не физика смешения. Цетановое число дешёвого компонента "
    "вообще не измеряется (допущение).",
    "Сохранение режима при неподтверждённом показателе утверждается с пометкой "
    "(ADR-017). Показатели базового компонента сверяются с агентом качества, "
    "показатели дешёвого компонента и текущий состав — с источниками, "
    "разрешёнными из данных и постановки блендинга без участия агента "
    "оптимизации (ADR-019); сами анализы компонента оркестратор не перепроверяет.",
    "Текущая уставка вне модельного диапазона p05–p95 у статус-кво даёт "
    "предупреждение, а не отказ: диапазон — допущение для эксперимента, а не "
    "промышленный предел (ТЗ, §4); допустимость режима оценивает агент "
    "надёжности (ADR-019).",
)


@dataclass(frozen=True)
class Decision:
    issue_time: str
    verdict: str
    recommendation: dict | None
    checks: list
    rejection_reasons: list
    explanation: str
    received_keys: list
    warnings: list = field(default_factory=list)
    limits: list = field(default_factory=lambda: list(REVIEW_LIMITS))
    agent: str = field(default="orchestrator", init=False)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "issue_time": self.issue_time,
            "verdict": self.verdict,
            "explanation": self.explanation,
            "warnings": self.warnings,
            "rejection_reasons": self.rejection_reasons,
            "recommendation": self.recommendation,
            "independent_checks": self.checks,
            "limits": self.limits,
            "received_keys": self.received_keys,
        }


def _assert_allowed(name: str, payload, allowed: frozenset) -> None:
    if not isinstance(payload, dict):
        return
    extra = set(payload) - allowed
    if extra:
        raise ValueError(
            f"оркестратор не принимает поля {sorted(extra)} в {name}: разрешены только "
            f"{sorted(allowed)}"
        )


def _assert_contract(submission: dict, sources: dict) -> None:
    """Пакет содержит только разрешённые части — на всех уровнях."""
    _assert_allowed("пакете", submission, SUBMISSION_KEYS)
    missing = {"input", "applied_constraints"} - set(submission)
    if missing:
        raise ValueError(f"в пакете нет обязательных частей: {sorted(missing)}")
    _assert_allowed("input", submission.get("input"), INPUT_KEYS)
    _assert_allowed("recommendation", submission.get("recommendation"), RECOMMENDATION_KEYS)
    _assert_allowed("applied_constraints", submission.get("applied_constraints"), APPLIED_CONSTRAINT_KEYS)
    absent = [key for key in SOURCE_KEYS if key not in (sources or {})]
    if absent:
        raise ValueError(
            f"нет независимых выводов ролей {absent}: без них ограничения нечем воспроизвести"
        )


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"check": name, "passed": passed, "detail": detail}


def _recompute_blend_sulfur(components: dict, fractions: dict) -> tuple[float | None, str]:
    unknown = [name for name in fractions if name not in components]
    if unknown:
        return None, f"состав ссылается на неизвестные компоненты: {sorted(unknown)}"
    total = 0.0
    for name, weight in fractions.items():
        sulfur = components[name].get("sulfur_mg_kg")
        if not isinstance(sulfur, (int, float)) or isinstance(sulfur, bool) or math.isnan(sulfur):
            return None, f"у компонента {name} нет числовой серы: {sulfur!r}"
        total += float(weight) * float(sulfur)
    return total, "S = Σ wᵢ·Sᵢ, пересчитано оркестратором по входу"


def _component_values(components: dict, fractions: dict, key: str) -> tuple[dict | None, str]:
    """Значения показателя по компонентам входа; None, если хоть одно неизвестно."""
    values = {}
    # Компонент с нулевой долей в смеси отсутствует — его анализ не нужен,
    # как и в `spec._contributing` (ADR-019).
    for name, weight in (fractions or {}).items():
        if _close(weight, 0.0, 1e-15):
            continue
        item = components.get(name) or {}
        value = item.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or math.isnan(value):
            return None, f"у компонента {name} нет числового значения {key}: {value!r}"
        values[name] = float(value)
    return values, f"значения {key} взяты из компонентов входа"


def _verify_blended(
    name: str,
    claimed,
    components: dict,
    fractions: dict,
    key: str,
    model,
    checks: list,
    reasons: list,
    unit: str,
) -> None:
    """Пересчитать показатель смеси своей копией модели и сверить с заявленным.

    Оркестратор не берёт число у агента оптимизации на слово: модель смешения
    вызывается здесь заново на компонентах входа. Неизвестный компонент — не
    «пропустим проверку», а причина отклонения (ADR-015).
    """
    values, detail = _component_values(components, fractions, key)
    if values is None and claimed is None:
        # Показатель не подтверждён ни во входе, ни в рекомендации — это
        # согласовано. Допустим ли такой вариант, решает `constraints.blocking`
        # по статус-кво, а не эта сверка (ADR-017).
        checks.append(_check(name, True, f"показатель не подтверждён, заявлено None: {detail}"))
        return
    if values is None:
        checks.append(_check(name, False, detail))
        reasons.append(f"показатель смеси не воспроизводится: {detail}")
        return
    recomputed = model(fractions, values)
    if recomputed is None:
        checks.append(_check(name, False, f"модель смешения не дала значения: {detail}"))
        reasons.append(f"показатель смеси не воспроизводится по модели: {detail}")
        return
    matches = (
        isinstance(claimed, (int, float))
        and not isinstance(claimed, bool)
        and math.isclose(recomputed, float(claimed), rel_tol=1e-9, abs_tol=1e-6)
    )
    checks.append(
        _check(name, matches, f"{detail}: получено {recomputed:.8g} {unit}, заявлено {claimed!r}")
    )
    if not matches:
        reasons.append(
            f"заявленное значение {claimed!r} {unit} не совпадает с пересчитанным "
            f"{recomputed:.8g} {unit}"
        )


def _close(left, right, tolerance: float = SETPOINT_TOLERANCE) -> bool:
    if not isinstance(left, (int, float)) or isinstance(left, bool):
        return False
    if not isinstance(right, (int, float)) or isinstance(right, bool):
        return False
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _same_mapping(left, right, tolerance: float = SETPOINT_TOLERANCE) -> bool:
    left = left or {}
    right = right or {}
    if set(left) != set(right):
        return False
    return all(_close(left[key], right[key], tolerance) for key in left)


def _same_bounds(left, right) -> bool:
    """Диапазоны совпадают покомпонентно; сериализация через JSON допускается."""
    left = left or {}
    right = right or {}
    if set(left) != set(right):
        return False
    for key in left:
        first, second = list(left[key]), list(right[key])
        if len(first) != len(second) or not all(
            _close(one, other, 1e-9) for one, other in zip(first, second)
        ):
            return False
    return True


def _worst_forecast(quality: dict) -> dict | None:
    forecast = (quality or {}).get("forecast") or []
    if not forecast:
        return None
    return max(forecast, key=lambda item: item["backtest_rmse_mg_kg"])


def _cross_check_with_roles(
    submission: dict, sources: dict, checks: list, reasons: list
) -> None:
    """Сверить пакет с тем, что роли выдали независимо от агента оптимизации."""
    applied = submission["applied_constraints"]
    scenario_input = submission["input"]
    quality = sources["quality"]
    reliability = sources["reliability"]
    snapshot = sources["snapshot"]

    same = _same_bounds(applied.get("allowed_ranges"), reliability.get("allowed_ranges"))
    checks.append(
        _check(
            "applied_ranges_match_reliability",
            same,
            "диапазоны сверены с выводом агента надёжности, а не с копией внутри пакета",
        )
    )
    if not same:
        reasons.append(
            "применённые диапазоны не совпадают с тем, что выдал агент надёжности: "
            "согласованно подменить обе копии внутри пакета недостаточно"
        )

    admissible_matches = applied.get("regime_admissible") == reliability.get("regime_admissible")
    checks.append(
        _check(
            "regime_admissible_matches_reliability",
            admissible_matches,
            f"в пакете {applied.get('regime_admissible')!r}, "
            f"у агента надёжности {reliability.get('regime_admissible')!r}",
        )
    )
    if not admissible_matches:
        reasons.append("допустимость режима в пакете не совпадает с выводом агента надёжности")

    worst = _worst_forecast(quality)
    if worst is None:
        checks.append(_check("applied_risk_matches_quality", False, "агент качества не дал прогноза"))
        reasons.append("в пакете есть рекомендация, но агент качества прогноза не выдавал")
    else:
        risk_matches = applied.get("quality_risk_exceed_spec") == worst["risk_exceed_spec"]
        rmse_matches = _close(
            applied.get("forecast_backtest_rmse_mg_kg"), worst["backtest_rmse_mg_kg"], 1e-9
        )
        checks.append(
            _check(
                "applied_risk_matches_quality",
                risk_matches and rmse_matches,
                f"риск {applied.get('quality_risk_exceed_spec')!r} против "
                f"{worst['risk_exceed_spec']!r}, RMSE "
                f"{applied.get('forecast_backtest_rmse_mg_kg')!r} против "
                f"{worst['backtest_rmse_mg_kg']!r} на худшем горизонте "
                f"{worst['horizon_hours']!r} ч",
            )
        )
        if not (risk_matches and rmse_matches):
            reasons.append(
                "оценка риска или ошибка прогноза в пакете не совпадают с выводом "
                "агента качества — смягчить порог подменой поля нельзя"
            )
        base_sulfur = worst["predicted_sulfur_mg_kg"]
        declared = [
            name
            for name, item in (scenario_input.get("components") or {}).items()
            if _close(item.get("sulfur_mg_kg"), base_sulfur, 1e-9)
        ]
        checks.append(
            _check(
                "component_sulfur_matches_quality",
                bool(declared),
                f"прогноз агента качества {base_sulfur!r} мг/кг найден среди компонентов входа: "
                f"{declared or 'нет'}",
            )
        )
        if not declared:
            reasons.append(
                f"ни у одного компонента входа сера не равна прогнозу агента качества "
                f"({base_sulfur!r} мг/кг): пересчёт баланса не спасает, если занижено "
                "само исходное число"
            )
        # Показатели спецификации базового компонента — измерения агента
        # качества. Без этой сверки измеренное нарушение можно выдать за «нет
        # анализа» и получить утверждённый hold по ADR-017.
        indicators = quality.get("spec_indicators") or {}
        measured = {
            field: (indicators.get(indicator) or {}).get("value")
            for field, indicator in SPEC_COMPONENT_KEYS.items()
        }
        components = scenario_input.get("components") or {}
        spec_matches = bool(declared) and all(
            all(
                (components[name].get(field) is None and value is None)
                or _close(components[name].get(field), value, 1e-9)
                for field, value in measured.items()
            )
            for name in declared
        )
        checks.append(
            _check(
                "component_spec_matches_quality",
                spec_matches,
                f"T95, ЦЧ и плотность базового компонента сверены с агентом качества: {measured}",
            )
        )
        if not spec_matches:
            reasons.append(
                "показатели спецификации базового компонента во входе не совпадают с "
                "измерениями агента качества — нарушение нельзя выдать за «нет анализа»"
            )

    # Показатели дешёвого компонента и текущий состав — против источников,
    # собранных без агента оптимизации (ADR-019): иначе занизить серу
    # компонента или объявить новую долю «текущей» можно было бы в самом пакете.
    independent = sources.get("components") or {}
    components = scenario_input.get("components") or {}
    mismatched = []
    for name, expected in independent.items():
        declared = components.get(name) or {}
        for field in COMPONENT_FIELDS:
            if field not in expected:
                continue
            left, right = declared.get(field), expected.get(field)
            if not ((left is None and right is None) or _close(left, right, 1e-9)):
                mismatched.append(f"{name}.{field}: в пакете {left!r}, независимо {right!r}")
    checks.append(
        _check(
            "component_values_match_sources",
            not mismatched,
            "показатели компонентов смеси сверены с независимо разрешёнными анализами: "
            + ("; ".join(mismatched) if mismatched else f"совпадают для {sorted(independent)}"),
        )
    )
    if mismatched:
        reasons.append(
            "показатели компонентов смеси во входе не совпадают с независимыми источниками: "
            + "; ".join(mismatched)
        )
    fractions_match = _same_mapping(
        scenario_input.get("current_blend_fractions"), sources.get("current_blend_fractions"), 1e-12
    )
    checks.append(
        _check(
            "current_blend_fractions_match_sources",
            fractions_match,
            "текущий состав во входе сверен с постановкой блендинга, а не с копией в пакете",
        )
    )
    if not fractions_match:
        reasons.append(
            "текущий состав во входе не совпадает с постановкой блендинга: объявить новую "
            "долю «текущей» и выдать изменение за статус-кво нельзя"
        )

    issue_matches = str(scenario_input.get("issue_time")) == str(quality.get("issue_time"))
    checks.append(
        _check(
            "issue_time_matches_quality",
            issue_matches,
            f"в пакете {scenario_input.get('issue_time')!r}, у агента качества {quality.get('issue_time')!r}",
        )
    )
    if not issue_matches:
        reasons.append("момент выдачи в пакете не совпадает с моментом выдачи агента качества")
    nonphysical = [
        name
        for name, item in components.items()
        if isinstance(item, dict)
        and item.get("sulfur_mg_kg") is not None
        and not (
            isinstance(item.get("sulfur_mg_kg"), (int, float))
            and not isinstance(item.get("sulfur_mg_kg"), bool)
            and math.isfinite(item["sulfur_mg_kg"])
            and item["sulfur_mg_kg"] >= 0.0
        )
    ]
    checks.append(
        _check(
            "component_sulfur_physical",
            not nonphysical,
            "сера каждого компонента входа — конечное неотрицательное число"
            + (f"; нарушают: {nonphysical}" if nonphysical else ""),
        )
    )
    if nonphysical:
        reasons.append(f"нефизичная сера у компонентов входа: {nonphysical}")

    current = {
        key: snapshot.get(f"{key}_value") for key in (scenario_input.get("current_setpoints") or {})
    }
    setpoints_match = _same_mapping(scenario_input.get("current_setpoints"), current, 1e-9)
    checks.append(
        _check(
            "current_setpoints_match_snapshot",
            setpoints_match,
            "текущие уставки во входе сверены со снимком данных на момент выдачи",
        )
    )
    if not setpoints_match:
        reasons.append("текущие уставки во входе не совпадают со снимком данных на момент выдачи")


def review(submission: dict, *, sources: dict) -> Decision:
    """Проверить рекомендацию заново и вынести вердикт.

    `sources` — выводы агентов качества и надёжности и снимок входа, собранные
    независимо от агента оптимизации. Без них проверка вырождается в сверку
    пакета с самим собой, поэтому аргумент обязателен.

    Возвращает либо утверждённую рекомендацию, либо явный отказ с причиной.
    «Почти допустимо» вердиктом не является.
    """
    _assert_contract(submission, sources)
    issue_time = str(submission["input"].get("issue_time"))
    applied = submission["applied_constraints"]
    recommendation = submission.get("recommendation")
    refusals = submission.get("refusals") or {}

    if recommendation is None:
        reasons = [
            f"{role}: {item['reason']}"
            for role, item in refusals.items()
            if isinstance(item, dict) and item.get("reason")
        ]
        return Decision(
            issue_time=issue_time,
            verdict="no_admissible_recommendation",
            recommendation=None,
            checks=[],
            rejection_reasons=reasons or ["агент оптимизации не предложил допустимого варианта"],
            explanation=(
                "Допустимой рекомендации нет. Причина: "
                + ("; ".join(reasons) if reasons else "допустимых кандидатов не осталось")
                + ". Рискованный вариант вместо отказа не выдаётся (ТЗ, §4)."
            ),
            received_keys=sorted(submission),
        )

    checks: list = []
    reasons: list = []
    warnings: list = []

    # Статус-кво определяется по фактическим числам, а не по ярлыку `action`:
    # иначе слово «hold» отключает весь gate ниже. См. ADR-010. Нужен уже здесь:
    # от него зависит, блокирует ли неподтверждённый показатель (ADR-017).
    scenario_input = submission["input"]
    unchanged_setpoints = _same_mapping(
        recommendation.get("setpoints"), scenario_input.get("current_setpoints")
    )
    unchanged_blend = _same_mapping(
        recommendation.get("blend_fractions"), scenario_input.get("current_blend_fractions")
    )
    is_status_quo = unchanged_setpoints and unchanged_blend

    violations = constraints.validate_recommendation(recommendation)
    blocking = constraints.blocking(violations, status_quo=is_status_quo)
    checks.append(
        _check(
            "hard_constraints_recomputed",
            not blocking,
            "constraints.validate_recommendation выполнен оркестратором заново на числах "
            "рекомендации; блокирующих нарушений: " + str(len(blocking))
            + f", всего: {len(violations)}, статус-кво: {is_status_quo}",
        )
    )
    for violation in blocking:
        if violation.kind == constraints.UNVERIFIED:
            reasons.append(
                f"обязательный показатель {violation.rule} не подтверждён, а рекомендация "
                "меняет режим или состав — изменение не сертифицируется (ADR-017)"
            )
        else:
            reasons.append(f"жёсткое ограничение {violation.rule}: {violation.detail}")
    # Текущая уставка вне модельного диапазона у статус-кво — предупреждение
    # оператору, а не отказ (ADR-019); у изменения она уже в blocking выше.
    for detail in constraints.out_of_model_range(violations) if is_status_quo else []:
        warnings.append(
            f"текущая уставка вне модельного диапазона p05–p95 обучающей выборки: {detail}. "
            "Режим сохраняется; диапазон — допущение для эксперимента, а не промышленный "
            "предел, допустимость режима оценивает агент надёжности (ADR-019)"
        )

    # Доказанное нарушение по измеренному показателю продукта пересчитывается
    # оркестратором по выводу агента качества, а не берётся из пакета: анализ
    # 10.5 мг/кг блокирует всё, включая сохранение режима (ADR-019).
    quality_source = sources.get("quality") or {}
    measured_indicators = quality_source.get("spec_indicators") or {}
    measured = constraints.check_measured_product(
        measured_indicators, (measured_indicators.get("sulfur") or {}).get("value")
    )
    checks.append(
        _check(
            "measured_product_within_spec",
            not measured,
            "измеренные сера, T95, ЦЧ и плотность продукта из вывода агента качества "
            "сравнены с пределами; доказанных нарушений: " + str(len(measured)),
        )
    )
    for violation in measured:
        reasons.append(
            f"доказанное нарушение спецификации {violation.rule}: {violation.detail} — "
            "блокирует и сохранение режима (ADR-019)"
        )

    # Пометку оркестратор выводит сам и сверяет с пакетом: снять её, чтобы
    # hold выглядел полностью подтверждённым, нельзя.
    unverified = constraints.unverified_indicators(violations)
    declared_unverified = recommendation.get("unverified_spec_indicators")
    marked = isinstance(declared_unverified, list) and sorted(declared_unverified) == unverified
    checks.append(
        _check(
            "unverified_indicators_declared",
            marked,
            f"неподтверждённые по пересчёту: {unverified}, заявлено: {declared_unverified!r}",
        )
    )
    if not marked:
        reasons.append(
            f"пометка unverified_spec_indicators {declared_unverified!r} не совпадает с "
            f"пересчитанной оркестратором {unverified}"
        )
    indicators = (sources.get("quality") or {}).get("spec_indicators") or {}
    field_of = {indicator: field for field, indicator in SPEC_COMPONENT_KEYS.items()}
    for violation in violations:
        if violation.kind != constraints.UNVERIFIED or not is_status_quo:
            continue
        measured = indicators.get(violation.rule) or {}
        if measured and not measured.get("verified"):
            detail = measured.get("detail")
        else:
            # Продукт измерен, значит не хватает значения у другого компонента
            # смеси — назвать его, чтобы оператор не проверял свежий анализ.
            components = scenario_input.get("components") or {}
            missing = [
                name
                for name in recommendation.get("blend_fractions") or {}
                if (components.get(name) or {}).get(field_of.get(violation.rule)) is None
            ]
            detail = (
                f"значение смеси не определено — нет подтверждённого значения у "
                f"компонента {', '.join(missing) or 'неизвестно какого'}"
            )
        warnings.append(
            f"обязательный показатель «{SPEC_TITLES.get(violation.rule, violation.rule)}» "
            f"не подтверждён: {detail}. Режим сохраняется, но соответствие по этому "
            "показателю не подтверждено, а изменение режима или состава не "
            "сертифицируется (ADR-017)"
        )

    limit = applied.get("sulfur_limit_mg_kg", config.SULFUR_LIMIT_MG_KG)
    if limit != config.SULFUR_LIMIT_MG_KG:
        reasons.append(
            f"применённый предел серы {limit} не совпадает с пределом проекта "
            f"{config.SULFUR_LIMIT_MG_KG} мг/кг"
        )
        checks.append(_check("sulfur_limit_unchanged", False, f"получено {limit}"))
    else:
        checks.append(
            _check("sulfur_limit_unchanged", True, f"{config.SULFUR_LIMIT_MG_KG} мг/кг")
        )

    for key, project_value, title in (
        ("t95_limit_c", config.T95_LIMIT_C, "предел T95"),
        ("cetane_number_min", config.CETANE_NUMBER_MIN, "минимум цетанового числа"),
        ("density_min_kg_m3", config.DENSITY_MIN_KG_M3, "нижняя граница плотности"),
        ("density_max_kg_m3", config.DENSITY_MAX_KG_M3, "верхняя граница плотности"),
    ):
        applied_value = applied.get(key, project_value)
        if applied_value != project_value:
            reasons.append(
                f"применённый {title} {applied_value} не совпадает со значением проекта "
                f"{project_value}"
            )
            checks.append(_check(f"{key}_unchanged", False, f"получено {applied_value}"))
        else:
            checks.append(_check(f"{key}_unchanged", True, str(project_value)))

    _cross_check_with_roles(submission, sources, checks, reasons)

    claimed = recommendation.get("predicted_sulfur_mg_kg")
    fractions = recommendation.get("blend_fractions")
    if fractions:
        recomputed, detail = _recompute_blend_sulfur(submission["input"].get("components", {}), fractions)
        if recomputed is None:
            reasons.append(f"сера смеси не воспроизводится: {detail}")
            checks.append(_check("blend_sulfur_recomputed", False, detail))
        else:
            matches = isinstance(claimed, (int, float)) and math.isclose(
                recomputed, float(claimed), rel_tol=1e-9, abs_tol=1e-6
            )
            checks.append(
                _check(
                    "blend_sulfur_recomputed",
                    matches,
                    f"{detail}: получено {recomputed:.6g}, заявлено {claimed!r}",
                )
            )
            if not matches:
                reasons.append(
                    f"заявленная сера {claimed!r} мг/кг не совпадает с пересчитанной "
                    f"{recomputed:.6g} мг/кг"
                )
        components = submission["input"].get("components") or {}
        _verify_blended(
            "blend_t95_recomputed",
            recommendation.get("predicted_t95_c"),
            components,
            fractions,
            "t95_c",
            spec.blend_t95,
            checks,
            reasons,
            "°C",
        )
        _verify_blended(
            "blend_cetane_recomputed",
            recommendation.get("predicted_cetane_number"),
            components,
            fractions,
            "cetane_number",
            spec.blend_cetane_number,
            checks,
            reasons,
            "ед. ЦЧ",
        )
        _verify_blended(
            "blend_density_recomputed",
            recommendation.get("predicted_density_kg_m3"),
            components,
            fractions,
            "density_kg_m3",
            spec.blend_density,
            checks,
            reasons,
            "кг/м³",
        )

    allowed = applied.get("allowed_ranges") or {}
    declared = recommendation.get("allowed_ranges") or {}
    same_ranges = {key: list(value) for key, value in declared.items()} == {
        key: list(value) for key, value in allowed.items()
    }
    checks.append(
        _check(
            "allowed_ranges_not_widened",
            same_ranges,
            "диапазоны в рекомендации сверены с применёнными ограничениями",
        )
    )
    if not same_ranges:
        reasons.append("рекомендация объявляет диапазоны, отличные от применённых ограничений")

    range_violations = constraints.check_within_ranges(recommendation.get("setpoints"), allowed)
    checks.append(
        _check(
            "setpoints_within_applied_ranges",
            not range_violations or is_status_quo,
            "уставки проверены против allowed_ranges из применённых ограничений; "
            "вне диапазона: " + str(len(range_violations))
            + (" (статус-кво — предупреждение, ADR-019)" if range_violations and is_status_quo else ""),
        )
    )
    if not is_status_quo:
        for violation in range_violations:
            reasons.append(f"уставка вне применённого диапазона: {violation.detail}")

    action = recommendation.get("action")
    if action == "hold" and not is_status_quo:
        checks.append(
            _check(
                "hold_really_holds",
                False,
                f"уставки не изменены: {unchanged_setpoints}; состав не изменён: {unchanged_blend}",
            )
        )
        reasons.append(
            "рекомендация помечена как «hold», но меняет уставки или состав — это не "
            "является сохранением режима и не освобождает от проверок действия"
        )
    elif action == "hold":
        checks.append(
            _check("hold_really_holds", True, "уставки и состав совпадают с текущими")
        )

    # Честность подписи: backtested имеет право нести только статус-кво с
    # чистым (без модельной доли) составом — прогноз и есть весь вывод.
    # Любое изменение — simulated_unvalidated по построению (ADR-019).
    evidence = recommendation.get("evidence_type")
    independent_fractions = sources.get("current_blend_fractions") or {}
    pure_status_quo = is_status_quo and all(
        _close(weight, 0.0, 1e-15) or _close(weight, 1.0, 1e-15)
        for weight in independent_fractions.values()
    )
    evidence_honest = evidence != "backtested" or pure_status_quo
    checks.append(
        _check(
            "evidence_type_honest",
            evidence_honest,
            f"evidence_type {evidence!r}; статус-кво: {is_status_quo}, "
            f"состав без модельной доли: {pure_status_quo}",
        )
    )
    if not evidence_honest:
        reasons.append(
            "подпись backtested заявлена для изменения режима/состава или для "
            "модельного состава — это оценка эффекта, а не проверенный историей прогноз"
        )

    risk = applied.get("quality_risk_exceed_spec")
    rmse = applied.get("forecast_backtest_rmse_mg_kg")
    if risk in BLOCKING_QUALITY_RISKS:
        # Утверждённое «ничего не менять» не означает «всё спокойно»: запас по
        # сере меньше проверенной историей ошибки прогноза, и оператор должен
        # увидеть это рядом с рекомендацией, а не в отдельном файле.
        warnings.append(
            f"агент качества оценил риск нарушения спецификации как {risk}: запас по сере "
            f"меньше backtest-RMSE прогноза ({rmse!r} мг/кг на худшем горизонте). "
            "Контроль серы ослаблять нельзя; следующий анализ ЛИМС обязателен"
        )
    if not is_status_quo:
        margin = (
            limit - float(claimed)
            if isinstance(claimed, (int, float)) and not isinstance(claimed, bool)
            else None
        )
        # Запас статус-кво оркестратор считает сам — по независимому текущему
        # составу и компонентам входа, уже сверенным выше (ADR-019).
        status_quo_sulfur, _ = _recompute_blend_sulfur(
            submission["input"].get("components", {}),
            sources.get("current_blend_fractions") or {},
        )
        status_quo_margin = None if status_quo_sulfur is None else limit - status_quo_sulfur
        covered = margin is not None and isinstance(rmse, (int, float)) and margin >= rmse
        # Действие, не уменьшающее запас по сере относительно статус-кво, не
        # рискованнее сохранения режима — тот же принцип, что у агента
        # оптимизации (ADR-019).
        not_worse = (
            margin is not None
            and status_quo_margin is not None
            and margin >= status_quo_margin - 1e-12
        )
        if risk in BLOCKING_QUALITY_RISKS:
            checks.append(
                _check(
                    "action_allowed_under_quality_risk",
                    covered or not_worse,
                    f"риск качества {risk}; запас {margin!r} против backtest-RMSE {rmse!r}, "
                    f"запас статус-кво {status_quo_margin!r}",
                )
            )
            if not (covered or not_worse):
                reasons.append(
                    f"агент качества оценил риск нарушения спецификации как {risk}, "
                    f"а действие не оставляет запаса больше ошибки прогноза "
                    f"({margin!r} против {rmse!r}) и уменьшает запас относительно "
                    f"статус-кво ({status_quo_margin!r}) — конфликт целей решён в пользу качества"
                )
        if applied.get("regime_admissible") is False:
            checks.append(
                _check("action_allowed_under_regime_risk", False, "режим признан недопустимым")
            )
            reasons.append("агент надёжности не признал режим допустимым, а рекомендация меняет его")

    if reasons:
        return Decision(
            issue_time=issue_time,
            verdict="rejected",
            recommendation=None,
            checks=checks,
            rejection_reasons=reasons,
            explanation=(
                "Рекомендация отклонена: " + "; ".join(reasons) + ". "
                "Воспроизвести «ограничения соблюдены» независимо не удалось, "
                "поэтому вариант не пропускается с оговоркой."
            ),
            received_keys=sorted(submission),
            warnings=warnings,
        )
    return Decision(
        issue_time=issue_time,
        verdict="approved",
        recommendation=recommendation,
        checks=checks,
        rejection_reasons=[],
        explanation=(
            f"Рекомендация «{action}» утверждена: жёсткие проверки выполнены оркестратором "
            "заново на числах рекомендации, ограничения сверены с независимыми выводами "
            "агентов качества и надёжности, показатели смеси — сера, T95, цетановое "
            "число и плотность — пересчитаны по входу своими моделями смешения, "
            "статус-кво определён по фактическим уставкам, а не по ярлыку. "
            f"Тип обоснования: {recommendation.get('evidence_type')}."
        ),
        received_keys=sorted(submission),
        warnings=warnings,
    )
