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

from .. import config, constraints

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
        "evidence_type",
        "blend_fractions",
        "setpoints",
        "allowed_ranges",
    }
)
APPLIED_CONSTRAINT_KEYS = frozenset(
    {
        "sulfur_limit_mg_kg",
        "blend_fraction_sum",
        "allowed_ranges",
        "allowed_ranges_basis",
        "forecast_backtest_rmse_mg_kg",
        "forecast_backtest_rmse_horizon_hours",
        "quality_risk_exceed_spec",
        "regime_admissible",
    }
)
SOURCE_KEYS = ("quality", "reliability", "snapshot")
BLOCKING_QUALITY_RISKS = frozenset({"high", "spec_violation_predicted"})
SETPOINT_TOLERANCE = 1e-9

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

    violations = constraints.validate_recommendation(recommendation)
    checks.append(
        _check(
            "hard_constraints_recomputed",
            not violations,
            "constraints.validate_recommendation выполнен оркестратором заново на числах "
            "рекомендации; нарушений: " + str(len(violations)),
        )
    )
    for violation in violations:
        reasons.append(f"жёсткое ограничение {violation.rule}: {violation.detail}")

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
            not range_violations,
            "уставки проверены против allowed_ranges из применённых ограничений; "
            "нарушений: " + str(len(range_violations)),
        )
    )
    for violation in range_violations:
        reasons.append(f"уставка вне применённого диапазона: {violation.detail}")

    # Статус-кво определяется по фактическим числам, а не по ярлыку `action`:
    # иначе слово «hold» отключает весь gate ниже. См. ADR-010.
    scenario_input = submission["input"]
    unchanged_setpoints = _same_mapping(
        recommendation.get("setpoints"), scenario_input.get("current_setpoints")
    )
    unchanged_blend = _same_mapping(
        recommendation.get("blend_fractions"), scenario_input.get("current_blend_fractions")
    )
    is_status_quo = unchanged_setpoints and unchanged_blend
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

    risk = applied.get("quality_risk_exceed_spec")
    rmse = applied.get("forecast_backtest_rmse_mg_kg")
    if risk in BLOCKING_QUALITY_RISKS:
        # Утверждённое «ничего не менять» не означает «всё спокойно»: запас по
        # сере меньше проверенной историей ошибки прогноза, и оператор должен
        # увидеть это рядом с рекомендацией, а не в отдельном файле.
        warnings.append(
            f"агент качества оценил риск нарушения спецификации как {risk}: запас по сере "
            f"меньше backtest-RMSE прогноза ({rmse!r} мг/кг на худшем горизонте). "
            "Режим сохраняется, но контроль серы ослаблять нельзя"
        )
    if not is_status_quo:
        margin = (
            limit - float(claimed)
            if isinstance(claimed, (int, float)) and not isinstance(claimed, bool)
            else None
        )
        covered = margin is not None and isinstance(rmse, (int, float)) and margin >= rmse
        if risk in BLOCKING_QUALITY_RISKS:
            checks.append(
                _check(
                    "action_allowed_under_quality_risk",
                    covered,
                    f"риск качества {risk}; запас {margin!r} против backtest-RMSE {rmse!r}",
                )
            )
            if not covered:
                reasons.append(
                    f"агент качества оценил риск нарушения спецификации как {risk}, "
                    f"а действие не оставляет запаса больше ошибки прогноза "
                    f"({margin!r} против {rmse!r}) — конфликт целей решён в пользу качества"
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
            "агентов качества и надёжности, сера смеси пересчитана по входу, статус-кво "
            "определён по фактическим уставкам, а не по ярлыку. "
            f"Тип обоснования: {recommendation.get('evidence_type')}."
        ),
        received_keys=sorted(submission),
        warnings=warnings,
    )
