"""Оркестратор: разрешение конфликта целей и adversarial-проверка результата.

Роль описана в `.claude/agents/orchestrator.md` и `docs/tz/tz_neftekod.md`, §3.

Оркестратор намеренно не получает reasoning исполнителей, перебор кандидатов
и их обоснования — только вход, саму рекомендацию и применённые ограничения.
Слова агента оптимизации «ограничения соблюдены» здесь не имеют силы: пока
оркестратор не воспроизвёл допустимость своими руками на числах самой
рекомендации, она отклоняется (CLAUDE.md, «Мультиагентность»).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .. import config, constraints

SUBMISSION_KEYS = frozenset({"input", "recommendation", "applied_constraints", "refusals"})
BLOCKING_QUALITY_RISKS = frozenset({"high", "spec_violation_predicted"})


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
            "received_keys": self.received_keys,
        }


def _assert_contract(submission: dict) -> None:
    """Пакет от агента оптимизации содержит только разрешённые части.

    Проверка не косметическая: если сюда просочатся кандидаты, метрики
    перебора или обоснования исполнителя, adversarial-ревью превратится в
    пересказ чужого вывода.
    """
    extra = set(submission) - SUBMISSION_KEYS
    if extra:
        raise ValueError(
            f"оркестратор не принимает поля {sorted(extra)}: разрешены только "
            f"{sorted(SUBMISSION_KEYS)}"
        )
    missing = {"input", "applied_constraints"} - set(submission)
    if missing:
        raise ValueError(f"в пакете нет обязательных частей: {sorted(missing)}")


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


def review(submission: dict) -> Decision:
    """Проверить рекомендацию заново и вынести вердикт.

    Возвращает либо утверждённую рекомендацию, либо явный отказ с причиной.
    «Почти допустимо» вердиктом не является.
    """
    _assert_contract(submission)
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
            "уставки проверены против allowed_ranges из применённых ограничений, "
            "а не против копии внутри рекомендации; нарушений: " + str(len(range_violations)),
        )
    )
    for violation in range_violations:
        reasons.append(f"уставка вне применённого диапазона: {violation.detail}")

    action = recommendation.get("action")
    risk = applied.get("quality_risk_exceed_spec")
    rmse = applied.get("forecast_backtest_rmse_mg_kg")
    if risk in BLOCKING_QUALITY_RISKS:
        # Утверждённое «ничего не менять» не означает «всё спокойно»: запас по
        # сере меньше проверенной историей ошибки прогноза, и оператор должен
        # увидеть это рядом с рекомендацией, а не в отдельном файле.
        warnings.append(
            f"агент качества оценил риск нарушения спецификации как {risk}: запас по сере "
            f"меньше backtest-RMSE прогноза ({rmse!r} мг/кг). Режим сохраняется, но "
            "контроль серы ослаблять нельзя"
        )
    if action != "hold":
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
            f"Рекомендация «{action}» утверждена: все жёсткие проверки выполнены "
            "оркестратором заново на числах самой рекомендации, сера смеси пересчитана "
            "по входу, уставки сверены с применёнными диапазонами. "
            f"Тип обоснования: {recommendation.get('evidence_type')}."
        ),
        received_keys=sorted(submission),
        warnings=warnings,
    )
