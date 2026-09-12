"""Карточка оператора по итогам прогона сценария.

Ничего не считает: берёт только то, что уже выдали агенты и оркестратор.
Любое число, которого нет в их выводе, в карточку попасть не может — иначе
объяснение на демонстрации перестаёт соответствовать проверенному пакету.
"""

from __future__ import annotations

from .. import config


def recommendation_markdown(title, decision, quality, reliability, proposal) -> str:
    lines = [
        f"# {title}",
        "",
        f"Момент выдачи: `{decision.issue_time}` (заморожен, test-период по ADR-006).",
        "",
        "## Вердикт оркестратора",
        "",
        f"**{decision.verdict}**",
        "",
        decision.explanation,
        "",
    ]
    if decision.rejection_reasons:
        lines += ["Причины:", ""] + [f"- {reason}" for reason in decision.rejection_reasons] + [""]
    if decision.warnings:
        lines += ["Предупреждения:", ""] + [f"- {warning}" for warning in decision.warnings] + [""]
    if decision.recommendation is not None:
        recommendation = decision.recommendation
        lines += [
            "## Рекомендация оператору",
            "",
            f"- Действие: `{recommendation['action']}`",
            f"- Прогноз серы: {recommendation['predicted_sulfur_mg_kg']:.4g} мг/кг "
            f"(предел {config.SULFUR_LIMIT_MG_KG:g} мг/кг)",
            f"- Тип обоснования: `{recommendation['evidence_type']}`",
            "- Доли блендинга: "
            + ", ".join(f"{name} {value:.4g}" for name, value in recommendation["blend_fractions"].items()),
            "- Уставки: "
            + ", ".join(f"{name} {value:.6g}" for name, value in recommendation["setpoints"].items()),
            "",
        ]
    if quality.forecast:
        lines += ["## Агент качества", ""]
        for item in quality.forecast:
            lines.append(
                f"- Горизонт {item['horizon_hours']:g} ч: {item['predicted_sulfur_mg_kg']:.4g} мг/кг, "
                f"запас {item['margin_mg_kg']:.4g} мг/кг = {item['margin_in_rmse']:.3g} backtest-RMSE, "
                f"риск `{item['risk_exceed_spec']}`, evidence_type `{item['evidence_type']}`"
            )
        lines += [
            "",
            f"Вход в диапазоне обучения: **{'да' if quality.in_training_range else 'нет'}**.",
            "",
        ]
    if quality.refusal:
        lines += ["## Агент качества", "", f"Отказ: {quality.refusal['reason']}.", ""]
    lines += ["## Агент надёжности", ""]
    if reliability.refusal:
        lines.append(f"Отказ: {reliability.refusal['reason']}.")
    else:
        lines.append(
            f"Индекс тяжести режима (прокси): {reliability.severity_index:.4g} — "
            f"класс `{reliability.severity_class}`, режим "
            f"{'допустим' if reliability.regime_admissible else 'недопустим'}."
        )
        if reliability.risk_factors:
            lines += [""] + [f"- Фактор риска: {factor}" for factor in reliability.risk_factors]
        else:
            lines += ["", "Отдельных факторов риска по прокси не выявлено."]
    lines += ["", "## Отклонённые кандидаты", ""]
    if proposal.candidates:
        for candidate in proposal.candidates:
            if candidate["id"] == proposal.recommended_id:
                continue
            lines.append(f"- `{candidate['id']}` — " + "; ".join(candidate["rejection_reasons"]))
    else:
        lines.append(
            f"Кандидаты не строились: {proposal.refusal['reason']}."
            if proposal.refusal
            else "Кандидатов нет."
        )
    lines += ["", "## Чего проверка не доказывает", ""]
    lines += [f"- {limit}" for limit in decision.limits]
    lines += ["", "## Оговорки", ""]
    lines += [f"- {notice}" for notice in proposal.mock_notices]
    lines += [
        "- `backtested` — ошибка прогноза проверена на отложенной по времени истории "
        "(ADR-008). Это не доказательство эффекта управляющего действия.",
        "- `simulated_unvalidated` — модельная оценка ещё не случавшегося действия. "
        "Симулятора установки нет, историей она не подтверждается.",
        "",
    ]
    return "\n".join(lines)
