"""Карточка оператора по семи блокам ТЗ, §5.

Модуль ничего не считает и ничего не додумывает: каждое число берётся из
вывода агентов и оркестратора. Если величина не моделируется — так и
написано, а не заменено правдоподобной оценкой.

Порядок блоков и их содержание заданы таблицей §5 `docs/tz/tz_neftekod.md`:
время и состояние → проблема/риск → предлагаемое действие → ожидаемый эффект
→ проверка ограничений → уверенность → объяснение выбора.
"""

from __future__ import annotations

from .. import config

SECTIONS = (
    "Время и состояние",
    "Проблема и риск",
    "Предлагаемое действие",
    "Ожидаемый эффект",
    "Проверка ограничений",
    "Уверенность",
    "Объяснение выбора",
)

# Единицы и масштабы P8/T11/F19 организаторами не подтверждены (ADR-007),
# поэтому уставка показывается как безразмерное значение сигнала с пометкой,
# а не как «температура 0.213».
UNITS_NOT_CONFIRMED = "ед. не подтв. (ADR-007)"


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "нет данных"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)


def _age(hours) -> str:
    if hours is None:
        return "возраст неизвестен"
    if hours < 48:
        return f"возраст {hours:.1f} ч"
    return f"возраст {hours / 24:.1f} сут"


def _state_block(decision, quality, reliability, submission) -> list[str]:
    lines = [f"Момент выдачи: `{decision.issue_time}` (заморожен, test-период по ADR-006).", ""]
    confidence = quality.data_confidence or {}
    lines += ["| Источник | Значение | Свежесть | Единицы подтверждены |", "|---|---|---|---|"]
    titles = {
        "lims_sulfur": "ЛИМС «Гидроочистка.2», сера, мг/кг",
        "pak_sulfur": "ПАК, сера, мг/кг",
        "P8": "P8 (сигнал)",
        "T11": "T11 (сигнал)",
        "F19": "F19 (сигнал)",
    }
    for name, title in titles.items():
        item = confidence.get(name)
        if item is None:
            continue
        value = "отсутствует" if item.get("missing") else _fmt(item.get("value"), 6)
        lines.append(
            f"| {title} | {value} | {_age(item.get('age_hours'))} | "
            f"{'да' if item.get('units_confirmed') else 'нет — ' + UNITS_NOT_CONFIRMED} |"
        )
    indicators = quality.spec_indicators or {}
    if indicators:
        lines += [
            "",
            "Обязательные показатели товарного ДТ (ТЗ и ответы организаторов — три, "
            "не один):",
            "",
            "| Показатель | Измерено | Предел | Запас | Свежесть | Подтверждён |",
            "|---|---|---|---|---|---|",
        ]
        t95 = indicators.get("t95") or {}
        cetane = indicators.get("cetane_number") or {}
        lines.append(
            f"| T95, °C | {_fmt(t95.get('value'), 5)} | ≤ {config.T95_LIMIT_C:g} | "
            f"{_fmt(t95.get('margin_c'), 3)} | {_age(t95.get('age_hours'))} | "
            f"{'да' if t95.get('verified') else 'НЕТ — ' + str(t95.get('detail'))} |"
        )
        lines.append(
            f"| Цетановое число | {_fmt(cetane.get('value'), 5)} | "
            f"≥ {config.CETANE_NUMBER_MIN:g} | {_fmt(cetane.get('margin'), 3)} | "
            f"{_age(cetane.get('age_hours'))} | "
            f"{'да' if cetane.get('verified') else 'НЕТ — ' + str(cetane.get('detail'))} |"
        )
        density = indicators.get("density") or {}
        lines.append(
            f"| Плотность при 15 °C, кг/м³ | {_fmt(density.get('value'), 6)} | "
            f"{config.DENSITY_MIN_KG_M3:g}–{config.DENSITY_MAX_KG_M3:g} | "
            f"{_fmt(density.get('margin_kg_m3'), 3)} | {_age(density.get('age_hours'))} | "
            f"{'да' if density.get('verified') else 'НЕТ — ' + str(density.get('detail'))} |"
        )
        lines += [
            "",
            f"Пределы T95 и цетанового числа — параметры сценария по ГОСТ 32511-2013 / "
            f"ГОСТ Р 52368 для К5: организаторы назвали показатели обязательными, но "
            f"численные пределы оставили команде (ADR-004, ADR-015). Предел серы "
            f"{config.SULFUR_LIMIT_MG_KG:g} мг/кг задан организаторами.",
        ]
    components = (submission.get("input") or {}).get("components") or {}
    fractions = (submission.get("input") or {}).get("current_blend_fractions") or {}
    if fractions:
        lines += ["", "Текущий состав товарного ДТ (модельный, ADR-013):", ""]
        for name, share in fractions.items():
            sulfur = (components.get(name) or {}).get("sulfur_mg_kg")
            lines.append(f"- {name}: доля {share:.4%}, сера {_fmt(sulfur, 5)} мг/кг")
    if reliability.severity_index is not None:
        lines += [
            "",
            f"Тяжесть режима (прокси): {reliability.severity_index:.4g} — класс "
            f"`{reliability.severity_class}`, режим "
            f"{'допустим' if reliability.regime_admissible else 'недопустим'}.",
        ]
    return lines


def _risk_block(decision, quality, reliability, proposal) -> list[str]:
    lines = []
    anomaly = quality.input_anomaly or {}
    if anomaly.get("anomalous"):
        band = anomaly.get("training_band") or []
        lines.append(
            "- **Контроль качества входа: последний анализ помечен как аномальный** "
            + "; ".join(item["detail"] for item in anomaly.get("flags", []))
            + (f" (полоса обучения {band[0]:g}–{band[1]:g} мг/кг)" if len(band) == 2 else "")
            + ". Действия на таком измерении не рекомендуются (ADR-014)."
        )
    for name, title in (
        ("t95", "T95"),
        ("cetane_number", "цетановое число"),
        ("density", "плотность"),
    ):
        item = (quality.spec_indicators or {}).get(name) or {}
        if item and not item.get("verified"):
            lines.append(
                f"- **Обязательный показатель «{title}» не подтверждён**: {item.get('detail')}. "
                "Изменение режима на таком входе не сертифицируется (ADR-015)."
            )
    if quality.refusal:
        lines.append(f"- Агент качества отказался от прогноза: {quality.refusal['reason']}.")
    else:
        for item in quality.forecast:
            lines.append(
                f"- Горизонт {item['horizon_hours']:g} ч: прогноз серы базового компонента "
                f"{item['predicted_sulfur_mg_kg']:.4g} мг/кг "
                f"(последний анализ {item['last_published_sulfur_mg_kg']:.4g}), запас "
                f"{item['margin_mg_kg']:.4g} мг/кг = {item['margin_in_rmse']:.3g} backtest-RMSE, "
                f"риск `{item['risk_exceed_spec']}`"
            )
    if reliability.refusal:
        lines.append(f"- Агент надёжности отказался от оценки: {reliability.refusal['reason']}.")
    for factor in reliability.risk_factors:
        lines.append(f"- Фактор риска режима: {factor}")
    if proposal.refusal:
        lines.append(f"- Агент оптимизации: {proposal.refusal['reason']}.")
    for warning in decision.warnings:
        lines.append(f"- Предупреждение оркестратора: {warning}")
    return lines or ["- Отдельных рисков по доступным признакам не выявлено."]


def _action_block(decision, submission) -> list[str]:
    recommendation = decision.recommendation
    if recommendation is None:
        lines = [
            f"**Действие не предлагается. Вердикт: `{decision.verdict}`.**",
            "",
            "Причины:",
            "",
        ]
        lines += [f"- {reason}" for reason in decision.rejection_reasons]
        lines += [
            "",
            "Это полноценный результат: при нехватке данных или отсутствии допустимого "
            "варианта система обязана отказаться, а не выдать рискованную догадку (ТЗ, §4).",
        ]
        return lines
    scenario_input = submission.get("input") or {}
    current_setpoints = scenario_input.get("current_setpoints") or {}
    current_fractions = scenario_input.get("current_blend_fractions") or {}
    lines = [f"Действие: `{recommendation['action']}`.", ""]
    lines += ["| Параметр | Текущее | Рекомендуемое | Примечание |", "|---|---|---|---|"]
    for tag, value in (recommendation.get("setpoints") or {}).items():
        current = current_setpoints.get(tag)
        changed = current is None or abs(float(value) - float(current)) > 1e-12
        lines.append(
            f"| {tag} | {_fmt(current, 6)} | {_fmt(value, 6)} | "
            f"{'изменить' if changed else 'без изменений'}, {UNITS_NOT_CONFIRMED} |"
        )
    for name, share in (recommendation.get("blend_fractions") or {}).items():
        current = current_fractions.get(name)
        changed = current is None or abs(float(share) - float(current)) > 1e-12
        lines.append(
            f"| доля {name} | {current:.4%} | {share:.4%} | "
            f"{'изменить' if changed else 'без изменений'} |"
        )
    return lines


def _effect_block(decision, proposal, submission) -> list[str]:
    recommendation = decision.recommendation
    if recommendation is None:
        return [
            "Эффекта нет: действие не предлагается. Сохранение текущего режима "
            "системой в этом случае тоже не подтверждается — проверить состояние "
            "должен оператор."
        ]
    candidate = proposal.recommended() or {}
    objective = candidate.get("objective") or {}
    current_share = (submission.get("input") or {}).get("current_blend_fractions") or {}
    current_cut = next(
        (value for name, value in current_share.items() if name != "hydrotreated_diesel"), None
    )
    lines = [
        f"- Прогноз серы товарного ДТ: **{recommendation['predicted_sulfur_mg_kg']:.4g} мг/кг** "
        f"при пределе {config.SULFUR_LIMIT_MG_KG:g} мг/кг "
        f"(запас {objective.get('sulfur_margin_mg_kg', float('nan')):.4g} мг/кг).",
        f"- Прогноз T95 смеси: **{_fmt(recommendation.get('predicted_t95_c'), 5)} °C** "
        f"при пределе {config.T95_LIMIT_C:g} °C (запас "
        f"{_fmt(objective.get('t95_margin_c'), 3)} °C). Модель смешения — массовый "
        "баланс, историей не проверялась (ADR-015).",
        f"- Прогноз цетанового числа смеси: "
        f"**{_fmt(recommendation.get('predicted_cetane_number'), 5)}** при минимуме "
        f"{config.CETANE_NUMBER_MIN:g} (запас {_fmt(objective.get('cetane_margin'), 3)}). "
        "Берётся худшая из двух рассматриваемых моделей смешения; цетановое число "
        "дешёвого компонента в пакете не измеряется и принято допущением (ADR-015).",
        f"- Прогноз плотности смеси: "
        f"**{_fmt(recommendation.get('predicted_density_kg_m3'), 6)} кг/м³** при "
        f"диапазоне {config.DENSITY_MIN_KG_M3:g}–{config.DENSITY_MAX_KG_M3:g} "
        f"(запас до ближайшей границы {_fmt(objective.get('density_margin_kg_m3'), 3)} "
        "кг/м³). Модель смешения — аддитивность объёмов 1/ρ = Σ wᵢ/ρᵢ, историей "
        "не проверялась (ADR-016).",
        f"- Тип обоснования: `{recommendation['evidence_type']}`.",
    ]
    cost = objective.get("cost_proxy_rel")
    if cost is not None:
        lines.append(
            f"- Стоимостный прокси: {cost:.8g} от цены базового компонента "
            f"(доля дешёвого компонента {objective.get('cut_component_share', 0.0):.4%} "
            f"против текущей {current_cut:.4%})."
            if current_cut is not None
            else f"- Стоимостный прокси: {cost:.8g}."
        )
    lines += [
        "- Экономика приведена **справочно**: она сравнивает уже допустимые варианты "
        "между собой и никогда не делает вариант допустимым (ТЗ, §4; контракт "
        "оркестратора не принимает экономических полей внутри рекомендации).",
        "- Влияние на выпуск не моделируется. Причина не в отсутствии данных: "
        "расход гидроочищенного ДТ с установки (`F17`, `T23`) и расход сырья "
        "(`T11`, `F26`) в телеметрии есть. Причина в том, что единицы этих "
        "сигналов организаторами не подтверждены (ADR-007), поэтому выпуск "
        "считался бы в относительных величинах, а не в тоннах. Число не "
        "выдумывается.",
        "- Влияние на энергозатраты не моделируется: прямых тегов потребления в "
        "пакете нет, есть только прокси (расход ВСГ `P24`, расходы пара на АВТ). "
        "Прокси не выдаётся за энергозатраты.",
        f"- Риск для оборудования: индекс тяжести режима "
        f"{_fmt(objective.get('severity_index'))} — действие не меняет уставки режима, "
        "если в таблице выше они помечены «без изменений».",
    ]
    return lines


def _constraints_block(decision) -> list[str]:
    if not decision.checks:
        return [
            "Независимые проверки не выполнялись: рекомендации, которую можно было бы "
            "проверить, нет."
        ]
    lines = ["| Проверка | Результат | Детали |", "|---|---|---|"]
    for check in decision.checks:
        lines.append(
            f"| `{check['check']}` | {'пройдена' if check['passed'] else 'НЕ пройдена'} | "
            f"{check['detail']} |"
        )
    lines += [
        "",
        "Все проверки выполнены оркестратором заново на числах рекомендации и сверены "
        "с независимо собранными выводами агентов качества и надёжности (ADR-010, §6а).",
    ]
    return lines


def _confidence_block(quality, proposal) -> list[str]:
    lines = []
    if quality.forecast:
        item = quality.forecast[0]
        lines += [
            f"- Модель прогноза: `{item['model']}` (shrinkage {item['baseline_shrinkage']:g}, "
            f"уровень {item['baseline_level_mg_kg']:g} мг/кг), отобрана на valid (ADR-012).",
            f"- Ошибка на отложенной истории: RMSE {item['backtest_rmse_mg_kg']:.4g}, "
            f"MAE {item['backtest_mae_mg_kg']:.4g}, MedAE {item['backtest_medae_mg_kg']:.4g} мг/кг "
            f"на {item['backtest_n']} точках; `evidence_type: {item['evidence_type']}`.",
            f"- Вход в диапазоне обучения: {'да' if quality.in_training_range else 'нет'}.",
        ]
    stale = [
        f"{name} ({_age(item.get('age_hours'))})"
        for name, item in (quality.data_confidence or {}).items()
        if item.get("missing") or (item.get("age_hours") or 0) > 24
    ]
    if stale:
        lines.append("- **Предупреждение о старых или пропущенных данных**: " + "; ".join(stale) + ".")
    else:
        lines.append("- Пропущенных и просроченных источников на момент выдачи нет.")
    unconfirmed = [
        name for name, item in (quality.data_confidence or {}).items() if not item.get("units_confirmed")
    ]
    if unconfirmed:
        lines.append(
            "- Единицы и масштаб не подтверждены организаторами для: "
            + ", ".join(unconfirmed)
            + " (ADR-007). Свежесть сигнала доверия к его смыслу не добавляет."
        )
    anomaly = quality.input_anomaly or {}
    if anomaly.get("value") is not None and not anomaly.get("anomalous"):
        band = anomaly.get("training_band") or []
        if len(band) == 2:
            lines.append(
                f"- Контроль качества входа пройден: последний анализ "
                f"{anomaly['value']:g} мг/кг внутри полосы обучения "
                f"{band[0]:g}–{band[1]:g} мг/кг (ADR-014)."
            )
    blending = proposal.blending_inputs or {}
    if blending.get("value_mg_kg") is not None:
        lines.append(
            f"- Сера дешёвого компонента взята из анализа `{blending['tag']}` "
            f"({blending['value_declared']:g} {blending['unit_declared']}, отбор "
            f"{blending['sample_time']}, {_age(blending['age_hours'])}), а не назначена (ADR-013)."
        )
    cut_density = (proposal.spec_inputs or {}).get("cut_density") or {}
    if cut_density.get("value") is not None:
        lines.append(
            f"- Плотность дешёвого компонента взята из анализа `{cut_density['tag']}` "
            f"({cut_density['value']:g} кг/м³, отбор {cut_density['sample_time']}, "
            f"{_age(cut_density['age_hours'])}), а не назначена (ADR-016)."
        )
    elif cut_density:
        lines.append(
            f"- Плотность дешёвого компонента не подтверждена: {cut_density.get('detail')}."
        )
    cut_t95 = (proposal.spec_inputs or {}).get("cut_t95") or {}
    if cut_t95.get("value") is not None:
        lines.append(
            f"- T95 дешёвого компонента взят из анализа `{cut_t95['tag']}` "
            f"({cut_t95['value']:g} °C, отбор {cut_t95['sample_time']}, "
            f"{_age(cut_t95['age_hours'])}), а не назначен (ADR-015)."
        )
    elif cut_t95:
        lines.append(f"- T95 дешёвого компонента не подтверждён: {cut_t95.get('detail')}.")
    for assumption in quality.assumptions:
        lines.append(f"- Допущение: {assumption}")
    return lines


def _explanation_block(decision, proposal) -> list[str]:
    lines = [decision.explanation, ""]
    if proposal.candidates:
        lines += ["Рассмотренные альтернативы и причины отклонения:", ""]
        for candidate in proposal.candidates:
            if candidate["id"] == proposal.recommended_id:
                lines.append(f"- **`{candidate['id']}` — выбран**: {candidate['description']}")
                continue
            reasons = "; ".join(candidate["rejection_reasons"]) or "нет причин отклонения"
            lines.append(
                f"- `{candidate['id']}` ({', '.join(candidate['rejection_codes']) or 'без кода'}): "
                f"{reasons}"
            )
    else:
        lines.append(
            "Кандидаты не строились: "
            + (proposal.refusal["reason"] if proposal.refusal else "нет данных")
            + "."
        )
    lines += ["", "Правила отбора, по которым сравнивались варианты:", ""]
    lines += [f"- {rule}" for rule in proposal.decision_rules]
    return lines


def recommendation_markdown(title, decision, quality, reliability, proposal, submission) -> str:
    """Собрать карточку оператора: семь блоков §5 плюс границы проверки."""
    blocks = {
        "Время и состояние": _state_block(decision, quality, reliability, submission),
        "Проблема и риск": _risk_block(decision, quality, reliability, proposal),
        "Предлагаемое действие": _action_block(decision, submission),
        "Ожидаемый эффект": _effect_block(decision, proposal, submission),
        "Проверка ограничений": _constraints_block(decision),
        "Уверенность": _confidence_block(quality, proposal),
        "Объяснение выбора": _explanation_block(decision, proposal),
    }
    lines = [f"# {title}", "", f"Вердикт оркестратора: **{decision.verdict}**", ""]
    for index, name in enumerate(SECTIONS, start=1):
        lines += [f"## {index}. {name}", ""] + blocks[name] + [""]
    lines += ["## Чего проверка не доказывает", ""]
    lines += [f"- {limit}" for limit in decision.limits]
    lines += ["", "## Оговорки", ""]
    lines += [f"- {notice}" for notice in proposal.mock_notices]
    lines += [
        "- `backtested` — ошибка прогноза проверена на отложенной по времени истории "
        "(ADR-008, ADR-012). Это не доказательство эффекта управляющего действия.",
        "- `simulated_unvalidated` — модельная оценка ещё не случавшегося действия. "
        "Симулятора установки нет, историей она не подтверждается.",
        "",
    ]
    return "\n".join(lines)
