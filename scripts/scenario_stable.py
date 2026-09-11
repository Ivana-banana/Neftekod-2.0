#!/usr/bin/env python3
"""Сценарий 1: устойчивый период без лишних действий.

Сквозной прогон «данные → оценки агентов → рекомендация» на реальных данных
с замороженным моментом выдачи. Запускать после scripts/prepare_data.py и
scripts/evaluate_forecast.py:

    python scripts/scenario_stable.py

Окно и момент выдачи зафиксированы ниже и не подбираются по результату
(CLAUDE.md: сценарии фиксируются до финальной настройки моделей). Момент
выдачи лежит в test-периоде по ADR-006, модели обучены только на train.

Вывод: build/scenario_stable/{context,quality,reliability,optimization,
submission,decision}.json и recommendation.md. Случайности в прогоне нет,
поэтому два запуска подряд дают побайтово одинаковые файлы.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neftekod import config, features  # noqa: E402
from neftekod.agents import (  # noqa: E402
    context as context_module,
    optimization,
    orchestrator,
    quality as quality_agent,
    reliability as reliability_agent,
    report,
)

SCENARIO = "stable"
# Самая спокойная неделя test-периода по данным: сера ЛИМС 8.4–9.8 мг/кг
# (std 0.48), телеметрия P8/T11/F19 со std 0.004 / 1.5 / 4.9. Окно выбрано
# по свойствам режима, а не по тому, какой ответ получится.
WINDOW_START = "2026-01-29 00:00:00"
ISSUE_TIME = "2026-02-04 12:00:00"

OUTPUT_DIR = config.BUILD_DIR / f"scenario_{SCENARIO}"

log = logging.getLogger("scenario_stable")


def _recommendation_markdown(decision, quality, reliability, proposal) -> str:
    lines = [
        "# Сценарий 1 — устойчивый период",
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
    for candidate in proposal.candidates:
        if candidate["id"] == proposal.recommended_id:
            continue
        lines.append(f"- `{candidate['id']}` — " + "; ".join(candidate["rejection_reasons"]))
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        choices=("all", "quality", "reliability", "optimization", "orchestrator"),
        default="all",
        help=(
            "какую роль сохранить. Роль считает и своих предшественников, но "
            "записывает только свой вывод — так subagent из .claude/agents/ "
            "получает ровно тот файл, за который отвечает."
        ),
    )
    role = parser.parse_args(argv).role
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for path in (config.LONG_FORMAT_CSV, config.FORECAST_CSV):
        if not path.exists():
            log.error("нет %s; сначала запустите scripts/prepare_data.py и scripts/evaluate_forecast.py", path)
            return 1
    if not config.HT_TELEMETRY_CSV.exists():
        log.error("нет %s; скачайте телеметрию 24-2000 по ссылке из README", config.HT_TELEMETRY_CSV)
        return 1

    blending = optimization.BlendingScenario()
    log.info("%s", blending.mock_notice())
    log.info("сценарий %s: окно %s … %s, момент выдачи %s", SCENARIO, WINDOW_START, ISSUE_TIME, ISSUE_TIME)

    scenario_context = context_module.load_context(SCENARIO, ISSUE_TIME, WINDOW_START)
    quality = quality_agent.assess(scenario_context)
    log.info(
        "агент качества: %s",
        "отказ — " + quality.refusal["reason"]
        if quality.refusal
        else f"риск {quality.forecast[0]['risk_exceed_spec']}, "
        f"прогноз {quality.forecast[0]['predicted_sulfur_mg_kg']:.4g} мг/кг",
    )
    reliability = reliability_agent.assess(scenario_context)
    log.info(
        "агент надёжности: %s",
        "отказ — " + reliability.refusal["reason"]
        if reliability.refusal
        else f"severity_index {reliability.severity_index:.4g} ({reliability.severity_class})",
    )
    effect_model = optimization.fit_effect_model(scenario_context)
    log.info(
        "модель эффекта уставок %s: test RMSE %.4g против persistence %.4g — %s",
        effect_model.name,
        effect_model.test_rmse,
        effect_model.persistence_test_rmse,
        "выигрывает" if effect_model.beats_persistence else "проигрывает",
    )
    proposal = optimization.propose(
        scenario_context, quality, reliability, blending=blending, effect_model=effect_model
    )
    log.info(
        "агент оптимизации: %d кандидатов, допустимых %d, выбран %s",
        len(proposal.candidates),
        sum(1 for item in proposal.candidates if item["admissible"]),
        proposal.recommended_id,
    )
    submission = optimization.build_submission(
        scenario_context, quality, reliability, proposal, blending=blending
    )
    # Оркестратор сверяет пакет с выводами ролей, собранными независимо от
    # агента оптимизации, а не с копиями внутри пакета (ADR-010).
    decision = orchestrator.review(
        submission,
        sources={
            "quality": quality.to_dict(),
            "reliability": reliability.to_dict(),
            "snapshot": scenario_context.snapshot,
        },
    )
    log.info("оркестратор: %s", decision.verdict)

    scenario_input = {
        "scenario": SCENARIO,
        "window_start": WINDOW_START,
        "issue_time": ISSUE_TIME,
        "snapshot": scenario_context.snapshot,
        "training_stats": scenario_context.training_stats,
        "backtest_test_split": scenario_context.backtest,
        "window_published_lims_n": int(len(scenario_context.window_lims)),
        "window_telemetry_n": int(len(scenario_context.window_telemetry)),
        "telemetry_tags": list(features.TELEMETRY_TAGS),
    }
    outputs = {
        "quality": [("quality.json", quality.to_dict())],
        "reliability": [("reliability.json", reliability.to_dict())],
        "optimization": [
            ("optimization.json", proposal.to_dict()),
            ("submission.json", submission),
        ],
        "orchestrator": [("decision.json", decision.to_dict())],
    }
    if role == "all":
        report.write_json(OUTPUT_DIR / "context.json", scenario_input)
    for name, payloads in outputs.items():
        if role in ("all", name):
            for filename, payload in payloads:
                report.write_json(OUTPUT_DIR / filename, payload)
    if role == "all":
        (OUTPUT_DIR / "recommendation.md").write_text(
            _recommendation_markdown(decision, quality, reliability, proposal), encoding="utf-8"
        )
    log.info("вывод роли %s → %s", role, OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
