"""Сценарии демонстрации: окна, моменты выдачи и общий прогон ролей.

Окна и моменты выдачи зафиксированы в ADR-011 (сценарии 2–4) и ADR-010
(сценарий 1) до правки моделей и не подбираются по полученному ответу.
Сценарий здесь — это параметры и критерий отбора, а не своя копия логики:
цепочка ролей одна на все сценарии, иначе «тот же прогон» на демонстрации
нельзя сравнить между окнами.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from . import config, features
from .agents import (
    card,
    context as context_module,
    optimization,
    orchestrator,
    quality as quality_agent,
    reliability as reliability_agent,
    report,
)

log = logging.getLogger("neftekod.scenarios")

ROLES = ("all", "quality", "reliability", "optimization", "orchestrator")


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    title: str
    window_start: str
    issue_time: str
    selection: str
    shows: str

    @property
    def output_dir(self) -> Path:
        return config.BUILD_DIR / f"scenario_{self.name}"


SCENARIOS: dict[str, ScenarioSpec] = {
    spec.name: spec
    for spec in (
        ScenarioSpec(
            name="stable",
            title="Сценарий 1 — устойчивый период",
            window_start="2026-01-29 00:00:00",
            issue_time="2026-02-04 12:00:00",
            selection=(
                "самая спокойная неделя test-периода по данным: сера ЛИМС 8.4–9.8 мг/кг "
                "(std 0.48), телеметрия P8/T11/F19 со std 0.004 / 1.5 / 4.9"
            ),
            shows="устойчивый период без лишних действий",
        ),
        ScenarioSpec(
            name="quality_risk",
            title="Сценарий 2 — риск ухудшения качества",
            window_start="2026-05-05 14:00:00",
            issue_time="2026-05-12 14:00:00",
            selection=(
                "максимальный рост серы за 72 ч среди моментов test-периода внутри "
                "полосы p05–p95 обучающей выборки при допустимом режиме: 3.2 → 9.9 мг/кг "
                "(+6.7), severity_index 0.124 (ADR-011)"
            ),
            shows="период с риском ухудшения качества",
        ),
        ScenarioSpec(
            name="stale_data",
            title="Сценарий 3 — неполные и устаревшие данные",
            window_start="2026-06-18 09:00:00",
            issue_time="2026-06-25 09:00:00",
            selection=(
                "середина крупнейшего разрыва публикации ЛИМС в test-периоде "
                "(350 ч, 2026-06-18 02:00 … 2026-07-02 16:00); возраст последнего "
                "анализа на момент выдачи 175 ч (ADR-011)"
            ),
            shows="неполные, устаревшие данные — отказ вместо догадки",
        ),
        ScenarioSpec(
            name="full_cycle",
            title="Сценарий 4 — полный цикл агентов",
            window_start="2026-03-28 20:00:00",
            issue_time="2026-04-04 20:00:00",
            selection=(
                "наибольший запас по сере среди моментов test-периода внутри полосы "
                "p05–p95 обучающей выборки при допустимом режиме: 5.5 мг/кг, запас 4.5, "
                "severity_index 0.068 (ADR-011)"
            ),
            shows="полный цикл от входа до финальной рекомендации",
        ),
    )
}


@dataclass(frozen=True)
class ScenarioRun:
    spec: ScenarioSpec
    context: context_module.ScenarioContext
    quality: quality_agent.QualityAssessment
    reliability: reliability_agent.ReliabilityAssessment
    proposal: optimization.OptimizationProposal
    submission: dict
    decision: orchestrator.Decision


def _scenario_input(spec: ScenarioSpec, scenario_context) -> dict:
    return {
        "scenario": spec.name,
        "title": spec.title,
        "window_selection": spec.selection,
        "window_start": spec.window_start,
        "issue_time": spec.issue_time,
        "snapshot": scenario_context.snapshot,
        "training_stats": scenario_context.training_stats,
        "backtest_test_split": scenario_context.backtest,
        "window_published_lims_n": int(len(scenario_context.window_lims)),
        "window_telemetry_n": int(len(scenario_context.window_telemetry)),
        "telemetry_tags": list(features.TELEMETRY_TAGS),
    }


def execute(spec: ScenarioSpec, *, blending: optimization.BlendingScenario | None = None) -> ScenarioRun:
    """Прогнать цепочку ролей на окне сценария. Случайности в прогоне нет."""
    blending = blending or optimization.BlendingScenario()
    log.info("%s", blending.mock_notice())
    log.info(
        "сценарий %s: окно %s … момент выдачи %s",
        spec.name,
        spec.window_start,
        spec.issue_time,
    )
    scenario_context = context_module.load_context(spec.name, spec.issue_time, spec.window_start)
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
        "модель эффекта уставок %s: test RMSE %.4g против baseline %s %.4g — %s",
        effect_model.name,
        effect_model.test_rmse,
        effect_model.baseline_model,
        effect_model.baseline_test_rmse,
        "выигрывает" if effect_model.beats_baseline else "проигрывает",
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
    return ScenarioRun(
        spec=spec,
        context=scenario_context,
        quality=quality,
        reliability=reliability,
        proposal=proposal,
        submission=submission,
        decision=decision,
    )


def write(run: ScenarioRun, *, role: str = "all") -> Path:
    """Сохранить вывод роли. Роль пишет только свой файл — см. --role."""
    if role not in ROLES:
        raise ValueError(f"неизвестная роль {role!r}: ожидалось одно из {list(ROLES)}")
    output_dir = run.spec.output_dir
    outputs = {
        "quality": [("quality.json", run.quality.to_dict())],
        "reliability": [("reliability.json", run.reliability.to_dict())],
        "optimization": [
            ("optimization.json", run.proposal.to_dict()),
            ("submission.json", run.submission),
        ],
        "orchestrator": [("decision.json", run.decision.to_dict())],
    }
    if role == "all":
        report.write_json(output_dir / "context.json", _scenario_input(run.spec, run.context))
    for name, payloads in outputs.items():
        if role in ("all", name):
            for filename, payload in payloads:
                report.write_json(output_dir / filename, payload)
    if role == "all":
        (output_dir / "recommendation.md").write_text(
            card.recommendation_markdown(
                run.spec.title,
                run.decision,
                run.quality,
                run.reliability,
                run.proposal,
                run.submission,
            ),
            encoding="utf-8",
        )
    log.info("вывод роли %s → %s", role, output_dir)
    return output_dir


def run(name: str, *, role: str = "all") -> ScenarioRun:
    """Прогнать сценарий по имени и сохранить вывод роли."""
    spec = SCENARIOS[name]
    executed = execute(spec)
    write(executed, role=role)
    return executed


def main(name: str, argv: list[str] | None = None) -> int:
    """Точка входа скрипта сценария: разбор --role, проверка входных файлов, прогон."""
    import argparse

    parser = argparse.ArgumentParser(description=SCENARIOS[name].title)
    parser.add_argument(
        "--role",
        choices=ROLES,
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
            log.error(
                "нет %s; сначала запустите scripts/prepare_data.py и scripts/evaluate_forecast.py",
                path,
            )
            return 1
    if not config.HT_TELEMETRY_CSV.exists():
        log.error("нет %s; скачайте телеметрию 24-2000 по ссылке из README", config.HT_TELEMETRY_CSV)
        return 1
    run(name, role=role)
    return 0
