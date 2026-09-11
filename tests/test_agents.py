"""Роли МАС: контракт между агентами и adversarial-проверка оркестратора.

Жёсткие проверки как таковые закреплены в tests/test_hard_constraints.py.
Здесь проверяется другое: что кандидат без evidence_type не проходит, что
оркестратор отклоняет выгодный, но нарушающий вариант, и что отсутствие
допустимого варианта заканчивается явным отказом, а не натянутым ответом.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from neftekod import config, constraints, features
from neftekod.agents import context as context_module
from neftekod.agents import optimization, orchestrator, quality, reliability, report

ISSUE_TIME = pd.Timestamp("2026-02-04 12:00:00")
WINDOW_START = pd.Timestamp("2026-01-29 00:00:00")

BACKTEST = {
    0.0: {
        "persistence": {
            "n": 253,
            "mae": 1.8,
            "rmse": 2.6,
            "r2": -0.13,
            "evidence_type": "backtested",
            "alpha": None,
            "telemetry_lag_hours": None,
        },
        "ridge_telemetry": {
            "n": 253,
            "mae": 25.4,
            "rmse": 168.2,
            "r2": -4665.9,
            "evidence_type": "backtested",
            "alpha": 100.0,
            "telemetry_lag_hours": 0.0,
        },
    }
}


def _telemetry(values: dict | None = None) -> pd.DataFrame:
    """Синтетическая телеметрия: train 2024–2025 плюс окно сценария."""
    train = pd.date_range("2024-01-01", "2025-12-01", freq="6h")
    window = pd.date_range(WINDOW_START, ISSUE_TIME, freq="10min")
    stamps = train.append(window)
    frame = pd.DataFrame({"timestamp": stamps})
    # Ровная «пила» даёт предсказуемые квантили: p05/p95 отстоят от середины.
    ramp = pd.Series(range(len(train)), dtype=float) / max(len(train) - 1, 1)
    for tag, (low, high) in {"P8": (0.1, 0.3), "T11": (340.0, 380.0), "F19": (150.0, 250.0)}.items():
        current = (values or {}).get(tag, (low + high) / 2)
        frame[tag] = list(low + ramp * (high - low)) + [current] * len(window)
    return frame


def _long_format(sulfur: float = 8.4, last_sample: str = "2026-02-03 10:00:00") -> pd.DataFrame:
    """ЛИМС «Гидроочистка.2» и ПАК с корректным временем доступности."""
    history = pd.date_range("2024-01-01 10:00", "2025-12-01 10:00", freq="1D")
    rows = [
        {
            "timestamp": stamp,
            "available_at": stamp + pd.Timedelta(hours=config.LIMS_PUBLICATION_DELAY_HOURS),
            "source": "lims",
            "tag": features.TARGET_TAG,
            "value": 8.0 + (index % 5) * 0.2,
        }
        for index, stamp in enumerate(history)
    ]
    sample = pd.Timestamp(last_sample)
    rows.append(
        {
            "timestamp": sample,
            "available_at": sample + pd.Timedelta(hours=config.LIMS_PUBLICATION_DELAY_HOURS),
            "source": "lims",
            "tag": features.TARGET_TAG,
            "value": sulfur,
        }
    )
    for index, stamp in enumerate(history):
        rows.append(
            {
                "timestamp": stamp,
                "available_at": stamp,
                "source": "pak",
                "tag": features.PAK_SULFUR_TAG,
                "value": 7.5 + (index % 4) * 0.1,
            }
        )
    rows.append(
        {
            "timestamp": ISSUE_TIME,
            "available_at": ISSUE_TIME,
            "source": "pak",
            "tag": features.PAK_SULFUR_TAG,
            "value": 7.9,
        }
    )
    return pd.DataFrame(rows)


def _context(**kwargs) -> context_module.ScenarioContext:
    return context_module.load_context(
        "unit-test",
        ISSUE_TIME,
        WINDOW_START,
        long_format=kwargs.pop("long_format", _long_format()),
        telemetry=kwargs.pop("telemetry", _telemetry()),
        backtest=kwargs.pop("backtest", BACKTEST),
    )


@pytest.fixture
def scenario_context():
    return _context()


class TestContext:
    def test_window_excludes_unpublished_analysis(self):
        """Проба, отобранная до момента выдачи, но не выданная лабораторией."""
        long_format = _long_format(last_sample="2026-02-04 10:00:00")
        context = _context(long_format=long_format)
        assert context.snapshot["lims_missing"] is False
        # Отбор 10:00 + 4 ч публикации = 14:00, а выдача в 12:00.
        assert context.snapshot["lims_timestamp"] < pd.Timestamp("2026-02-04 10:00:00")
        assert len(context.window_lims) == 0

    def test_empty_training_history_is_an_error_not_a_made_up_range(self):
        telemetry = _telemetry()
        telemetry.loc[:, "F19"] = float("nan")
        with pytest.raises(ValueError, match="нет обучающих значений"):
            _context(telemetry=telemetry)

    def test_training_stats_ignore_test_period(self, scenario_context):
        stats = scenario_context.training_stats["T11_value"]
        # Значение окна (360.0) в train не попадает: полоса построена по «пиле».
        assert stats["p05"] < stats["p95"] <= 380.0
        assert stats["n"] > 100


class TestQuality:
    def test_forecast_is_backtested_persistence(self, scenario_context):
        assessment = quality.assess(scenario_context)
        assert assessment.refusal is None
        assert [item["evidence_type"] for item in assessment.forecast] == ["backtested"]
        assert assessment.forecast[0]["predicted_sulfur_mg_kg"] == pytest.approx(8.4)

    def test_risk_is_high_when_margin_below_one_rmse(self, scenario_context):
        assessment = quality.assess(scenario_context)
        item = assessment.forecast[0]
        assert item["margin_mg_kg"] == pytest.approx(1.6)
        assert item["risk_exceed_spec"] == "high"

    def test_out_of_training_range_is_flagged(self):
        """Сигнал вне p01–p99 обучения не выдаётся с прежней уверенностью."""
        context = _context(telemetry=_telemetry({"T11": 999.0}))
        assessment = quality.assess(context)
        assert assessment.in_training_range is False
        assert assessment.training_range["T11_value"]["in_training_range"] is False
        assert all(item["in_training_range"] is False for item in assessment.forecast)

    def test_distance_to_the_edge_is_visible_not_only_the_flag(self, scenario_context):
        """Значение у самого края формально «внутри» — это должно быть видно."""
        position = scenario_context.training_stats["T11_value"]
        edge = position["p99"] - 1e-9
        near_edge = dataclasses.replace(
            scenario_context, snapshot=scenario_context.snapshot | {"T11_value": edge}
        )
        assessment = quality.assess(near_edge)
        item = assessment.training_range["T11_value"]
        assert item["in_training_range"] is True
        assert item["relative_position"] == pytest.approx(1.0, abs=1e-6)

    def test_unconfirmed_units_are_flagged_next_to_freshness(self, scenario_context):
        confidence = quality.assess(scenario_context).data_confidence
        assert confidence["P8"]["units_confirmed"] is False
        assert confidence["lims_sulfur"]["units_confirmed"] is True

    def test_risk_thresholds_are_declared_as_assumption(self, scenario_context):
        assumptions = quality.assess(scenario_context).assumptions
        assert any("backtest-RMSE" in item for item in assumptions)

    def test_stale_analysis_is_refused_not_guessed(self):
        context = _context(long_format=_long_format(last_sample="2026-01-30 10:00:00"))
        assessment = quality.assess(context)
        assert assessment.refusal is not None
        assert assessment.forecast == []

    def test_missing_analysis_is_refused(self, scenario_context):
        """Нет опубликованного анализа — отказ, а не «считаем, что в норме»."""
        blind = dataclasses.replace(
            scenario_context,
            snapshot=scenario_context.snapshot
            | {"lims_missing": True, "lims_value": None, "lims_age_hours": None},
        )
        assessment = quality.assess(blind)
        assert assessment.refusal is not None
        assert "ЛИМС" in assessment.refusal["reason"]
        assert assessment.forecast == []


class TestReliability:
    def test_allowed_ranges_are_marked_as_assumption(self, scenario_context):
        assessment = reliability.assess(scenario_context)
        assert set(assessment.allowed_ranges) == set(features.TELEMETRY_TAGS)
        assert "не подтверждённый промышленный предел" in assessment.allowed_ranges_basis
        assert any("промышленный предел" in item for item in assessment.assumptions)

    def test_proxy_assumptions_are_in_the_output_not_only_in_code(self, scenario_context):
        assessment = reliability.assess(scenario_context)
        assert any("прокси" in item for item in assessment.assumptions)
        assert any("ADR-007" in item for item in assessment.assumptions)

    def test_fast_excursion_becomes_a_risk_factor(self):
        """Скачок внутри окна не должен молча тонуть в спокойном std."""
        telemetry = _telemetry()
        window = telemetry["timestamp"] >= WINDOW_START
        spike = telemetry.index[window][len(telemetry.index[window]) // 2]
        stats_width = 250.0 - 150.0
        telemetry.loc[spike, "F19"] = telemetry.loc[spike, "F19"] + 2 * stats_width
        assessment = reliability.assess(_context(telemetry=telemetry))
        assert assessment.refusal is None
        assert any("скорост" in factor for factor in assessment.risk_factors)

    def test_admissibility_threshold_is_declared_as_assumption(self, scenario_context):
        assessment = reliability.assess(scenario_context)
        assert any(
            str(reliability.ADMISSIBLE_SEVERITY_MAX) in item for item in assessment.assumptions
        )

    def test_missing_signal_gives_refusal_not_a_score(self):
        """История сигнала есть, но актуального значения на момент выдачи нет."""
        telemetry = _telemetry()
        telemetry.loc[telemetry["timestamp"] >= WINDOW_START, "F19"] = float("nan")
        assessment = reliability.assess(_context(telemetry=telemetry))
        assert assessment.refusal is not None
        assert "F19" in assessment.refusal["reason"]
        assert assessment.severity_index is None
        assert assessment.regime_admissible is False

    def test_stale_signal_is_refused_even_though_as_of_found_a_value(self):
        """As-of подставляет старое значение; спокойный скор по нему — обман."""
        telemetry = _telemetry()
        telemetry.loc[telemetry["timestamp"] >= WINDOW_START, "T11"] = float("nan")
        context = _context(telemetry=telemetry)
        assert context.snapshot["T11_missing"] is False
        assert context.snapshot["T11_age_hours"] > reliability.MAX_SIGNAL_AGE_HOURS
        assessment = reliability.assess(context)
        assert assessment.refusal is not None
        assert "возраст" in assessment.refusal["reason"]


class TestOptimization:
    def _proposal(self, context):
        return optimization.propose(
            context, quality.assess(context), reliability.assess(context)
        )

    def test_no_candidate_is_missing_evidence_type(self, scenario_context):
        """CLAUDE.md, «Верификация и честность»: поле обязательно у каждого вывода."""
        proposal = self._proposal(scenario_context)
        assert proposal.candidates
        for candidate in proposal.candidates:
            evidence = candidate["recommendation"].get("evidence_type")
            assert constraints.check_evidence_type(evidence) == [], candidate["id"]

    def test_only_no_action_may_claim_backtested(self, scenario_context):
        proposal = self._proposal(scenario_context)
        for candidate in proposal.candidates:
            expected = "backtested" if candidate["kind"] == "no_action" else "simulated_unvalidated"
            assert candidate["recommendation"]["evidence_type"] == expected, candidate["id"]

    def test_blending_scenario_is_marked_mocked(self, scenario_context):
        proposal = self._proposal(scenario_context)
        assert any("[MOCKED:" in notice for notice in proposal.mock_notices)

    def test_high_sulfur_component_is_rejected_by_hard_constraint(self, scenario_context):
        proposal = self._proposal(scenario_context)
        blends = [item for item in proposal.candidates if item["kind"] == "blend"]
        assert blends
        for candidate in blends:
            assert candidate["admissible"] is False
            assert any("sulfur" in reason for reason in candidate["rejection_reasons"])

    def test_stable_period_recommends_holding(self, scenario_context):
        proposal = self._proposal(scenario_context)
        assert proposal.recommended_id == "hold"
        assert proposal.recommended()["recommendation"]["evidence_type"] == "backtested"

    def test_refuses_when_quality_refused(self):
        context = _context(long_format=_long_format(last_sample="2026-01-30 10:00:00"))
        proposal = optimization.propose(
            context, quality.assess(context), reliability.assess(context)
        )
        assert proposal.candidates == []
        assert proposal.recommended_id is None
        assert proposal.refusal is not None


def _submission(context) -> dict:
    quality_assessment = quality.assess(context)
    reliability_assessment = reliability.assess(context)
    proposal = optimization.propose(context, quality_assessment, reliability_assessment)
    return optimization.build_submission(
        context, quality_assessment, reliability_assessment, proposal
    )


class TestOrchestrator:
    def test_approves_reproducible_recommendation(self, scenario_context):
        decision = orchestrator.review(_submission(scenario_context))
        assert decision.verdict == "approved"
        assert decision.recommendation["action"] == "hold"
        assert all(check["passed"] for check in decision.checks)

    def test_high_quality_risk_is_warned_even_when_approved(self, scenario_context):
        decision = orchestrator.review(_submission(scenario_context))
        assert decision.warnings

    def test_better_economics_do_not_buy_a_violation(self, scenario_context):
        """Кандидат с лучшей экономикой, но с нарушением ограничения, отклонён."""
        submission = _submission(scenario_context)
        submission["recommendation"] = submission["recommendation"] | {
            "action": "adjust_blend",
            "predicted_sulfur_mg_kg": 10.4,
            "evidence_type": "simulated_unvalidated",
            "blend_fractions": {"hydrotreated_diesel": 0.9995, "straight_run_240_350": 0.0005},
            "expected_cost_saving_pct": 12.0,
        }
        decision = orchestrator.review(submission)
        assert decision.verdict == "rejected"
        assert decision.recommendation is None
        assert any("sulfur" in reason for reason in decision.rejection_reasons)

    def test_claimed_sulfur_must_survive_independent_recomputation(self, scenario_context):
        """Слово агента оптимизации не заменяет пересчёт по входу."""
        submission = _submission(scenario_context)
        submission["recommendation"] = submission["recommendation"] | {
            "action": "adjust_blend",
            "predicted_sulfur_mg_kg": 9.0,
            "evidence_type": "simulated_unvalidated",
            "blend_fractions": {"hydrotreated_diesel": 0.99, "straight_run_240_350": 0.01},
        }
        decision = orchestrator.review(submission)
        assert decision.verdict == "rejected"
        assert any("не совпадает с пересчитанной" in reason for reason in decision.rejection_reasons)
        check = next(item for item in decision.checks if item["check"] == "blend_sulfur_recomputed")
        assert check["passed"] is False

    def test_agent_cannot_widen_its_own_ranges(self, scenario_context):
        submission = _submission(scenario_context)
        widened = {tag: [-1e9, 1e9] for tag in features.TELEMETRY_TAGS}
        submission["recommendation"] = submission["recommendation"] | {"allowed_ranges": widened}
        decision = orchestrator.review(submission)
        assert decision.verdict == "rejected"
        assert any("диапазоны" in reason for reason in decision.rejection_reasons)

    def test_setpoint_outside_applied_range_is_rejected(self, scenario_context):
        submission = _submission(scenario_context)
        setpoints = dict(submission["recommendation"]["setpoints"])
        setpoints["T11"] = 10_000.0
        submission["recommendation"] = submission["recommendation"] | {
            "action": "adjust_setpoint",
            "setpoints": setpoints,
        }
        decision = orchestrator.review(submission)
        assert decision.verdict == "rejected"
        assert any("вне применённого диапазона" in reason for reason in decision.rejection_reasons)

    def test_no_admissible_option_gives_explicit_refusal(self):
        context = _context(long_format=_long_format(last_sample="2026-01-30 10:00:00"))
        decision = orchestrator.review(_submission(context))
        assert decision.verdict == "no_admissible_recommendation"
        assert decision.recommendation is None
        assert decision.rejection_reasons
        assert "Допустимой рекомендации нет" in decision.explanation

    def test_reasoning_of_executors_is_not_accepted(self, scenario_context):
        """Оркестратор — ревьюер, а не читатель чужих обоснований."""
        submission = _submission(scenario_context) | {
            "candidates": [{"id": "hold", "why": "мне кажется, так лучше"}]
        }
        with pytest.raises(ValueError, match="не принимает поля"):
            orchestrator.review(submission)


class TestReport:
    def test_json_text_is_stable_and_json_parsable(self, scenario_context):
        payload = quality.assess(scenario_context).to_dict()
        first = report.to_json_text(payload)
        assert first == report.to_json_text(payload)
        assert first.endswith("\n")

    def test_nan_and_timestamps_are_serialised_explicitly(self):
        text = report.to_json_text(
            {"t": pd.Timestamp("2026-02-04 12:00"), "missing": float("nan"), "flag": True}
        )
        assert '"t": "2026-02-04 12:00:00"' in text
        assert '"missing": null' in text
        assert '"flag": true' in text
