"""Контроль качества входа: аномальное значение названо аномальным."""

from __future__ import annotations

import pytest

from neftekod import anomaly
from neftekod.agents import optimization, quality, reliability
from test_agents import WIDE_HISTORY, _context, _long_format

STATS = {"p01": 2.68, "p99": 15.72}


def test_value_inside_the_band_is_not_flagged():
    report = anomaly.assess_lab_value(8.4, STATS)
    assert report["anomalous"] is False
    assert report["flags"] == []
    assert report["training_band"] == [2.68, 15.72]


@pytest.mark.parametrize("value", [16.0, 20.3, 2120.0])
def test_value_above_the_band_is_flagged(value):
    report = anomaly.assess_lab_value(value, STATS)
    assert report["anomalous"] is True
    assert [item["code"] for item in report["flags"]] == ["outside_training_band"]


@pytest.mark.parametrize("value", [0.0, -1.5])
def test_nonphysical_value_is_flagged_separately(value):
    """Ноль и отрицательная сера — дефект измерения, а не чистый продукт."""
    report = anomaly.assess_lab_value(value, STATS)
    assert [item["code"] for item in report["flags"]] == ["nonphysical"]


def test_missing_value_is_not_called_anomalous():
    """Отсутствие значения — отдельная ситуация, ею занимается отказ агента."""
    report = anomaly.assess_lab_value(None, STATS)
    assert report["anomalous"] is False
    assert report["value"] is None


def test_all_codes_are_declared():
    for value in (0.0, 20.3):
        for flag in anomaly.assess_lab_value(value, STATS)["flags"]:
            assert flag["code"] in anomaly.ANOMALY_CODES


class TestAnomalyInsideTheChain:
    def test_quality_publishes_the_flag(self):
        context = _context(long_format=_long_format(sulfur=2120.0))
        assessment = quality.assess(context)
        assert assessment.input_anomaly["anomalous"] is True

    def test_actions_are_not_recommended_on_an_anomalous_analysis(self):
        """Менять режим по подозрительному измерению нельзя; hold остаётся."""
        context = _context(long_format=_long_format(sulfur=0.05))
        quality_assessment = quality.assess(context)
        assert quality_assessment.input_anomaly["anomalous"] is True
        proposal = optimization.propose(
            context, quality_assessment, reliability.assess(context)
        )
        actions = [item for item in proposal.candidates if item["kind"] != "no_action"]
        assert actions
        for candidate in actions:
            assert candidate["admissible"] is False
            assert "anomalous_input" in candidate["rejection_codes"], candidate["id"]
        hold = next(item for item in proposal.candidates if item["kind"] == "no_action")
        assert hold["admissible"] is True

    def test_clean_analysis_leaves_the_chain_untouched(self):
        context = _context(long_format=_long_format(sulfur=4.0, history_values=WIDE_HISTORY))
        proposal = optimization.propose(
            context, quality.assess(context), reliability.assess(context)
        )
        for candidate in proposal.candidates:
            assert "anomalous_input" not in candidate["rejection_codes"]
