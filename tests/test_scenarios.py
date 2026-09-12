"""Реестр сценариев: окна зафиксированы в ADR-011 и не правятся молча."""

from __future__ import annotations

import pandas as pd
import pytest

from neftekod import scenarios

# Окна и моменты выдачи — ADR-010 (сценарий 1) и ADR-011 (сценарии 2–4).
# Тест существует ровно затем, чтобы правка окна требовала правки ADR:
# подбор окна под полученный ответ — дефект, а не мелочь (CLAUDE.md).
FIXED_WINDOWS = {
    "stable": ("2026-01-29 00:00:00", "2026-02-04 12:00:00"),
    "quality_risk": ("2026-05-05 14:00:00", "2026-05-12 14:00:00"),
    "stale_data": ("2026-06-18 09:00:00", "2026-06-25 09:00:00"),
    "full_cycle": ("2026-03-28 20:00:00", "2026-04-04 20:00:00"),
}


def test_all_four_demo_scenarios_present():
    assert set(scenarios.SCENARIOS) == set(FIXED_WINDOWS)


@pytest.mark.parametrize("name", sorted(FIXED_WINDOWS))
def test_window_matches_adr(name):
    spec = scenarios.SCENARIOS[name]
    assert (spec.window_start, spec.issue_time) == FIXED_WINDOWS[name]


@pytest.mark.parametrize("name", sorted(set(FIXED_WINDOWS) - {"stable"}))
def test_adr_011_windows_are_seven_days(name):
    # Сценарий 1 зафиксирован раньше (ADR-010) с окном 6.5 суток и задним
    # числом не переписывается; правило 7 суток введено ADR-011.
    spec = scenarios.SCENARIOS[name]
    assert pd.Timestamp(spec.issue_time) - pd.Timestamp(spec.window_start) == pd.Timedelta(days=7)


@pytest.mark.parametrize("name", sorted(FIXED_WINDOWS))
def test_issue_time_inside_test_period(name):
    from neftekod import config

    # Момент выдачи внутри test-периода по ADR-006: сценарий не имеет права
    # стоять на данных, по которым подбирались модели.
    assert pd.Timestamp(scenarios.SCENARIOS[name].issue_time) >= pd.Timestamp(config.TEST_START)


@pytest.mark.parametrize("name", sorted(FIXED_WINDOWS))
def test_selection_criterion_is_written_down(name):
    spec = scenarios.SCENARIOS[name]
    assert spec.selection.strip(), "критерий отбора окна обязан быть записан рядом с окном"
    assert spec.shows.strip()


def test_unknown_role_is_rejected():
    assert "all" in scenarios.ROLES
    with pytest.raises(ValueError, match="неизвестная роль"):
        scenarios.write(None, role="нет такой")
