"""Карточка оператора: семь блоков ТЗ §5 присутствуют в каждом сценарии."""

from __future__ import annotations

import pytest

from neftekod import config, scenarios
from neftekod.agents import card, optimization, orchestrator, quality, reliability
from test_agents import _context, _long_format  # общие фикстуры ролей


def _card(context, title: str = "Тестовая карточка") -> str:
    quality_assessment = quality.assess(context)
    reliability_assessment = reliability.assess(context)
    proposal = optimization.propose(context, quality_assessment, reliability_assessment)
    submission = optimization.build_submission(
        context, quality_assessment, reliability_assessment, proposal
    )
    decision = orchestrator.review(
        submission,
        sources={
            "quality": quality_assessment.to_dict(),
            "reliability": reliability_assessment.to_dict(),
            "snapshot": context.snapshot,
        },
    )
    return card.recommendation_markdown(
        title, decision, quality_assessment, reliability_assessment, proposal, submission
    )


@pytest.mark.parametrize(
    "context_kwargs",
    [
        pytest.param({}, id="рекомендация"),
        pytest.param(
            {"long_format": _long_format(last_sample="2026-01-30 10:00:00")}, id="отказ-по-данным"
        ),
        pytest.param({"long_format": _long_format(sulfur=20.0)}, id="отказ-по-качеству"),
    ],
)
def test_all_seven_sections_are_present(context_kwargs):
    """§5 задаёт семь блоков; отказ — тоже ответ оператору и тоже с блоками."""
    text = _card(_context(**context_kwargs))
    for index, name in enumerate(card.SECTIONS, start=1):
        assert f"## {index}. {name}" in text, name


def test_card_states_what_is_not_modelled_instead_of_inventing_it():
    text = _card(_context())
    assert "Влияние на выпуск и энергозатраты не моделируется" in text
    assert "Число не выдумывается" in text


def test_economics_is_marked_as_informational_only():
    """Экономика в §5 нужна, но она не имеет права делать вариант допустимым."""
    text = _card(_context())
    assert "Стоимостный прокси" in text
    assert "никогда не делает вариант допустимым" in text


def test_setpoints_are_shown_as_current_to_recommended_with_units_warning():
    text = _card(_context())
    assert "| Параметр | Текущее | Рекомендуемое | Примечание |" in text
    assert card.UNITS_NOT_CONFIRMED in text


def test_freshness_of_lims_and_pak_is_in_the_card_not_only_in_json():
    text = _card(_context())
    assert "ЛИМС «Гидроочистка.2», сера, мг/кг" in text
    assert "ПАК, сера, мг/кг" in text
    assert "возраст" in text


def test_refusal_card_names_the_reason_and_calls_it_a_result():
    text = _card(_context(long_format=_long_format(last_sample="2026-01-30 10:00:00")))
    assert "Действие не предлагается" in text
    assert "старше 48 ч" in text
    assert "рискованную догадку" in text


@pytest.mark.parametrize("name", sorted(scenarios.SCENARIOS))
def test_saved_scenario_cards_have_all_seven_sections(name):
    """Проверка на сохранённых карточках: пропущенный блок — брак вывода."""
    path = config.BUILD_DIR / f"scenario_{name}" / "recommendation.md"
    if not path.exists():
        pytest.skip(f"нет {path} — сначала прогоните сценарий {name}")
    text = path.read_text(encoding="utf-8")
    for index, section in enumerate(card.SECTIONS, start=1):
        assert f"## {index}. {section}" in text, f"{name}: {section}"


def test_all_three_mandatory_indicators_are_shown_with_freshness():
    """Оператор видит все три обязательных показателя, а не только серу (ADR-015)."""
    text = _card(_context())
    assert "| T95, °C |" in text
    assert "| Цетановое число |" in text
    assert f"≤ {config.T95_LIMIT_C:g}" in text
    assert f"≥ {config.CETANE_NUMBER_MIN:g}" in text
    assert "Прогноз T95 смеси" in text
    assert "Прогноз цетанового числа смеси" in text


def test_spec_limits_are_labelled_as_scenario_parameters_not_as_a_norm():
    """Предел серы задан организаторами, два других — нет; путать нельзя."""
    text = _card(_context())
    assert "параметры сценария по ГОСТ" in text
    assert "численные пределы оставили команде" in text


def test_unverified_indicator_is_named_in_the_risk_block():
    stale = _long_format(cetane_sample="2025-11-01 10:00:00")
    text = _card(_context(long_format=stale))
    assert "не подтверждён" in text
    assert "не сертифицируется" in text
