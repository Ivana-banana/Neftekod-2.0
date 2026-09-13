"""Показатели спецификации помимо серы: T95 и цетановое число (ADR-015).

Проверка идёт на реальном `lims_export_2023.xlsx`, а не на синтетике: весь
смысл этих показателей в том, что они измеряются редко, с разной частотой и
иногда содержат физически невозможные значения. Синтетика ровно эти свойства
и потеряла бы.
"""

import pandas as pd
import pytest

from neftekod import config, spec

ISSUE_STABLE = "2026-02-04 12:00:00"
ISSUE_QUALITY_RISK = "2026-05-12 14:00:00"
ISSUE_FULL_CYCLE = "2026-04-04 20:00:00"


class TestResolveMeasuredOnRealData:
    def test_product_t95_is_measured_not_computed(self, lims_long):
        """T95 берётся измерением: формула ВАК заблокирована ссылкой Pipeline (ADR-003)."""
        measured = spec.resolve_measured(
            lims_long,
            ISSUE_STABLE,
            spec.BASE_T95_TAG,
            max_age_hours=config.T95_MAX_AGE_HOURS,
            min_physical=spec.MIN_PHYSICAL_T95_C,
        )
        assert measured.verified
        assert measured.value == pytest.approx(345.0)
        assert measured.unit == "°C"
        assert measured.age_hours == pytest.approx(26.0)

    def test_future_analysis_is_not_used(self, lims_long):
        """Анализ, опубликованный позже момента выдачи, оператору неизвестен."""
        measured = spec.resolve_measured(
            lims_long,
            ISSUE_STABLE,
            spec.BASE_T95_TAG,
            max_age_hours=config.T95_MAX_AGE_HOURS,
        )
        assert pd.Timestamp(measured.available_at) <= pd.Timestamp(ISSUE_STABLE)

    def test_cetane_is_verified_when_within_its_own_threshold(self, lims_long):
        """У цетанового числа свой порог: отбор примерно раз в месяц."""
        measured = spec.resolve_measured(
            lims_long,
            ISSUE_FULL_CYCLE,
            spec.BASE_CETANE_TAG,
            max_age_hours=config.CETANE_MAX_AGE_HOURS,
        )
        assert measured.verified
        assert measured.value == pytest.approx(51.7, abs=1e-4)
        # Тот же анализ по порогу ЛИМС (48 ч) был бы просрочен: пороги разные
        # не по недосмотру, а потому что частота отбора разная (ADR-015).
        assert measured.age_hours > config.T95_MAX_AGE_HOURS

    def test_stale_cetane_is_not_verified_and_carries_no_value(self, lims_long):
        """Просроченный показатель не подставляется как будто он свежий."""
        measured = spec.resolve_measured(
            lims_long,
            ISSUE_QUALITY_RISK,
            spec.BASE_CETANE_TAG,
            max_age_hours=config.CETANE_MAX_AGE_HOURS,
        )
        assert not measured.verified
        assert measured.value is None
        assert measured.age_hours > config.CETANE_MAX_AGE_HOURS
        assert "не подтверждён" in measured.detail

    def test_absent_indicator_refuses_instead_of_guessing(self, lims_long):
        """До первого анализа показателя нет значения — и не выдумывается."""
        measured = spec.resolve_measured(
            lims_long,
            "2023-01-02 00:00:00",
            spec.BASE_CETANE_TAG,
            max_age_hours=config.CETANE_MAX_AGE_HOURS,
        )
        assert not measured.verified
        assert measured.value is None
        assert "нет опубликованного анализа" in measured.detail

    def test_physically_impossible_value_is_rejected(self, lims_long):
        """В `Гидроочистка.1.95%.T` есть 0 °C — это не измерение (ADR-015).

        Без этой проверки нулевой T95 компонента давал бы смеси
        несуществующий запас по показателю.
        """
        part = lims_long[lims_long["tag"].eq(spec.CUT_T95_TAG)].copy()
        part["value"] = pd.to_numeric(part["value"], errors="coerce")
        zeros = part[part["value"] < spec.MIN_PHYSICAL_T95_C]
        assert not zeros.empty, "в выгрузке больше нет нефизичных T95 — проверку надо пересмотреть"
        issue = pd.Timestamp(zeros.sort_values("timestamp").iloc[0]["available_at"])
        measured = spec.resolve_measured(
            lims_long,
            issue,
            spec.CUT_T95_TAG,
            max_age_hours=config.T95_MAX_AGE_HOURS,
            min_physical=spec.MIN_PHYSICAL_T95_C,
        )
        assert not measured.verified
        assert measured.value is None
        assert "физически невозможно" in measured.detail


class TestProductIndicators:
    def test_both_indicators_resolved_together(self, lims_long):
        indicators = spec.resolve_product_indicators(lims_long, ISSUE_FULL_CYCLE)
        assert indicators.verified
        payload = indicators.to_dict()
        assert payload["t95"]["limit_c"] == config.T95_LIMIT_C
        assert payload["cetane_number"]["minimum"] == config.CETANE_NUMBER_MIN

    def test_margin_can_be_zero_and_is_not_hidden(self, lims_long):
        """На 2026-04-04 T95 продукта равен ровно пределу — запас 0, не «норма»."""
        payload = spec.resolve_product_indicators(lims_long, ISSUE_FULL_CYCLE).to_dict()
        assert payload["t95"]["value"] == pytest.approx(config.T95_LIMIT_C)
        assert payload["t95"]["margin_c"] == pytest.approx(0.0)

    def test_unverified_indicator_reports_its_reason(self, lims_long):
        indicators = spec.resolve_product_indicators(lims_long, ISSUE_QUALITY_RISK)
        assert not indicators.verified
        assert len(indicators.unverified_reasons()) == 1


class TestBlendModels:
    def test_t95_mass_balance(self):
        value = spec.blend_t95({"base": 0.9, "cut": 0.1}, {"base": 350.0, "cut": 360.0})
        assert value == pytest.approx(351.0)

    def test_t95_unknown_component_gives_no_value(self):
        """Неизвестный T95 компонента не заменяется значением базового."""
        assert spec.blend_t95({"base": 0.9, "cut": 0.1}, {"base": 350.0, "cut": None}) is None

    def test_nonlinear_index_is_not_conservative_by_itself(self):
        """Причина, по которой одной формой из ADR-004 обойтись нельзя.

        (0.5·55^1.5 + 0.5·45^1.5)^(1/1.5) = 354.86^(2/3) = 50.125 — выше
        линейного среднего 50.0. Защитную проверку на оптимистичной оценке
        ставить нельзя, поэтому политика — минимум из двух (ADR-015).
        """
        fractions = {"base": 0.5, "cut": 0.5}
        values = {"base": 55.0, "cut": 45.0}
        index = spec.cetane_blend_index_form(fractions, values)
        linear = spec.cetane_linear_form(fractions, values)
        assert index == pytest.approx(50.125, abs=1e-3)
        assert linear == pytest.approx(50.0)
        assert index > linear
        assert spec.blend_cetane_number(fractions, values) == pytest.approx(linear)

    def test_cetane_of_single_component_is_its_own_value(self):
        assert spec.blend_cetane_number({"base": 1.0}, {"base": 52.5}) == pytest.approx(52.5)

    def test_cetane_unknown_component_gives_no_value(self):
        assert spec.blend_cetane_number({"a": 0.5, "b": 0.5}, {"a": 52.0, "b": None}) is None

    def test_index_and_linear_agree_at_the_fractions_actually_used(self):
        """Обоснование выбора модели числом, а не на словах (ADR-015).

        На доле дешёвого компонента 7.4e-5 нелинейный индекс и линейное
        смешение расходятся на четыре порядка меньше разрешения лабораторного
        значения, поэтому спор о форме модели на этих данных не решает ничего.
        """
        fraction = 7.4e-5
        fractions = {"base": 1.0 - fraction, "cut": fraction}
        values = {"base": 51.7, "cut": spec.CUT_CETANE_NUMBER}
        index = spec.blend_cetane_number(fractions, values)
        linear = (1.0 - fraction) * values["base"] + fraction * values["cut"]
        assert abs(index - linear) < 1e-4
