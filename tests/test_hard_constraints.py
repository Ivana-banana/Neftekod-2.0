"""Жёсткие ограничения (ТЗ, §4). Прогоняется Stop-hook перед завершением хода.

Эти проверки не смягчаются никогда: ни ради лучшей метрики, ни из-за
отсутствия данных. Отсутствие данных здесь означает отказ от рекомендации,
а не «считаем, что в пределах нормы».
"""

import pytest

from neftekod import config, constraints


def rules(violations):
    return {violation.rule for violation in violations}


class TestSulfur:
    def test_limit_value_is_allowed(self):
        assert constraints.check_sulfur(config.SULFUR_LIMIT_MG_KG) == []

    @pytest.mark.parametrize("value", [10.0001, 10.5, 12, 1000])
    def test_above_limit_is_violation(self, value):
        assert rules(constraints.check_sulfur(value)) == {"sulfur"}

    @pytest.mark.parametrize("value", [None, float("nan"), "9", "", True])
    def test_missing_or_non_numeric_is_violation(self, value):
        """Неизвестная сера — не «допустимо по умолчанию»."""
        assert rules(constraints.check_sulfur(value)) == {"sulfur"}


class TestT95:
    """T95 — обязательный показатель наравне с серой (ADR-004, ADR-015)."""

    def test_limit_value_is_allowed(self):
        assert constraints.check_t95(config.T95_LIMIT_C) == []

    @pytest.mark.parametrize("value", [360.0001, 361.0, 370, 400])
    def test_above_limit_is_violation(self, value):
        assert rules(constraints.check_t95(value)) == {"t95"}

    @pytest.mark.parametrize("value", [None, float("nan"), "350", "", True])
    def test_missing_or_non_numeric_is_violation(self, value):
        """«Не измерили» не означает «в пределах нормы»."""
        assert rules(constraints.check_t95(value)) == {"t95"}


class TestCetaneNumber:
    """Единственное ограничение проекта снизу: чем меньше, тем хуже."""

    def test_minimum_value_is_allowed(self):
        assert constraints.check_cetane_number(config.CETANE_NUMBER_MIN) == []

    @pytest.mark.parametrize("value", [50.9999, 50.0, 45, 0])
    def test_below_minimum_is_violation(self, value):
        assert rules(constraints.check_cetane_number(value)) == {"cetane_number"}

    def test_direction_is_not_confused_with_the_others(self):
        """Большое значение — это хорошо: перепутанное направление поймано тестом."""
        assert constraints.check_cetane_number(58.0) == []

    @pytest.mark.parametrize("value", [None, float("nan"), "52", "", True])
    def test_missing_or_non_numeric_is_violation(self, value):
        assert rules(constraints.check_cetane_number(value)) == {"cetane_number"}


class TestDensity:
    """Плотность — единственная проверка проекта с двусторонним диапазоном."""

    @pytest.mark.parametrize("value", [820.0, 830.0, 845.0])
    def test_inside_range_is_allowed(self, value):
        assert constraints.check_density(value) == []

    @pytest.mark.parametrize("value", [845.0001, 846.14, 856.2])
    def test_above_maximum_is_violation(self, value):
        """На реальных данных 12 анализов продукта из 1015 выше 845 кг/м³."""
        assert rules(constraints.check_density(value)) == {"density"}

    @pytest.mark.parametrize("value", [819.9999, 800.0, 0.0])
    def test_below_minimum_is_violation(self, value):
        """Слишком лёгкое топливо так же вне спецификации, как слишком тяжёлое."""
        assert rules(constraints.check_density(value)) == {"density"}

    @pytest.mark.parametrize("value", [None, float("nan"), "836", "", True])
    def test_missing_or_non_numeric_is_violation(self, value):
        assert rules(constraints.check_density(value)) == {"density"}


class TestBlendFractions:
    def test_sum_of_one_is_allowed(self):
        assert constraints.check_blend_fractions({"a": 0.6, "b": 0.4}) == []

    @pytest.mark.parametrize("fractions", [{"a": 0.6, "b": 0.3}, {"a": 0.7, "b": 0.4}])
    def test_sum_not_one_is_violation(self, fractions):
        assert rules(constraints.check_blend_fractions(fractions)) == {"blend_fractions"}

    def test_negative_fraction_is_violation_even_if_sum_is_one(self):
        """Сумма 100% сама по себе не делает состав физичным."""
        assert rules(constraints.check_blend_fractions({"a": 1.3, "b": -0.3})) == {
            "blend_fractions"
        }

    def test_empty_composition_is_violation(self):
        assert rules(constraints.check_blend_fractions({})) == {"blend_fractions"}

    def test_float_noise_within_tolerance(self):
        assert constraints.check_blend_fractions([0.1] * 10) == []


class TestRanges:
    def test_value_inside_declared_range(self):
        assert constraints.check_within_ranges({"T5": 350.0}, {"T5": (300.0, 380.0)}) == []

    def test_value_outside_declared_range(self):
        assert rules(constraints.check_within_ranges({"T5": 400.0}, {"T5": (300.0, 380.0)})) == {
            "range"
        }

    def test_parameter_without_range_is_violation(self):
        """Нет подтверждённого диапазона — нет разрешения (ТЗ, §4)."""
        assert rules(constraints.check_within_ranges({"T5": 350.0}, {})) == {"range"}


class TestEvidenceType:
    @pytest.mark.parametrize("value", sorted(constraints.EVIDENCE_TYPES))
    def test_allowed_values(self, value):
        assert constraints.check_evidence_type(value) == []

    @pytest.mark.parametrize("value", [None, "", "validated", "backtest"])
    def test_missing_or_unknown_is_violation(self, value):
        """Модельная оценка эффекта не может выдаваться за проверенную историей."""
        assert rules(constraints.check_evidence_type(value)) == {"evidence_type"}


class TestValidateRecommendation:
    def _valid(self):
        return {
            "predicted_sulfur_mg_kg": 8.2,
            "predicted_t95_c": 352.0,
            "predicted_cetane_number": 53.0,
            "predicted_density_kg_m3": 836.0,
            "evidence_type": "simulated_unvalidated",
            "blend_fractions": {"godt": 0.85, "fraction_240_350": 0.15},
            "setpoints": {"T5": 352.0},
            "allowed_ranges": {"T5": (330.0, 370.0)},
        }

    def test_valid_recommendation_passes(self):
        assert constraints.validate_recommendation(self._valid()) == []

    def test_better_metrics_do_not_excuse_a_violation(self):
        """Экономический выигрыш не компенсирует нарушение по качеству."""
        recommendation = self._valid() | {
            "predicted_sulfur_mg_kg": 10.4,
            "expected_throughput_gain_pct": 12.0,
            "expected_energy_saving_pct": 8.0,
        }
        assert rules(constraints.validate_recommendation(recommendation)) == {"sulfur"}

    @pytest.mark.parametrize(
        "key", ["predicted_t95_c", "predicted_cetane_number", "predicted_density_kg_m3"]
    )
    def test_missing_mandatory_indicator_rejects_recommendation(self, key):
        """Три обязательных показателя проверяются безусловно, а не «если поле есть».

        Иначе рекомендацию можно было бы провести, просто не положив показатель
        в пакет — ровно то, что запрещает ответ организаторов (ADR-004).
        """
        recommendation = {name: value for name, value in self._valid().items() if name != key}
        assert constraints.validate_recommendation(recommendation) != []

    def test_t95_violation_is_not_excused_by_sulfur_margin(self):
        """Запас по одному показателю не компенсирует нарушение по другому."""
        recommendation = self._valid() | {"predicted_sulfur_mg_kg": 1.0, "predicted_t95_c": 366.0}
        assert rules(constraints.validate_recommendation(recommendation)) == {"t95"}

    def test_cetane_violation_is_not_excused_by_cheaper_blend(self):
        recommendation = self._valid() | {
            "predicted_cetane_number": 49.5,
            "expected_cost_saving_pct": 15.0,
        }
        assert rules(constraints.validate_recommendation(recommendation)) == {"cetane_number"}

    def test_violations_accumulate(self):
        recommendation = {
            "predicted_sulfur_mg_kg": 11.0,
            "predicted_t95_c": 372.0,
            "predicted_cetane_number": 44.0,
            "predicted_density_kg_m3": 851.0,
            "blend_fractions": {"a": 0.5, "b": 0.4},
            "setpoints": {"T5": 500.0},
            "allowed_ranges": {"T5": (330.0, 370.0)},
        }
        assert rules(constraints.validate_recommendation(recommendation)) == {
            "sulfur",
            "t95",
            "cetane_number",
            "density",
            "evidence_type",
            "blend_fractions",
            "range",
        }

    def test_empty_recommendation_is_rejected(self):
        assert constraints.validate_recommendation({}) != []
