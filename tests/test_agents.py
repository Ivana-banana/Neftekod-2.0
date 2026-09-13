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

from neftekod import config, constraints, features, spec
from neftekod.agents import context as context_module
from neftekod.agents import optimization, orchestrator, quality, reliability, report

ISSUE_TIME = pd.Timestamp("2026-02-04 12:00:00")
WINDOW_START = pd.Timestamp("2026-01-29 00:00:00")
# Широкая обучающая полоса — как на реальных данных (p01–p99 = 2.68…15.72):
# на узкой синтетической полосе любой чистый вход считался бы аномальным.
WIDE_HISTORY = (3.0, 5.0, 8.0, 9.0, 12.0, 15.0)

BACKTEST = {
    0.0: {
        # Выбранный baseline — сдвинутый persistence (ADR-012): прогноз =
        # shrinkage · последний анализ + (1 − shrinkage) · уровень обучения.
        "shrunk_persistence": {
            "n": 253,
            "mae": 1.53,
            "rmse": 2.6,
            "r2": 0.13,
            "medae": 0.9,
            "evidence_type": "backtested",
            "baseline_shrinkage": 0.3,
            "baseline_level_mg_kg": 8.6,
            "alpha": None,
            "telemetry_lag_hours": None,
        },
        "persistence": {
            "n": 253,
            "mae": 1.8,
            "rmse": 2.6,
            "r2": -0.13,
            "medae": 1.2,
            "evidence_type": "backtested",
            "baseline_shrinkage": None,
            "baseline_level_mg_kg": None,
            "alpha": None,
            "telemetry_lag_hours": None,
        },
        "ridge_telemetry": {
            "n": 253,
            "mae": 25.4,
            "rmse": 168.2,
            "r2": -4665.9,
            "medae": 4.2,
            "evidence_type": "backtested",
            "baseline_shrinkage": None,
            "baseline_level_mg_kg": None,
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


def _long_format(
    sulfur: float = 8.4,
    last_sample: str = "2026-02-03 10:00:00",
    feed_sulfur_percent: float | None = 0.95,
    feed_sample: str = "2026-02-01 10:00:00",
    history_values: tuple[float, ...] = (8.0, 8.2, 8.4, 8.6, 8.8),
    product_t95: float | None = 352.0,
    cut_t95: float | None = 356.0,
    product_cetane: float | None = 53.0,
    cetane_sample: str = "2026-01-20 10:00:00",
) -> pd.DataFrame:
    """ЛИМС «Гидроочистка.2», ПАК, сера сырья и показатели спецификации.

    T95 и цетановое число добавлены с разной частотой отбора не для красоты:
    именно разные пороги устаревания у этих показателей отличают их от серы
    (ADR-015). `None` означает «анализа нет вовсе» — тогда рекомендация
    обязана не выдаваться.
    """
    history = pd.date_range("2024-01-01 10:00", "2025-12-01 10:00", freq="1D")
    rows = [
        {
            "timestamp": stamp,
            "available_at": stamp + pd.Timedelta(hours=config.LIMS_PUBLICATION_DELAY_HOURS),
            "source": "lims",
            "tag": features.TARGET_TAG,
            "value": history_values[index % len(history_values)],
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
    if feed_sulfur_percent is not None:
        # Сера сырья гидроочистки — реальный источник серы дешёвого компонента
        # блендинга (ADR-013); отбор редкий, публикация +4 ч.
        feed_stamp = pd.Timestamp(feed_sample)
        rows.append(
            {
                "timestamp": feed_stamp,
                "available_at": feed_stamp + pd.Timedelta(hours=config.LIMS_PUBLICATION_DELAY_HOURS),
                "source": "lims",
                "tag": optimization.BlendingScenario().cut_sulfur_tag,
                "value": feed_sulfur_percent,
            }
        )
    # Показатели спецификации: T95 отбирается той же суточной пробой, что и
    # сера, цетановое число — примерно раз в месяц (ADR-015).
    for tag, value, stamp in (
        (spec.BASE_T95_TAG, product_t95, last_sample),
        (spec.CUT_T95_TAG, cut_t95, last_sample),
        (spec.BASE_CETANE_TAG, product_cetane, cetane_sample),
    ):
        if value is None:
            continue
        moment = pd.Timestamp(stamp)
        rows.append(
            {
                "timestamp": moment,
                "available_at": moment
                + pd.Timedelta(hours=config.LIMS_PUBLICATION_DELAY_HOURS),
                "source": "lims",
                "tag": tag,
                "value": value,
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
    def test_forecast_is_the_backtested_shrunk_baseline(self, scenario_context):
        assessment = quality.assess(scenario_context)
        assert assessment.refusal is None
        assert [item["evidence_type"] for item in assessment.forecast] == ["backtested"]
        item = assessment.forecast[0]
        # 0.3 · 8.4 + 0.7 · 8.6: прогноз воспроизводится по параметрам backtest,
        # а не подбирается внутри сценария (ADR-012).
        assert item["predicted_sulfur_mg_kg"] == pytest.approx(8.54)
        assert item["last_published_sulfur_mg_kg"] == pytest.approx(8.4)
        assert item["model"] == "shrunk_persistence"

    def test_forecast_without_baseline_parameters_is_an_error(self, scenario_context):
        """Без параметров выбранного baseline прогноз не выдумывается."""
        broken = {
            0.0: {
                "shrunk_persistence": BACKTEST[0.0]["shrunk_persistence"]
                | {"baseline_shrinkage": None},
                "ridge_telemetry": BACKTEST[0.0]["ridge_telemetry"],
            }
        }
        context = dataclasses.replace(scenario_context, backtest=broken)
        with pytest.raises(ValueError, match="нет параметров baseline"):
            quality.assess(context)

    def test_risk_is_high_when_margin_below_one_rmse(self, scenario_context):
        assessment = quality.assess(scenario_context)
        item = assessment.forecast[0]
        assert item["margin_mg_kg"] == pytest.approx(1.46)
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

    def test_actions_are_never_backtested(self, scenario_context):
        proposal = self._proposal(scenario_context)
        for candidate in proposal.candidates:
            if candidate["kind"] != "no_action":
                assert candidate["recommendation"]["evidence_type"] == "simulated_unvalidated", (
                    candidate["id"]
                )

    def test_hold_claims_backtested_only_with_a_measured_composition(self, scenario_context):
        """Модельная доля в текущем составе делает и «ничего не менять» моделью.

        Сера смеси при ненулевой модельной доле считается балансом, а не
        измеряется, поэтому подпись backtested ей не принадлежит (ADR-013).
        """
        modelled = self._proposal(scenario_context).recommended()
        assert modelled["recommendation"]["evidence_type"] == "simulated_unvalidated"
        pure = optimization.propose(
            scenario_context,
            quality.assess(scenario_context),
            reliability.assess(scenario_context),
            blending=optimization.BlendingScenario(current_cut_fraction=0.0),
        ).recommended()
        assert pure["recommendation"]["evidence_type"] == "backtested"

    def test_every_candidate_carries_all_three_mandatory_indicators(self, scenario_context):
        """Организаторы назвали обязательными три показателя, не один (ADR-004)."""
        proposal = self._proposal(scenario_context)
        assert proposal.candidates
        for candidate in proposal.candidates:
            recommendation = candidate["recommendation"]
            assert constraints.check_sulfur(recommendation["predicted_sulfur_mg_kg"]) == []
            assert isinstance(recommendation["predicted_t95_c"], float), candidate["id"]
            assert isinstance(recommendation["predicted_cetane_number"], float), candidate["id"]

    def test_unverified_cetane_blocks_every_candidate(self):
        """Просроченный обязательный показатель — отказ, а не оговорка.

        Цетановое число отбирается примерно раз в месяц; анализ старше порога
        не подставляется как будто он свежий, поэтому сертифицировать нечего.
        """
        stale = str(ISSUE_TIME - pd.Timedelta(hours=config.CETANE_MAX_AGE_HOURS + 24))
        context = _context(long_format=_long_format(cetane_sample=stale))
        assessment = quality.assess(context)
        # Агент качества при этом не отказывается от прогноза серы: оператор
        # обязан увидеть риск по сере, даже когда ЦЧ не подтверждено.
        assert assessment.refusal is None
        assert assessment.spec_indicators["cetane_number"]["verified"] is False
        proposal = optimization.propose(context, assessment, reliability.assess(context))
        assert proposal.recommended_id is None
        assert proposal.refusal is not None
        for candidate in proposal.candidates:
            assert "cetane_number" in " ".join(candidate["rejection_reasons"]), candidate["id"]
            assert "hard_constraint" in candidate["rejection_codes"], candidate["id"]

    def test_absent_t95_blocks_every_candidate(self):
        context = _context(long_format=_long_format(product_t95=None))
        proposal = optimization.propose(
            context, quality.assess(context), reliability.assess(context)
        )
        assert proposal.recommended_id is None
        assert all(
            "t95" in " ".join(candidate["rejection_reasons"]) for candidate in proposal.candidates
        )

    def test_t95_above_limit_rejects_the_blend_even_with_sulfur_headroom(self):
        """Запас по сере не покупает нарушение по T95 (ТЗ, §4)."""
        context = _context(long_format=_long_format(sulfur=3.0, product_t95=366.0))
        proposal = optimization.propose(
            context, quality.assess(context), reliability.assess(context)
        )
        assert proposal.recommended_id is None
        reasons = " ".join(
            reason for candidate in proposal.candidates for reason in candidate["rejection_reasons"]
        )
        assert "T95" in reasons

    def test_setpoint_candidates_admit_the_spec_effect_is_unmodelled(self, scenario_context):
        """Система не делает вид, что знает T95 после изменения уставки."""
        model = optimization.fit_effect_model(scenario_context)
        proposal = optimization.propose(
            scenario_context,
            quality.assess(scenario_context),
            reliability.assess(scenario_context),
            effect_model=model,
        )
        setpoint_candidates = [
            item for item in proposal.candidates if item["kind"] == "setpoint"
        ]
        assert setpoint_candidates
        for candidate in setpoint_candidates:
            assert "unmodelled_spec_effect" in candidate["rejection_codes"], candidate["id"]
            assert not candidate["admissible"]

    def test_margin_gate_uses_the_worst_horizon_not_the_most_favourable(self):
        """Порог действия берётся по худшему горизонту, на который оно влияет."""
        backtest = {
            0.0: BACKTEST[0.0],
            3.0: {
                "shrunk_persistence": BACKTEST[0.0]["shrunk_persistence"] | {"rmse": 9.5},
                "persistence": BACKTEST[0.0]["persistence"] | {"rmse": 9.5},
                "ridge_telemetry": BACKTEST[0.0]["ridge_telemetry"],
            },
        }
        context = _context(backtest=backtest)
        proposal = self._proposal(context)
        blend = next(item for item in proposal.candidates if item["kind"] == "blend")
        assert any("9.5" in reason for reason in blend["rejection_reasons"])
        assert not any("2.6 мг/кг" in reason for reason in blend["rejection_reasons"])

    def test_rejection_reasons_are_machine_readable(self, scenario_context):
        """UI и тест не должны разбирать причину отклонения по подстроке."""
        proposal = self._proposal(scenario_context)
        rejected = [item for item in proposal.candidates if not item["admissible"]]
        assert rejected
        for candidate in rejected:
            assert candidate["rejection_codes"], candidate["id"]
            assert set(candidate["rejection_codes"]) <= set(optimization.REJECTION_CODES)
            assert len(candidate["rejection_codes"]) == len(candidate["rejection_reasons"])
        blend = next(item for item in rejected if item["kind"] == "blend")
        assert set(blend["rejection_codes"]) <= set(optimization.REJECTION_CODES)

    def test_setpoint_rejected_when_effect_model_loses_to_baseline(self, scenario_context):
        """Модель эффекта, проигравшая backtest, не даёт права на действие."""
        losing = optimization.EffectModel(
            name="ridge_telemetry",
            horizon_hours=0.0,
            alpha=100.0,
            telemetry_lag_hours=0.0,
            sensitivity={tag: 0.0 for tag in features.TELEMETRY_TAGS},
            test_rmse=168.2,
            baseline_model="shrunk_persistence",
            baseline_test_rmse=2.6,
        )
        assert losing.beats_baseline is False
        proposal = optimization.propose(
            scenario_context,
            quality.assess(scenario_context),
            reliability.assess(scenario_context),
            effect_model=losing,
        )
        setpoints = [item for item in proposal.candidates if item["kind"] == "setpoint"]
        assert setpoints
        for candidate in setpoints:
            assert candidate["admissible"] is False
            assert "unvalidated_effect_model" in candidate["rejection_codes"]

    def test_blending_scenario_is_marked_mocked(self, scenario_context):
        proposal = self._proposal(scenario_context)
        assert any("[MOCKED:" in notice for notice in proposal.mock_notices)

    def test_blend_candidates_stay_inside_the_physical_sulfur_maximum(self, scenario_context):
        """Сетка кандидатов строится от массового баланса, а не от круглых процентов."""
        proposal = self._proposal(scenario_context)
        blends = [item for item in proposal.candidates if item["kind"] == "blend"]
        assert blends
        for candidate in blends:
            predicted = candidate["recommendation"]["predicted_sulfur_mg_kg"]
            assert predicted <= config.SULFUR_LIMIT_MG_KG + 1e-9, candidate["id"]
            assert "hard_constraint" not in candidate["rejection_codes"], candidate["id"]

    def test_base_component_above_the_limit_cannot_be_fixed_by_blending(self):
        """Сера базового компонента выше предела — отказ, а не «подберём долю»."""
        context = _context(long_format=_long_format(sulfur=20.0))
        proposal = optimization.propose(
            context, quality.assess(context), reliability.assess(context)
        )
        assert proposal.recommended_id is None
        assert proposal.refusal is not None
        for candidate in proposal.candidates:
            assert "hard_constraint" in candidate["rejection_codes"], candidate["id"]

    def test_cheaper_blend_is_recommended_when_the_margin_covers_the_forecast_error(self):
        """Экономика решает между допустимыми вариантами — когда прогноз точен.

        Порог допустимости действия — запас не меньше backtest-RMSE, поэтому
        механизм оживает только при достаточно точном прогнозе. Здесь RMSE
        задана 0.5 мг/кг вместо реальных 2.409: на реальных данных такого
        запаса внутри полосы p05–p95 не встречается ни разу (ADR-013), и это
        записано как результат, а не обойдено подкруткой порога.
        """
        accurate = {
            0.0: {
                "shrunk_persistence": BACKTEST[0.0]["shrunk_persistence"] | {"rmse": 0.5},
                "persistence": BACKTEST[0.0]["persistence"],
                "ridge_telemetry": BACKTEST[0.0]["ridge_telemetry"],
            }
        }
        context = _context(backtest=accurate)
        proposal = optimization.propose(
            context, quality.assess(context), reliability.assess(context)
        )
        recommended = proposal.recommended()
        assert recommended["kind"] == "blend"
        assert recommended["objective"]["cut_component_share"] > (
            optimization.BlendingScenario().current_cut_fraction
        )
        assert recommended["objective"]["sulfur_margin_mg_kg"] >= 0.5  # backtest-RMSE фикстуры

    def test_missing_feed_analysis_stops_blending(self):
        """Без анализа серы компонента состав смеси не определён — отказ."""
        context = _context(long_format=_long_format(feed_sulfur_percent=None))
        proposal = optimization.propose(
            context, quality.assess(context), reliability.assess(context)
        )
        assert proposal.candidates == []
        assert "нет опубликованного анализа" in proposal.refusal["reason"]

    def test_stale_feed_analysis_stops_blending(self):
        context = _context(long_format=_long_format(feed_sample="2025-11-01 10:00:00"))
        proposal = optimization.propose(
            context, quality.assess(context), reliability.assess(context)
        )
        assert proposal.candidates == []
        assert "старше" in proposal.refusal["reason"]

    def test_stable_period_recommends_holding(self, scenario_context):
        proposal = self._proposal(scenario_context)
        assert proposal.recommended_id == "hold"

    def test_refuses_when_quality_refused(self):
        context = _context(long_format=_long_format(last_sample="2026-01-30 10:00:00"))
        proposal = optimization.propose(
            context, quality.assess(context), reliability.assess(context)
        )
        assert proposal.candidates == []
        assert proposal.recommended_id is None
        assert proposal.refusal is not None


def _submission(context) -> tuple[dict, dict]:
    """Пакет для оркестратора и независимые выводы ролей для сверки."""
    quality_assessment = quality.assess(context)
    reliability_assessment = reliability.assess(context)
    proposal = optimization.propose(context, quality_assessment, reliability_assessment)
    submission = optimization.build_submission(
        context, quality_assessment, reliability_assessment, proposal
    )
    sources = {
        "quality": quality_assessment.to_dict(),
        "reliability": reliability_assessment.to_dict(),
        "snapshot": context.snapshot,
    }
    return submission, sources


class TestOrchestrator:
    def test_approves_reproducible_recommendation(self, scenario_context):
        submission, sources = _submission(scenario_context)
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "approved"
        assert decision.recommendation["action"] == "hold"
        assert all(check["passed"] for check in decision.checks)

    def test_high_quality_risk_is_warned_even_when_approved(self, scenario_context):
        submission, sources = _submission(scenario_context)
        decision = orchestrator.review(submission, sources=sources)
        assert decision.warnings

    def test_better_economics_do_not_buy_a_violation(self, scenario_context):
        """Кандидат с лучшей экономикой, но с нарушением ограничения, отклонён."""
        submission, sources = _submission(scenario_context)
        submission["recommendation"] = submission["recommendation"] | {
            "action": "adjust_blend",
            "predicted_sulfur_mg_kg": 10.4,
            "evidence_type": "simulated_unvalidated",
            "blend_fractions": {"hydrotreated_diesel": 0.9995, "straight_run_240_350": 0.0005},
        }
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "rejected"
        assert decision.recommendation is None
        assert any("sulfur" in reason for reason in decision.rejection_reasons)

    def test_economic_claims_have_no_place_in_the_recommendation(self, scenario_context):
        """Выигрыш нельзя даже предъявить оркестратору — он его не принимает."""
        submission, sources = _submission(scenario_context)
        submission["recommendation"] = submission["recommendation"] | {
            "expected_cost_saving_pct": 12.0
        }
        with pytest.raises(ValueError, match="не принимает поля"):
            orchestrator.review(submission, sources=sources)

    def test_claimed_sulfur_must_survive_independent_recomputation(self, scenario_context):
        """Слово агента оптимизации не заменяет пересчёт по входу."""
        submission, sources = _submission(scenario_context)
        submission["recommendation"] = submission["recommendation"] | {
            "action": "adjust_blend",
            "predicted_sulfur_mg_kg": 9.0,
            "evidence_type": "simulated_unvalidated",
            "blend_fractions": {"hydrotreated_diesel": 0.99, "straight_run_240_350": 0.01},
        }
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "rejected"
        assert any("не совпадает с пересчитанной" in reason for reason in decision.rejection_reasons)
        check = next(item for item in decision.checks if item["check"] == "blend_sulfur_recomputed")
        assert check["passed"] is False

    def test_claimed_t95_must_survive_independent_recomputation(self, scenario_context):
        """Заниженный T95 в пакете не проходит: оркестратор считает смесь сам."""
        submission, sources = _submission(scenario_context)
        submission["recommendation"] = submission["recommendation"] | {"predicted_t95_c": 340.0}
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "rejected"
        check = next(item for item in decision.checks if item["check"] == "blend_t95_recomputed")
        assert check["passed"] is False

    def test_claimed_cetane_must_survive_independent_recomputation(self, scenario_context):
        submission, sources = _submission(scenario_context)
        submission["recommendation"] = submission["recommendation"] | {
            "predicted_cetane_number": 58.0
        }
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "rejected"
        check = next(item for item in decision.checks if item["check"] == "blend_cetane_recomputed")
        assert check["passed"] is False

    @pytest.mark.parametrize(
        ("key", "value"),
        [("t95_limit_c", 380.0), ("cetane_number_min", 45.0)],
    )
    def test_spec_limits_cannot_be_relaxed_in_the_package(self, scenario_context, key, value):
        """Смягчить предел подменой поля в пакете нельзя — как и предел серы."""
        submission, sources = _submission(scenario_context)
        submission["applied_constraints"] = submission["applied_constraints"] | {key: value}
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "rejected"
        check = next(item for item in decision.checks if item["check"] == f"{key}_unchanged")
        assert check["passed"] is False

    def test_component_indicator_removed_from_input_is_not_silently_skipped(
        self, scenario_context
    ):
        """Пропавший показатель компонента — причина отклонения, а не пропуск проверки."""
        submission, sources = _submission(scenario_context)
        components = {
            name: {key: item for key, item in value.items() if key != "t95_c"}
            for name, value in submission["input"]["components"].items()
        }
        submission["input"] = submission["input"] | {"components": components}
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "rejected"
        assert any("не воспроизводится" in reason for reason in decision.rejection_reasons)

    def test_agent_cannot_widen_its_own_ranges(self, scenario_context):
        submission, sources = _submission(scenario_context)
        widened = {tag: [-1e9, 1e9] for tag in features.TELEMETRY_TAGS}
        submission["recommendation"] = submission["recommendation"] | {"allowed_ranges": widened}
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "rejected"
        assert any("диапазоны" in reason for reason in decision.rejection_reasons)

    def test_setpoint_outside_applied_range_is_rejected(self, scenario_context):
        submission, sources = _submission(scenario_context)
        setpoints = dict(submission["recommendation"]["setpoints"])
        setpoints["T11"] = 10_000.0
        submission["recommendation"] = submission["recommendation"] | {
            "action": "adjust_setpoint",
            "setpoints": setpoints,
        }
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "rejected"
        assert any("вне применённого диапазона" in reason for reason in decision.rejection_reasons)

    def test_no_admissible_option_gives_explicit_refusal(self):
        context = _context(long_format=_long_format(last_sample="2026-01-30 10:00:00"))
        submission, sources = _submission(context)
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "no_admissible_recommendation"
        assert decision.recommendation is None
        assert decision.rejection_reasons
        assert "Допустимой рекомендации нет" in decision.explanation

    def test_ranges_widened_consistently_in_both_places_are_caught(self, scenario_context):
        """Согласованная ложь — не «две сверенные копии», а одна подделка."""
        submission, sources = _submission(scenario_context)
        widened = {tag: [200.0, 500.0] for tag in features.TELEMETRY_TAGS}
        setpoints = dict(submission["recommendation"]["setpoints"]) | {"T11": 480.0}
        submission["recommendation"] = submission["recommendation"] | {
            "action": "adjust_setpoint",
            "allowed_ranges": widened,
            "setpoints": setpoints,
        }
        submission["applied_constraints"] = submission["applied_constraints"] | {
            "allowed_ranges": widened
        }
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "rejected"
        assert any("агент надёжности" in reason for reason in decision.rejection_reasons)

    def test_component_sulfur_must_match_the_quality_agent(self, scenario_context):
        """Пересчёт баланса не спасает, если занижено само число во входе."""
        submission, sources = _submission(scenario_context)
        components = {
            name: dict(item) for name, item in submission["input"]["components"].items()
        }
        base = "hydrotreated_diesel"
        components[base]["sulfur_mg_kg"] = 3.0
        submission["input"] = submission["input"] | {"components": components}
        submission["recommendation"] = submission["recommendation"] | {
            "predicted_sulfur_mg_kg": 3.0
        }
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "rejected"
        assert any("агента качества" in reason for reason in decision.rejection_reasons)

    def test_hold_that_changes_setpoints_is_not_a_hold(self, scenario_context):
        """Ярлык «hold» не должен отключать проверки, положенные действию."""
        submission, sources = _submission(scenario_context)
        setpoints = dict(submission["recommendation"]["setpoints"])
        setpoints["T11"] = float(sources["snapshot"]["T11_value"]) - 5.0
        submission["recommendation"] = submission["recommendation"] | {
            "action": "hold",
            "setpoints": setpoints,
        }
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "rejected"
        assert any("не является сохранением режима" in reason for reason in decision.rejection_reasons)

    def test_quality_risk_cannot_be_softened_in_the_package(self, scenario_context):
        """Снять предупреждение подменой одного поля нельзя."""
        submission, sources = _submission(scenario_context)
        submission["applied_constraints"] = submission["applied_constraints"] | {
            "quality_risk_exceed_spec": "low",
            "forecast_backtest_rmse_mg_kg": 0.01,
        }
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "rejected"
        assert any("агента качества" in reason for reason in decision.rejection_reasons)

    def test_nested_reasoning_is_rejected_too(self, scenario_context):
        """Плоского allowlist мало: обоснование пролезало внутрь рекомендации."""
        submission, sources = _submission(scenario_context)
        submission["recommendation"] = submission["recommendation"] | {
            "rationale": "я перебрал 10 кандидатов, hold лучший по экономике"
        }
        with pytest.raises(ValueError, match="не принимает поля"):
            orchestrator.review(submission, sources=sources)

    def test_sources_are_required_for_a_verdict(self, scenario_context):
        """Без независимых выводов ролей воспроизвести ограничения нечем."""
        submission, _ = _submission(scenario_context)
        with pytest.raises(TypeError):
            orchestrator.review(submission)

    def test_limits_of_the_review_are_stated(self, scenario_context):
        """Чего проверка не доказывает — часть вывода, а не устная оговорка."""
        submission, sources = _submission(scenario_context)
        decision = orchestrator.review(submission, sources=sources)
        assert decision.limits
        assert any("прогноз" in item for item in decision.limits)

    def test_reasoning_of_executors_is_not_accepted(self, scenario_context):
        """Оркестратор — ревьюер, а не читатель чужих обоснований."""
        submission, sources = _submission(scenario_context)
        leaky = submission | {"candidates": [{"id": "hold", "why": "мне кажется, так лучше"}]}
        with pytest.raises(ValueError, match="не принимает поля"):
            orchestrator.review(leaky, sources=sources)


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


class TestSubmissionConsistency:
    """Пакет описывает один момент: риск и ошибка прогноза с одного горизонта."""

    def test_risk_and_rmse_come_from_the_same_worst_horizon(self):
        # Горизонты с разной ошибкой: на ближайшем запас укладывается в RMSE,
        # на худшем — нет. Дефект, найденный оркестратором на сценарии 4.
        backtest = {
            0.0: BACKTEST[0.0],
            2.0: {
                "shrunk_persistence": BACKTEST[0.0]["shrunk_persistence"] | {"rmse": 9.5},
                "persistence": BACKTEST[0.0]["persistence"] | {"rmse": 9.5},
                "ridge_telemetry": BACKTEST[0.0]["ridge_telemetry"],
            },
        }
        context = _context(backtest=backtest)
        quality_assessment = quality.assess(context)
        reliability_assessment = reliability.assess(context)
        proposal = optimization.propose(context, quality_assessment, reliability_assessment)
        submission = optimization.build_submission(
            context, quality_assessment, reliability_assessment, proposal
        )
        applied = submission["applied_constraints"]
        worst = max(quality_assessment.forecast, key=lambda item: item["backtest_rmse_mg_kg"])
        assert applied["forecast_backtest_rmse_horizon_hours"] == worst["horizon_hours"]
        assert applied["quality_risk_exceed_spec"] == worst["risk_exceed_spec"]

    def test_orchestrator_rejects_a_package_mixing_two_horizons(self):
        """Подмена класса риска на более мягкий не проходит ревью."""
        context = _context()
        submission, sources = _submission(context)
        submission["applied_constraints"]["quality_risk_exceed_spec"] = "moderate"
        decision = orchestrator.review(submission, sources=sources)
        assert decision.verdict == "rejected"
        assert any("оценка риска" in reason for reason in decision.rejection_reasons)
