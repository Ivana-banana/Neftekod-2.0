"""Проверки прогноза с доступностью источников и временным split."""

import numpy as np
import pandas as pd
import pytest

from neftekod import config, forecast


def _observations():
    return pd.DataFrame(
        [
            # Предыдущий анализ доступен с задержкой четыре часа.
            {
                "timestamp": "2025-01-01 00:00",
                "available_at": "2025-01-01 04:00",
                "source": "lims",
                "tag": "LIMS:Гидроочистка.2.Mg.Sulfur",
                "value": 8.0,
            },
            # Это целевая строка: на момент прогноза она ещё не опубликована.
            {
                "timestamp": "2025-01-01 10:00",
                "available_at": "2025-01-01 14:00",
                "source": "lims",
                "tag": "LIMS:Гидроочистка.2.Mg.Sulfur",
                "value": 9.0,
            },
            {
                "timestamp": "2025-01-01 08:00",
                "available_at": "2025-01-01 08:00",
                "source": "pak",
                "tag": "24-2000:Mg.Sulfur",
                "value": 8.5,
            },
            # Измерено до момента выдачи, но опубликовано после него.
            {
                "timestamp": "2025-01-01 09:00",
                "available_at": "2025-01-01 10:30",
                "source": "pak",
                "tag": "24-2000:Mg.Sulfur",
                "value": 8.9,
            },
        ]
    )


def test_prediction_frame_uses_only_values_available_at_issue_time():
    frame = forecast.build_prediction_frame(_observations(), horizon_hours=1)

    row = frame.iloc[0]
    assert row["target_time"] == pd.Timestamp("2025-01-01 10:00")
    assert row["issue_time"] == pd.Timestamp("2025-01-01 09:00")
    assert row["target_value"] == pytest.approx(9.0)
    assert row["baseline_prediction"] == pytest.approx(8.0)
    assert row["baseline_age_hours"] == pytest.approx(9.0)
    assert row["pak_value"] == pytest.approx(8.5)
    assert row["pak_age_hours"] == pytest.approx(1.0)


def test_score_predictions_reports_hand_checked_metrics():
    actual = pd.Series([9.0, 11.0])
    predicted = pd.Series([8.0, 10.0])

    score = forecast.score_predictions(actual, predicted)

    assert score["n"] == 2
    assert score["mae"] == pytest.approx(1.0)
    assert score["rmse"] == pytest.approx(1.0)
    assert score["r2"] == pytest.approx(0.0)


def test_prediction_frame_ignores_non_finite_observations():
    rows = _observations()
    rows.loc[len(rows)] = {
        "timestamp": "2025-01-01 11:00",
        "available_at": "2025-01-01 11:00",
        "source": "pak",
        "tag": "24-2000:Mg.Sulfur",
        "value": float("inf"),
    }
    rows.loc[len(rows)] = {
        "timestamp": "2025-01-01 12:00",
        "available_at": "2025-01-01 16:00",
        "source": "lims",
        "tag": "LIMS:Гидроочистка.2.Mg.Sulfur",
        "value": float("-inf"),
    }

    prediction = forecast.build_prediction_frame(rows, horizon_hours=1)

    assert len(prediction) == 1
    assert prediction.iloc[0]["pak_value"] == pytest.approx(8.5)


def test_prediction_frame_adds_asof_telemetry_features():
    rows = _observations()
    telemetry = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01 07:50", "2025-01-01 09:10"]),
            "P8": [1.0, 2.0],
            "T11": [10.0, 20.0],
            "F19": [100.0, 200.0],
        }
    )

    prediction = forecast.build_prediction_frame(rows, horizon_hours=1, telemetry=telemetry)

    row = prediction.iloc[0]
    assert row["issue_time"] == pd.Timestamp("2025-01-01 09:00")
    assert row["P8_value"] == pytest.approx(1.0)
    assert row["T11_value"] == pytest.approx(10.0)
    assert row["F19_value"] == pytest.approx(100.0)
    assert row["P8_age_hours"] == pytest.approx(1 + 10 / 60)


def test_evaluate_forecasts_labels_backtest_and_keeps_train_valid_test_separate():
    rows = []
    for index, day in enumerate(pd.date_range("2023-01-01", "2026-01-10", freq="7D")):
        value = 7.0 + index * 0.01
        rows.extend(
            [
                {
                    "timestamp": day,
                    "available_at": day + pd.Timedelta(hours=4),
                    "source": "lims",
                    "tag": "LIMS:Гидроочистка.2.Mg.Sulfur",
                    "value": value,
                },
                {
                    "timestamp": day,
                    "available_at": day,
                    "source": "pak",
                    "tag": "24-2000:Mg.Sulfur",
                    "value": value + 0.1,
                },
            ]
        )
    report = forecast.evaluate_forecasts(pd.DataFrame(rows), horizons=(0, 1))

    assert set(report["split"]) == {"fit", "valid", "test"}
    assert set(report["model"]) == {"persistence", "shrunk_persistence", "ridge"}
    assert set(report["evidence_type"]) == {"backtested"}
    assert set(report["horizon_hours"]) == {0, 1}
    assert (report["n"] > 0).all()


def test_telemetry_evaluation_reports_all_candidate_lags_but_scores_one_on_test():
    rows = []
    days = pd.date_range("2023-01-01", "2026-01-10", freq="7D")
    for index, day in enumerate(days):
        value = 7.0 + index * 0.01
        rows.extend(
            [
                {
                    "timestamp": day,
                    "available_at": day + pd.Timedelta(hours=4),
                    "source": "lims",
                    "tag": "LIMS:Гидроочистка.2.Mg.Sulfur",
                    "value": value,
                },
                {
                    "timestamp": day,
                    "available_at": day,
                    "source": "pak",
                    "tag": "24-2000:Mg.Sulfur",
                    "value": value + 0.1,
                },
            ]
        )
    telemetry = pd.DataFrame(
        {
            "date": days,
            "P8": range(len(days)),
            "T11": [100 + index for index in range(len(days))],
            "F19": [200 + index for index in range(len(days))],
        }
    )

    report = forecast.evaluate_forecasts(
        pd.DataFrame(rows),
        horizons=(0,),
        telemetry=telemetry,
        telemetry_lags=(0, 1, 2, 3),
    )

    ridge = report[report["model"] == "ridge_telemetry"]
    assert set(ridge["telemetry_lag_hours"].dropna()) == {0.0, 1.0, 2.0, 3.0}
    assert len(ridge[ridge["split"] == "test"]) == 1
    assert ridge[ridge["split"] == "test"]["selected"].eq(True).all()
    assert report["evidence_type"].eq("backtested").all()
    assert {"lims_age_hours_median", "pak_age_hours_median", "P8_age_hours_median"} <= set(report)


def _baseline_frame(values: list[float], start: str = "2023-01-01") -> pd.DataFrame:
    """Кадр цели/baseline с суточным шагом; split по времени как в ADR-006."""
    stamps = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "target_time": stamps,
            "issue_time": stamps,
            "target_value": values,
            "baseline_prediction": [values[0]] + values[:-1],
            "baseline_age_hours": 24.0,
            "pak_age_hours": 24.0,
            "pak_missing": False,
        }
    )


def test_shrinkage_is_selected_on_valid_and_prefers_the_fresh_analysis_when_it_helps():
    """Если последний анализ действительно предсказывает следующий — shrinkage 1."""
    # Монотонный ряд: persistence точен, константа ошибается систематически.
    frame = _baseline_frame([5.0 + 0.01 * index for index in range(1400)])
    parts = forecast._split_rows(frame)
    assert forecast.select_shrinkage(parts["fit"], parts["valid"]) == 1.0


def test_shrinkage_is_zero_when_the_fresh_analysis_is_pure_noise():
    """Шум вокруг уровня: свежий анализ не помогает, выбирается константа."""
    generator = np.random.default_rng(0)
    values = list(8.5 + generator.normal(0.0, 2.0, 1400))
    frame = _baseline_frame(values)
    parts = forecast._split_rows(frame)
    assert forecast.select_shrinkage(parts["fit"], parts["valid"]) <= 0.3


def test_test_split_never_participates_in_shrinkage_selection():
    """Подмена test не меняет выбор: отбор идёт только на fit/valid."""
    values = [5.0 + 0.01 * index for index in range(1400)]
    frame = _baseline_frame(values)
    parts = forecast._split_rows(frame)
    chosen = forecast.select_shrinkage(parts["fit"], parts["valid"])
    broken_test = frame.copy()
    mask = pd.to_datetime(broken_test["target_time"]) >= pd.Timestamp(config.TEST_START)
    broken_test.loc[mask, "target_value"] = 1000.0
    parts_broken = forecast._split_rows(broken_test)
    assert forecast.select_shrinkage(parts_broken["fit"], parts_broken["valid"]) == chosen


def test_report_carries_baseline_parameters_needed_to_reproduce_the_forecast():
    values = [8.0 + 0.5 * ((index % 5) - 2) for index in range(1400)]
    frame = _baseline_frame(values)
    rows = forecast._shrunk_baseline_rows(0.0, forecast._split_rows(frame))
    report = pd.DataFrame(rows)
    test_rows = report[report["split"].eq("test")]
    assert len(test_rows) == 1, "на test оценивается ровно один выбранный вариант"
    assert bool(test_rows.iloc[0]["selected"]) is True
    assert test_rows.iloc[0]["baseline_level_mg_kg"] > 0
    assert report[report["split"].eq("valid")]["selected"].sum() == 1
