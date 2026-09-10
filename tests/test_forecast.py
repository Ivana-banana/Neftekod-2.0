"""Проверки прогноза с доступностью источников и временным split."""

import pandas as pd
import pytest

from neftekod import forecast


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
    assert set(report["model"]) == {"persistence", "ridge"}
    assert set(report["evidence_type"]) == {"backtested"}
    assert set(report["horizon_hours"]) == {0, 1}
    assert (report["n"] > 0).all()
