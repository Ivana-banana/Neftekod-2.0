"""Контракт синхронизации ЛИМС/ПАК/телеметрии по времени."""

import pandas as pd
import pytest

from neftekod import features


TARGET = "LIMS:Гидроочистка.2.Mg.Sulfur"
PAK = "24-2000:Mg.Sulfur"


def _quality_rows(rows):
    return pd.DataFrame(rows, columns=["timestamp", "available_at", "source", "tag", "value"])


def _telemetry_rows(rows):
    return pd.DataFrame(rows, columns=["date", "P8", "T11", "F19"])


def test_asof_sync_uses_time_not_row_position_and_preserves_each_source_age():
    quality = _quality_rows(
        [
            ("2026-01-01 06:00", "2026-01-01 10:00", "lims", TARGET, 8.0),
            # Измерено до issue_time, но опубликовано после него.
            ("2026-01-01 09:00", "2026-01-01 12:00", "lims", TARGET, 7.0),
            ("2026-01-01 08:40", "2026-01-01 08:40", "pak", PAK, 8.5),
            ("2026-01-01 09:40", "2026-01-01 10:30", "pak", PAK, 8.9),
        ]
    )
    telemetry = _telemetry_rows(
        [
            ("2026-01-01 08:00", 1.0, 10.0, 100.0),
            ("2026-01-01 09:50", 2.0, 20.0, 200.0),
            ("2026-01-01 10:10", 3.0, 30.0, 300.0),
        ]
    )

    result = features.synchronize_features(
        [pd.Timestamp("2026-01-01 10:00")], quality, telemetry
    ).iloc[0]

    assert result["lims_value"] == pytest.approx(8.0)
    assert result["lims_timestamp"] == pd.Timestamp("2026-01-01 06:00")
    assert result["lims_available_at"] == pd.Timestamp("2026-01-01 10:00")
    assert result["lims_age_hours"] == pytest.approx(4.0)
    assert result["pak_value"] == pytest.approx(8.5)
    assert result["pak_timestamp"] == pd.Timestamp("2026-01-01 08:40")
    assert result["pak_age_hours"] == pytest.approx(1 + 20 / 60)
    assert result["P8_value"] == pytest.approx(2.0)
    assert result["T11_value"] == pytest.approx(20.0)
    assert result["F19_value"] == pytest.approx(200.0)
    assert result["P8_age_hours"] == pytest.approx(10 / 60)
    assert result["T11_age_hours"] == pytest.approx(10 / 60)
    assert result["F19_age_hours"] == pytest.approx(10 / 60)


def test_exact_availability_boundary_is_allowed():
    quality = _quality_rows(
        [("2026-01-01 09:00", "2026-01-01 10:00", "lims", TARGET, 7.0)]
    )
    result = features.synchronize_features(
        [pd.Timestamp("2026-01-01 10:00")], quality, _telemetry_rows([])
    ).iloc[0]

    assert result["lims_value"] == pytest.approx(7.0)
    assert result["lims_missing"] is False


def test_telemetry_gap_is_carried_as_old_measurement_not_interpolated():
    quality = _quality_rows(
        [("2026-01-01 00:00", "2026-01-01 00:00", "lims", TARGET, 7.0)]
    )
    telemetry = _telemetry_rows(
        [
            ("2026-01-01 00:00", 10.0, 20.0, 30.0),
            ("2026-01-01 10:00", 30.0, 40.0, 50.0),
        ]
    )

    result = features.synchronize_features(
        [pd.Timestamp("2026-01-01 05:00")], quality, telemetry
    ).iloc[0]

    assert result["P8_value"] == pytest.approx(10.0)
    assert result["P8_age_hours"] == pytest.approx(5.0)


def test_nonfinite_values_are_ignored_and_missing_feature_is_explicit():
    quality = _quality_rows(
        [
            ("2026-01-01 00:00", "2026-01-01 00:00", "lims", TARGET, 7.0),
            ("2026-01-01 02:00", "2026-01-01 02:00", "pak", PAK, float("inf")),
        ]
    )
    telemetry = _telemetry_rows(
        [
            ("2026-01-01 01:00", float("inf"), float("nan"), -float("inf")),
        ]
    )

    result = features.synchronize_features(
        [pd.Timestamp("2026-01-01 03:00")], quality, telemetry
    ).iloc[0]

    assert pd.isna(result["pak_value"])
    assert result["pak_missing"] is True
    for tag in ("P8", "T11", "F19"):
        assert pd.isna(result[f"{tag}_value"])
        assert pd.isna(result[f"{tag}_age_hours"])
        assert result[f"{tag}_missing"] is True


def test_missing_quality_availability_is_rejected():
    quality = pd.DataFrame(
        [{"timestamp": "2026-01-01 00:00", "source": "lims", "tag": TARGET, "value": 7.0}]
    )

    with pytest.raises(ValueError, match="available_at"):
        features.synchronize_features(
            [pd.Timestamp("2026-01-01 01:00")], quality, _telemetry_rows([])
        )


def test_sync_is_reproducible_for_same_inputs():
    quality = _quality_rows(
        [("2026-01-01 00:00", "2026-01-01 00:00", "lims", TARGET, 7.0)]
    )
    telemetry = _telemetry_rows(
        [("2026-01-01 00:00", 10.0, 20.0, 30.0)]
    )
    issue_times = [pd.Timestamp("2026-01-01 01:00"), pd.Timestamp("2026-01-01 02:00")]

    first = features.synchronize_features(issue_times, quality, telemetry)
    second = features.synchronize_features(issue_times, quality, telemetry)

    pd.testing.assert_frame_equal(first, second)
