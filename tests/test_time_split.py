"""Границы train/test (ADR-006) не должны разъезжаться молча."""

import pandas as pd
import pytest

from neftekod import splits


def test_embargo_covers_longest_feature_window():
    """Зазор не короче самого длинного окна признака — иначе утечка на стыке."""
    gap = splits.TEST_START - splits.TRAIN_END
    assert gap >= pd.Timedelta(days=7)
    assert splits.EMBARGO >= pd.Timedelta(days=7)


def test_train_and_test_do_not_overlap():
    stamps = pd.date_range("2023-01-01", "2026-08-11", freq="6h")
    frame = pd.DataFrame({"timestamp": stamps, "value": range(len(stamps))})
    parts = splits.split(frame)

    assert parts["train"]["timestamp"].max() <= splits.TRAIN_END
    assert parts["test"]["timestamp"].min() >= splits.TEST_START
    assert set(parts["train"].index) & set(parts["test"].index) == set()
    # Строки зазора не попадают ни в одну часть.
    assert len(parts["train"]) + len(parts["test"]) < len(frame)


def test_validation_lives_inside_train():
    stamps = pd.date_range("2023-01-01", "2026-08-11", freq="6h")
    frame = pd.DataFrame({"timestamp": stamps})
    inner = splits.train_valid_split(frame)

    assert inner["fit"]["timestamp"].max() < inner["valid"]["timestamp"].min()
    assert inner["valid"]["timestamp"].max() <= splits.TRAIN_END


def test_assert_no_leakage_rejects_future_rows():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01", "2026-01-05"])})
    splits.assert_no_leakage(frame, "2026-01-05")
    with pytest.raises(ValueError, match="утечка из будущего"):
        splits.assert_no_leakage(frame, "2026-01-02")


def test_real_data_split_is_non_degenerate(lims_long, pak_long):
    """На реальных данных обе части непустые и содержат ключевые показатели."""
    frame = pd.concat([lims_long, pak_long], ignore_index=True)
    parts = splits.split(frame)

    for part in ("train", "test"):
        tags = set(parts[part]["tag"])
        # ADR-006, п. 1–3: оба показателя должны быть по обе стороны отсечки.
        assert "24-2000:Mg.Sulfur" in tags
        assert "24-2000:D15" in tags
        assert "LIMS:Гидроочистка.2.Mg.Sulfur" in tags
