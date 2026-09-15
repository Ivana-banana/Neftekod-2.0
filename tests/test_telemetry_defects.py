"""Дефекты телеметрии: маркер 307.0 и залипание, без заглядывания в будущее (ADR-018)."""

import numpy as np
import pandas as pd
import pytest

from neftekod import config, features, telemetry_defects

TAGS = ("P8", "T11", "F19")


def _frame(values: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(values.values())))
    frame = pd.DataFrame(values)
    frame.insert(0, "timestamp", pd.date_range("2026-01-01", periods=n, freq="10min"))
    return frame


def _real_fragment(start: str, end: str) -> pd.DataFrame:
    path = config.HT_TELEMETRY_CSV
    if not path.exists():
        pytest.skip(f"нет {path} — скачать по ссылке из README")
    frame = pd.read_csv(path, usecols=["date", *TAGS], parse_dates=["date"])
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    return frame.rename(columns={"date": "timestamp"}).reset_index(drop=True)


def test_real_f19_sentinel_run_is_masked_and_values_before_it_are_kept():
    # Серия 307.0 у F19 начинается 2024-03-18 10:20 (ADR-007, ADR-018).
    frame = _real_fragment("2024-03-18 09:00", "2024-03-18 12:00")
    before = frame["timestamp"] < pd.Timestamp("2024-03-18 10:20")
    assert (frame.loc[~before, "F19"] == 307.0).all()

    masked = telemetry_defects.mask_defects(frame, TAGS)

    assert masked.loc[~before, "F19"].isna().all()
    pd.testing.assert_series_equal(masked.loc[before, "F19"], frame.loc[before, "F19"])


def test_real_t11_run_of_251_is_masked_from_sixth_row():
    frame = _real_fragment("2024-04-05 03:00", "2024-04-05 12:00")
    run = frame["T11"] == 251.0
    assert run.sum() == 45

    masks = telemetry_defects.defect_masks(frame, TAGS)

    run_positions = np.flatnonzero(run.to_numpy())
    assert not masks["frozen"]["T11"].iloc[run_positions[:5]].any()
    assert masks["frozen"]["T11"].iloc[run_positions[5:]].all()


def test_frozen_detection_does_not_look_into_the_future():
    """Маска на обрезанном ряде совпадает с началом маски на полном ряде."""
    values = [1.0, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0]
    frame = _frame({"P8": values, "T11": values, "F19": values})
    full = telemetry_defects.defect_masks(frame, TAGS)
    for cut in range(1, len(values) + 1):
        prefix = telemetry_defects.defect_masks(frame.iloc[:cut], TAGS)
        for code in telemetry_defects.DEFECT_CODES:
            pd.testing.assert_frame_equal(prefix[code], full[code].iloc[:cut])
    # Серия 5.0 из 8 строк: помечены только строки начиная с шестой.
    assert full["frozen"]["P8"].tolist() == [False] * 7 + [True] * 3 + [False]


def test_short_repeats_and_negative_values_are_not_defects():
    frame = _frame({"P8": [-0.001, -0.001, 0.2], "T11": [360.0, 360.0, 361.0], "F19": [-0.1, 214.0, 214.0]})
    masked = telemetry_defects.mask_defects(frame, TAGS)
    pd.testing.assert_frame_equal(masked, frame)


def test_synchronized_features_skip_sentinel_and_age_grows():
    quality = pd.DataFrame(columns=["timestamp", "available_at", "source", "tag", "value"])
    telemetry = pd.DataFrame(
        {
            "date": ["2026-01-01 08:00", "2026-01-01 09:00", "2026-01-01 09:50"],
            "P8": [0.2, 0.21, 0.22],
            "T11": [360.0, 361.0, 362.0],
            "F19": [214.0, 307.0, 307.0],
        }
    )
    row = features.synchronize_features([pd.Timestamp("2026-01-01 10:00")], quality, telemetry).iloc[0]
    assert row["F19_value"] == pytest.approx(214.0)
    assert row["F19_age_hours"] == pytest.approx(2.0)



def test_raw_loader_keeps_markers_so_context_can_count_them_and_loader_masks_them():
    """Контекст считает дефекты по сырому ряду: по очищенному он насчитал бы ноль."""
    if not config.HT_TELEMETRY_CSV.exists():
        pytest.skip(f"нет {config.HT_TELEMETRY_CSV} — скачать по ссылке из README")
    raw = features.load_raw_telemetry()
    masked = features.load_telemetry()
    assert (raw["F19"] == 307.0).sum() == 2324
    assert (masked["F19"] == 307.0).sum() == 0
