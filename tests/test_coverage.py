"""Отчёт покрытия обязан показывать разрывы охвата как факт данных."""

import pandas as pd

from neftekod import coverage


def test_d15_gap_is_visible_in_report(lims_long, pak_long):
    """`24-2000:D15` появляется только в 2025 при данных с 2023 — это в отчёте."""
    frame = pd.concat([lims_long, pak_long], ignore_index=True)
    report = coverage.build_coverage_report(frame).set_index("tag")

    d15 = report.loc["24-2000:D15"]
    assert d15["first_seen"] == pd.Timestamp("2025-03-05 10:20:00")
    # Больше двух лет после начала выгрузки — не «пропуск за выходные».
    assert d15["lag_from_dataset_start_days"] > 700
    assert report.loc["24-2000:Mg.Sulfur", "lag_from_dataset_start_days"] == 0.0


def test_report_covers_every_series(lims_long, pak_long):
    frame = pd.concat([lims_long, pak_long], ignore_index=True)
    report = coverage.build_coverage_report(frame)
    assert len(report) == frame["tag"].nunique() == 56
    assert report["n_values"].sum() == len(frame)
    assert report["n_non_numeric"].sum() == 6


def test_freshness_uses_only_past_values(lims_long):
    """Возраст анализа считается по значениям не позже момента оценки."""
    as_of = pd.Timestamp("2025-06-01")
    fresh = coverage.freshness(lims_long, as_of)
    assert (fresh["last_value_ts"] <= as_of).all()
    assert (fresh["age_hours"] >= 0).all()
    # Показатели, которых на эту дату ещё нет, в таблицу не попадают —
    # «последнее известное значение» для них не выдумывается.
    assert "LIMS:АВТ.1.CloudPoint_1" not in set(fresh["tag"])


def test_freshness_waits_for_publication_and_uses_sample_age():
    """В 11:59 свежим остаётся старый анализ; в 12:00 доступен утренний."""
    frame = pd.DataFrame({
        "tag": ["lab", "lab", "pak", "late"],
        "source": ["lims", "lims", "pak", "pak"],
        "timestamp": pd.to_datetime([
            "2026-01-01 00:00", "2026-01-01 08:00", "2026-01-01 11:50", "2026-01-02 00:00",
        ]),
        "available_at": pd.to_datetime([
            "2026-01-01 04:00", "2026-01-01 12:00", "2026-01-01 11:50", "2026-01-02 00:00",
        ]),
        "value": [9.0, 7.0, 8.0, 10.0],
    })
    before = coverage.freshness(frame, pd.Timestamp("2026-01-01 11:59:59")).set_index("tag")
    assert before.loc["lab", "last_value_ts"] == pd.Timestamp("2026-01-01 00:00")
    assert before.loc["pak", "last_value_ts"] == pd.Timestamp("2026-01-01 11:50")
    assert "late" not in before.index
    at = coverage.freshness(frame, pd.Timestamp("2026-01-01 12:00")).set_index("tag")
    assert at.loc["lab", "last_value_ts"] == pd.Timestamp("2026-01-01 08:00")
    assert at.loc["lab", "age_hours"] == 4.0


def test_freshness_does_not_treat_non_numeric_marker_as_new_analysis():
    frame = pd.DataFrame({
        "tag": ["lab", "lab"], "source": ["lims", "lims"],
        "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 08:00"]),
        "available_at": pd.to_datetime(["2026-01-01 04:00", "2026-01-01 12:00"]),
        "value": [9.0, float("nan")],
    })
    fresh = coverage.freshness(frame, pd.Timestamp("2026-01-01 12:00"))
    assert fresh.iloc[0]["last_value_ts"] == pd.Timestamp("2026-01-01 00:00")
