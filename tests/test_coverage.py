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
