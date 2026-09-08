"""Отчёт покрытия и актуальности по лабораторным и поточным показателям.

Требование ТЗ (§2): нужно знать не только последнее доступное значение
ЛИМС/ПАК, но и сколько времени прошло с момента измерения и насколько ряд
вообще покрывает период. Отчёт отвечает на это по каждому показателю:
где начинается охват, где кончается, с каким шагом и какой самый длинный
разрыв внутри.

Отдельная колонка `lag_from_dataset_start_days` показывает, что часть
показателей появляется намного позже начала выгрузки (`24-2000:D15` —
только с 2025-03-05 при данных с 2023-01-01). Это факт данных, а не
пропуск: интерполировать через такой разрыв нельзя.
"""

from __future__ import annotations

import pandas as pd

from . import config, parse_lims_pak

REPORT_COLUMNS = [
    "tag",
    "source",
    "sample_point",
    "unit_declared",
    "n_values",
    "n_non_numeric",
    "first_seen",
    "last_seen",
    "lag_from_dataset_start_days",
    "span_days",
    "median_interval_h",
    "max_gap_days",
    "gap_ratio",
]


def build_coverage_report(long_format: pd.DataFrame) -> pd.DataFrame:
    dataset_start = long_format["timestamp"].min()
    rows = []
    for tag, group in long_format.groupby("tag", sort=True):
        series = group.sort_values("timestamp")
        stamps = series["timestamp"]
        deltas = stamps.diff().dropna()
        span = stamps.max() - stamps.min()
        median_interval = deltas.median() if len(deltas) else pd.NaT
        max_gap = deltas.max() if len(deltas) else pd.NaT
        rows.append(
            {
                "tag": tag,
                "source": series["source"].iloc[0],
                "sample_point": series["sample_point"].iloc[0],
                "unit_declared": series["unit_declared"].iloc[0],
                "n_values": len(series),
                "n_non_numeric": int(series["value"].isna().sum()),
                "first_seen": stamps.min(),
                "last_seen": stamps.max(),
                "lag_from_dataset_start_days": round(
                    (stamps.min() - dataset_start).total_seconds() / 86400, 1
                ),
                "span_days": round(span.total_seconds() / 86400, 1),
                "median_interval_h": (
                    round(median_interval.total_seconds() / 3600, 2)
                    if pd.notna(median_interval)
                    else None
                ),
                "max_gap_days": (
                    round(max_gap.total_seconds() / 86400, 1) if pd.notna(max_gap) else None
                ),
                # Во сколько раз самый длинный разрыв больше типичного шага:
                # чем выше, тем сильнее «дырявый» ряд отличается от регулярного.
                "gap_ratio": (
                    round(max_gap / median_interval, 1)
                    if pd.notna(max_gap) and pd.notna(median_interval) and median_interval.total_seconds()
                    else None
                ),
            }
        )
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def freshness(long_format: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Возраст последнего значения каждого показателя на момент `as_of`.

    Используется агентами: устаревший лабораторный анализ — основание для
    отказа от рекомендации, а не повод «подставить последнее известное».
    """
    last = long_format[long_format["timestamp"] <= as_of].groupby("tag")["timestamp"].max()
    frame = last.rename("last_value_ts").reset_index()
    frame["age_hours"] = (as_of - frame["last_value_ts"]).dt.total_seconds() / 3600
    return frame.sort_values("age_hours", ascending=False).reset_index(drop=True)


def to_markdown(report: pd.DataFrame, conflicts: pd.DataFrame) -> str:
    late = report[report["lag_from_dataset_start_days"] > 30].sort_values(
        "lag_from_dataset_start_days", ascending=False
    )
    lines = [
        "# Покрытие и актуальность показателей ЛИМС/ПАК",
        "",
        f"Показателей: {len(report)} "
        f"(ЛИМС {(report['source'] == 'lims').sum()}, ПАК {(report['source'] == 'pak').sum()}). "
        f"Период выгрузки: {report['first_seen'].min():%Y-%m-%d} … {report['last_seen'].max():%Y-%m-%d}.",
        "",
        "## Показатели, появляющиеся позже начала выгрузки (>30 сут)",
        "",
        "Это факт данных, а не пропуск: показатель физически не измерялся раньше.",
        "Интерполяция через такой разрыв запрещена (CLAUDE.md, «Данные — известные ловушки»).",
        "",
        late[["tag", "source", "first_seen", "last_seen", "n_values", "lag_from_dataset_start_days"]]
        .to_markdown(index=False),
        "",
        "## Конфликты единиц измерения (ADR-005)",
        "",
        conflicts.to_markdown(index=False) if len(conflicts) else "нет",
        "",
        "## Полный отчёт",
        "",
        report.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    long_format = parse_lims_pak.build_long_format()
    report = build_coverage_report(long_format)
    conflicts = parse_lims_pak.unit_conflicts(long_format)
    config.BUILD_DIR.mkdir(exist_ok=True)
    report.to_csv(config.COVERAGE_CSV, index=False)
    config.COVERAGE_MD.write_text(to_markdown(report, conflicts), encoding="utf-8")
    print(f"{len(report)} показателей → {config.COVERAGE_CSV}, {config.COVERAGE_MD}")
