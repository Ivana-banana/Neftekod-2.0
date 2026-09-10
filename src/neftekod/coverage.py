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

from . import config, parse_lims_pak, splits

REPORT_COLUMNS = [
    "tag",
    "source",
    "sample_point",
    "unit_declared",
    "unit",
    "unit_basis",
    "availability_basis",
    "first_available_at",
    "last_available_at",
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
                "unit": series["unit"].iloc[0],
                "unit_basis": series["unit_basis"].iloc[0],
                "availability_basis": series["availability_basis"].iloc[0],
                "first_available_at": series["available_at"].min(),
                "last_available_at": series["available_at"].max(),
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
    """Возраст последнего доступного числового значения на момент `as_of`.

    Используется агентами: устаревший лабораторный анализ — основание для
    отказа от рекомендации, а не повод «подставить последнее известное».
    """
    known = splits.available_history(long_format, as_of)
    last = known.dropna(subset=["value"]).groupby("tag")["timestamp"].max()
    frame = last.rename("last_value_ts").reset_index()
    # Возраст от отбора, а не от публикации: свежедоступный анализ уже имеет
    # возраст до четырёх часов. См. ADR-006.
    frame["age_hours"] = (pd.Timestamp(as_of) - frame["last_value_ts"]).dt.total_seconds() / 3600
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
        "Показаны исходные заголовки и нормализованные единицы. Ошибка строки единиц",
        "ЛИМС подтверждена организаторами; массовая сера ПАК: 1 ppm = 1 мг/кг.",
        "Числа сохранены без масштабирования, расхождения заголовков оставлены для аудита.",
        "",
        conflicts.to_markdown(index=False) if len(conflicts) else "нет",
        "",
        "## Доступность для исторических признаков (ADR-006)",
        "",
        "`timestamp` — отбор/измерение; `available_at` — оценка времени доступности.",
        "ЛИМС: отбор + 4 часа (консервативно, точных публикаций нет).",
        "ПАК: время измерения (допущение, задержка не предоставлена).",
        "Возраст анализа считается от отбора. Лабораторные цели не сдвигаются.",
        "Все источники в одном часовом поясе, его название не задано; UTC не назначается.",
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
