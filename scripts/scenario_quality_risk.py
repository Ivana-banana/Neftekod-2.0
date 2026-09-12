#!/usr/bin/env python3
"""Сценарий 2: период с риском ухудшения качества.

Прогон той же цепочки ролей на окне, где сера товарного ДТ быстро выросла.
Запускать после scripts/prepare_data.py и scripts/evaluate_forecast.py:

    python scripts/scenario_quality_risk.py

Окно выбрано по свойству данных и зафиксировано в ADR-011: максимальный рост
серы за 72 ч среди моментов test-периода, лежащих внутри полосы p05–p95
обучающей выборки, при допустимом режиме — с 3.2 до 9.9 мг/кг к моменту
выдачи 2026-05-12 14:00 (severity_index 0.124).

Ожидаемое поведение — не заранее заданный вердикт, а видимый конфликт целей:
агент качества даёт запас меньше ошибки прогноза, агент оптимизации предлагает
экономически выгодные варианты, и они отклоняются по качеству.

Вывод: build/scenario_quality_risk/. Случайности в прогоне нет, два запуска
подряд дают побайтово одинаковые файлы.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neftekod import scenarios  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(scenarios.main("quality_risk"))
