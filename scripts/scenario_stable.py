#!/usr/bin/env python3
"""Сценарий 1: устойчивый период без лишних действий.

Сквозной прогон «данные → оценки агентов → рекомендация» на реальных данных
с замороженным моментом выдачи. Запускать после scripts/prepare_data.py и
scripts/evaluate_forecast.py:

    python scripts/scenario_stable.py

Окно выбрано по свойствам режима, а не по тому, какой ответ получится: самая
спокойная неделя test-периода (сера ЛИМС 8.4–9.8 мг/кг, std 0.48; телеметрия
P8/T11/F19 со std 0.004 / 1.5 / 4.9). Момент выдачи лежит в test-периоде по
ADR-006, модели обучены только на train.

Вывод: build/scenario_stable/{context,quality,reliability,optimization,
submission,decision}.json и recommendation.md. Случайности в прогоне нет,
поэтому два запуска подряд дают побайтово одинаковые файлы.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neftekod import scenarios  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(scenarios.main("stable"))
