#!/usr/bin/env python3
"""Сценарий 3: неполные и устаревшие данные — отказ вместо догадки.

Прогон той же цепочки ролей на окне, где последний опубликованный анализ
ЛИМС старше порога 48 ч (ADR-010, §2). Запускать после scripts/prepare_data.py
и scripts/evaluate_forecast.py:

    python scripts/scenario_stale_data.py

Окно выбрано по свойству данных и зафиксировано в ADR-011: середина
крупнейшего разрыва публикации ЛИМС внутри test-периода (350 ч, с
2026-06-18 02:00 по 2026-07-02 16:00). На момент выдачи 2026-06-25 09:00
возраст последнего анализа — 175 ч, телеметрия при этом живая.

Ожидаемый результат — отказ, а не рекомендация: агент качества не строит
persistence на устаревшем анализе, агент оптимизации без прогноза качества
не формирует кандидатов, оркестратор выдаёт no_admissible_recommendation.

Вывод: build/scenario_stale_data/. Случайности в прогоне нет, два запуска
подряд дают побайтово одинаковые файлы.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neftekod import scenarios  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(scenarios.main("stale_data"))
