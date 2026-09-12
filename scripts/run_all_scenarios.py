#!/usr/bin/env python3
"""Прогон всех демонстрационных сценариев одной командой.

    python scripts/run_all_scenarios.py

Сценарии 1–4 прогоняются подряд на своих замороженных окнах (ADR-010,
ADR-011), каждый пишет свой каталог build/scenario_*/ и свою карточку
оператора. В конце печатается сводка вердиктов — она же служит быстрым
контролем перед демонстрацией.

Случайности в прогоне нет: повторный запуск даёт побайтово те же файлы.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neftekod import config, scenarios  # noqa: E402

log = logging.getLogger("run_all_scenarios")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for path in (config.LONG_FORMAT_CSV, config.FORECAST_CSV, config.HT_TELEMETRY_CSV):
        if not path.exists():
            log.error("нет %s; см. README, раздел «Как запустить»", path)
            return 1

    rows = []
    for name, spec in scenarios.SCENARIOS.items():
        run = scenarios.execute(spec)
        scenarios.write(run)
        # Тот же худший горизонт, что и в пакете для оркестратора (ADR-010, §3б):
        # сводка не должна показывать более мягкий класс риска, чем вердикт.
        forecast = (
            max(run.quality.forecast, key=lambda item: item["backtest_rmse_mg_kg"])
            if run.quality.forecast
            else None
        )
        recommended = run.proposal.recommended()
        rows.append(
            {
                "сценарий": name,
                "момент выдачи": spec.issue_time,
                "риск качества": "отказ" if forecast is None else forecast["risk_exceed_spec"],
                "запас, мг/кг": "—" if forecast is None else f"{forecast['margin_mg_kg']:.2f}",
                "severity": "отказ"
                if run.reliability.severity_index is None
                else f"{run.reliability.severity_index:.3f}",
                "кандидатов": len(run.proposal.candidates),
                "допустимых": sum(1 for item in run.proposal.candidates if item["admissible"]),
                "рекомендация": recommended["id"] if recommended else "—",
                "вердикт": run.decision.verdict,
            }
        )

    headers = list(rows[0])
    widths = [max(len(str(row[key])) for row in [dict(zip(headers, headers))] + rows) for key in headers]
    print("\nСводка прогона всех сценариев:\n")
    print("  " + "  ".join(name.ljust(width) for name, width in zip(headers, widths)))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print("  " + "  ".join(str(row[key]).ljust(width) for key, width in zip(headers, widths)))
    print(f"\nВывод сценариев — в {config.BUILD_DIR}/scenario_*/\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
