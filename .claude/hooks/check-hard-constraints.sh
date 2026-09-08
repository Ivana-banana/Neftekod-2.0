#!/usr/bin/env bash
# Stop hook: не даёт ходу агента считаться завершённым, если жёсткие
# ограничения (сера <=10 мг/кг, доли блендинга = 100%) не проходят тест.
# См. CLAUDE.md, "Работа с ИИ-ассистентом (Claude Code)".
set -uo pipefail

TEST_FILE="tests/test_hard_constraints.py"

if [ ! -f "$TEST_FILE" ]; then
  # Теста ещё нет — пропускаем, не блокируем работу на раннем этапе проекта.
  exit 0
fi

# Окружение проекта, а не системный pytest: тест импортирует пакет neftekod,
# который ставится в .venv (`pip install -e ".[dev]"`, см. README).
if [ -x ".venv/bin/pytest" ]; then
  PYTEST=".venv/bin/pytest"
else
  PYTEST="pytest"
fi

OUTPUT="$($PYTEST "$TEST_FILE" -q 2>&1)"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  TAIL="$(printf '%s' "$OUTPUT" | tail -n 40)"
  jq -n --arg file "$TEST_FILE" --arg out "$TAIL" \
    '{decision: "block", reason: ("Жёсткие ограничения не проходят тест (" + $file + "):\n" + $out)}'
fi

exit 0
