#!/usr/bin/env bash
# «Посылка по пути» — запуск одной командой.
#   ./run.sh                    весь конвейер на лучших доступных пунктах СДЭК
#   ./run.sh --points FILE      свой файл пунктов (JSON API СДЭК, Overpass JSON, CSV)
#   ./run.sh --open             то же и открыть карту-демо в браузере (локальный сервер)
#   ./run.sh demo               только открыть уже готовую карту
#   ./run.sh test               только тесты
# Остальные параметры: ./run.sh --help
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv
if [ ! -x "$VENV/bin/python" ]; then
  PY=""
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
      PY="$c"; break
    fi
  done
  [ -n "$PY" ] || { echo "Нужен Python 3.10+ (brew install python@3.12)"; exit 1; }
  echo "Создаю окружение $VENV ($PY)…"
  "$PY" -m venv "$VENV"
fi
if [ ! -f "$VENV/.deps-ok" ] || [ requirements.txt -nt "$VENV/.deps-ok" ]; then
  echo "Ставлю зависимости…"
  "$VENV/bin/pip" install -q -r requirements.txt -r requirements-dev.txt
  touch "$VENV/.deps-ok"
fi
export PYTHONPATH=src MPLBACKEND=Agg

if [ "${1:-}" = "demo" ]; then
  set -- --open
  SKIP_PIPELINE=1
fi
if [ "${1:-}" = "test" ]; then
  exec "$VENV/bin/python" -m pytest tests -q
fi

if [ ! -f data/processed/moscow_walk.graphml ]; then
  echo "Нет пешеходного графа, строю из OpenStreetMap (1–2 мин, нужен интернет)…"
  "$VENV/bin/python" scripts/01_build_graph.py
fi

OPEN=0
ARGS=()
for a in "$@"; do
  if [ "$a" = "--open" ]; then OPEN=1; else ARGS+=("$a"); fi
done
if [ -z "${SKIP_PIPELINE:-}" ]; then
  "$VENV/bin/python" scripts/run_all.py ${ARGS[@]+"${ARGS[@]}"}
fi

if [ "$OPEN" = 1 ]; then
  # serve over http: map tiles refuse pages opened straight from disk
  PORT=8877
  while lsof -i ":$PORT" >/dev/null 2>&1; do PORT=$((PORT + 1)); done
  URL="http://localhost:$PORT/"
  echo "Карта: $URL  (Ctrl+C — закрыть)"
  (sleep 1; open "$URL" 2>/dev/null || xdg-open "$URL" 2>/dev/null || true) &
  exec "$VENV/bin/python" -m http.server "$PORT" --bind 127.0.0.1 --directory demo
fi
