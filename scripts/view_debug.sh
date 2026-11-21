#!/usr/bin/env bash
set -euo pipefail

url="${1:-http://127.0.0.1:8080/}"

if ! curl -fsS "${url}health" >/dev/null; then
  echo "调试网页未运行，请先启动 live.launch.py 或 sim.launch.py" >&2
  exit 1
fi

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$url" >/dev/null 2>&1 &
else
  echo "$url"
fi
