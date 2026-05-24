#!/usr/bin/env bash
set -euo pipefail

PORTS=("$@")
if [ "${#PORTS[@]}" -eq 0 ]; then
  PORTS=(8000 8002 8004)
fi

for port in "${PORTS[@]}"; do
  echo "== Port ${port} =="
  if command -v lsof >/dev/null 2>&1; then
    output="$(lsof -nP -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)"
    if [ -n "$output" ]; then
      echo "$output"
      continue
    fi
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | awk -v suffix=":${port}" '$4 ~ suffix "$" { print }'
  else
    echo "Neither lsof nor ss is available."
  fi
done
