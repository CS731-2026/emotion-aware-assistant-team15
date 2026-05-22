#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="127.0.0.1"
PORT="8004"
FORCE=0

usage() {
  cat <<USAGE
Usage: scripts/dev_restart_server.sh [--port PORT] [--host HOST] [--force]

Finds local project web server processes, optionally stops them, then starts:
  python -u main.py --mode web --host HOST --port PORT
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --force|-f)
      FORCE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$PORT" ] || [ -z "$HOST" ]; then
  echo "Host and port are required." >&2
  exit 2
fi

find_project_server_pids() {
  ps -eo pid=,args= | while read -r pid args; do
    case "$args" in
      *main.py*--mode\ web*)
        cwd=""
        if command -v pwdx >/dev/null 2>&1; then
          cwd="$(pwdx "$pid" 2>/dev/null | cut -d: -f2- | sed 's/^ //')"
        fi
        if [ -z "$cwd" ] || [ "$cwd" = "$PROJECT_ROOT" ]; then
          printf '%s\n' "$pid"
        fi
        ;;
    esac
  done
}

mapfile -t PIDS < <(find_project_server_pids)

"$PROJECT_ROOT/scripts/check_ports.sh" 8000 8002 8004 || true

if [ "${#PIDS[@]}" -gt 0 ]; then
  echo "Found project web server process(es): ${PIDS[*]}"
  if [ "$FORCE" -ne 1 ]; then
    read -r -p "Stop these processes before restarting? [y/N] " answer
    case "$answer" in
      y|Y|yes|YES) ;;
      *)
        echo "Leaving existing processes running."
        exit 0
        ;;
    esac
  fi
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping PID $pid"
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
fi

cd "$PROJECT_ROOT"
echo "Starting web app at http://${HOST}:${PORT}"
exec python -u main.py --mode web --host "$HOST" --port "$PORT"
