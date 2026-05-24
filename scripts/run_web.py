from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_aware_assistant.web.server import run_web_server


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the web-first assistant.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_web_server(host=args.host, port=args.port, config_path=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
