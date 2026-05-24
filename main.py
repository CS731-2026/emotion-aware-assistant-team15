from __future__ import annotations

import argparse

from emotion_aware_assistant.core.config import DEFAULT_CONFIG


def main() -> int:
    parser = argparse.ArgumentParser(description="Emotion-Aware Academic Assistant")
    parser.add_argument("--mode", choices=["web", "gui", "terminal"], default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    mode = args.mode or DEFAULT_CONFIG["app"]["default_mode"]
    if mode == "terminal":
        from emotion_aware_assistant.cli import run_terminal

        return run_terminal(args.config)
    if mode == "web":
        from emotion_aware_assistant.web.server import run_web_server

        run_web_server(host=args.host, port=args.port, config_path=args.config)
        return 0
    from emotion_aware_assistant.ui.gui_app import run_gui

    return run_gui(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
