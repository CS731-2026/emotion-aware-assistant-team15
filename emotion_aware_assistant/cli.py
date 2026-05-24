from __future__ import annotations

import shlex
from pathlib import Path

from emotion_aware_assistant.app import AssistantSession
from emotion_aware_assistant.core.config import load_config


HELP = """Commands:
  /open path/to/file.pdf|txt
  /page N
  /select start:end
  /context pasted text
  /emotion auto|neutral|happy|angry|sad|fear|surprise|disgust|contempt
  /state confusion|frustration|boredom|engagement|uncertain
  /model alias
  /ask question
  /status
  /quit
Plain text is treated as a question.
"""


def run_terminal(config_path: str = "config.yaml") -> int:
    config = load_config(config_path)
    session = AssistantSession(config=config)
    model_alias = config.get("llm", {}).get("default_model", "dummy")
    print("Emotion-Aware Academic Assistant terminal mode")
    print("Type /help for commands, /quit to exit.")
    print(f"Initial state: {session.learning_snapshot_text()}")

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if not line.startswith("/"):
            _ask(session, line, model_alias)
            continue
        command, _, rest = line.partition(" ")
        command = command.lower()
        try:
            if command in ("/quit", "/exit"):
                break
            if command == "/help":
                print(HELP)
            elif command == "/open":
                path = _strip_quotes(rest)
                document = session.load_document(path)
                print(f"Opened {document.title} ({document.page_count} page(s)).")
            elif command == "/page":
                text = session.set_page(int(rest.strip()))
                print(f"Page {session.current_page_number}: {len(text)} characters")
            elif command == "/select":
                start_s, end_s = rest.split(":", 1)
                context = session.select_range(int(start_s), int(end_s))
                print(f"Selected {len(context.selected_text)} characters; passage={context.passage_type}")
            elif command == "/context":
                context = session.set_manual_context(rest)
                print(f"Context set ({len(context.selected_text)} characters; passage={context.passage_type}).")
            elif command == "/emotion":
                value = rest.strip().lower()
                if value == "auto":
                    snapshot = session.set_override("auto")
                else:
                    snapshot = session.set_manual_emotion(value)
                print(_format_snapshot(snapshot))
            elif command == "/state":
                snapshot = session.set_manual_state(rest.strip().lower())
                print(_format_snapshot(snapshot))
            elif command == "/model":
                model_alias = rest.strip() or model_alias
                print(f"Model alias set to {model_alias}")
            elif command == "/ask":
                _ask(session, rest.strip(), model_alias)
            elif command == "/status":
                status = session.status(mode="terminal")
                print(status)
                print(session.learning_snapshot_text())
                print(f"Log file: {session.logger.path}")
            else:
                print(f"Unknown command: {command}. Type /help.")
        except Exception as exc:
            print(f"Error: {exc}")
    print("Terminal mode closed.")
    return 0


def _ask(session: AssistantSession, question: str, model_alias: str) -> None:
    if not question:
        print("Question is empty.")
        return
    response = session.ask(question, model_alias=model_alias)
    print(response.text)
    if response.error:
        print(f"[LLM error: {response.error}]")
    print(f"[model={response.model_name}, latency={response.latency_sec:.2f}s]")


def _format_snapshot(snapshot) -> str:
    return (
        f"raw={snapshot.raw_emotion}, smoothed={snapshot.smoothed_emotion}, "
        f"state={snapshot.state}, trend={snapshot.trend}, confidence={snapshot.confidence:.2f}, "
        f"strategy={snapshot.strategy}"
    )


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Path is required.")
    try:
        parts = shlex.split(value)
        return parts[0]
    except ValueError:
        return value.strip('"').strip("'")
