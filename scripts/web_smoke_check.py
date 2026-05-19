from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_aware_assistant.web.server import create_web_app
from scripts.create_sample_data import create_sample_data


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    create_sample_data(ROOT)
    app = create_web_app(force_dummy_llm=True)

    status = app.test_request("GET", "/api/status")
    require(status["status"] == 200, f"status failed: {status}")
    require("models" in status["json"], "status did not include model registry")

    loaded = app.test_request("POST", "/api/document/load-sample")
    require(loaded["status"] == 200, f"load sample failed: {loaded}")
    page_text = loaded["json"]["current_page_text"]
    require("Method" in page_text, "sample paper text did not load")

    page = app.test_request("GET", "/api/document/page/1")
    require(page["status"] == 200 and "Method" in page["json"]["text"], "page endpoint did not return page text")

    upload = app.test_request(
        "POST",
        "/api/document/upload",
        files={"file": ("uploaded_demo.txt", b"Title: Uploaded Demo\n\nMethod\nThis uploaded method has steps and outputs.")},
    )
    require(upload["status"] == 200, f"TXT upload failed: {upload}")
    require("Uploaded Demo" in upload["json"]["current_page_text"], "uploaded TXT content was not returned")

    app.test_request("POST", "/api/document/load-sample")

    selected = "The proposed method receives a selected passage, surrounding paper context, and a compact learning-state signal."
    context = app.test_request(
        "POST",
        "/api/document/context",
        {
            "selected_text": selected,
            "page_number": 1,
            "user_question": "Can you explain this method?",
        },
    )
    require(context["status"] == 200, f"context failed: {context}")
    require(context["json"]["passage_type"] == "method/process/mechanism", "passage type was not method/process/mechanism")
    require(len(context["json"]["retrieved_chunks"]) > 0, "retrieval returned no chunks")
    require(context["json"]["selected_text"] == selected, "selected_text was not preserved in context")
    require(context["json"]["difficulty_hint"], "difficulty_hint missing")
    require(context["json"]["passage_analysis"]["detected_keywords"], "passage analysis did not expose keywords")
    require(context["json"]["retrieval_debug"]["ranked_chunks"], "retrieval debug did not include ranked chunks")

    document_id = context["json"]["document_id"]
    require(document_id, "context did not include document_id")
    highlight = app.test_request(
        "POST",
        "/api/document/highlight",
        {
            "document_id": document_id,
            "selected_text": selected,
            "page_number": 1,
            "rects": [{"left": 24, "top": 48, "width": 260, "height": 18}],
            "color": "yellow",
        },
    )
    require(highlight["status"] == 200, f"highlight failed: {highlight}")
    highlight_id = highlight["json"]["highlight_id"]
    require(highlight["json"]["context"]["selected_text"] == selected, "highlight context did not preserve selected text")
    stored_highlights = app.test_request("GET", f"/api/document/highlights/{document_id}")
    require(stored_highlights["status"] == 200, f"stored highlights failed: {stored_highlights}")
    require(stored_highlights["json"]["highlights"][0]["highlight_id"] == highlight_id, "stored highlight id mismatch")

    answers: dict[str, str] = {}
    for state in ["confusion", "frustration", "boredom", "engagement"]:
        manual = app.test_request("POST", "/api/emotion/manual", {"emotion": state})
        require(manual["status"] == 200, f"manual emotion failed for {state}: {manual}")
        require(manual["json"]["state"] == state, f"manual state mismatch for {state}")
        chat = app.test_request(
            "POST",
            "/api/chat",
            {
                "selected_text": selected,
                "document_id": document_id,
                "highlight_id": highlight_id,
                "page_number": 1,
                "user_question": "Can you explain this method?",
                "model_alias": "dummy",
            },
        )
        require(chat["status"] == 200, f"chat failed for {state}: {chat}")
        require(chat["json"]["learning_state"]["state"] == state, f"chat state mismatch for {state}")
        require(selected in chat["json"]["prompt_preview"], "prompt preview did not contain selected text")
        require(chat["json"]["highlight_id"] == highlight_id, "chat did not preserve highlight id")
        require(chat["json"]["passage_type"] == "method/process/mechanism", "chat ignored passage type")
        require(chat["json"]["response_policy"]["strategy"] == chat["json"]["strategy"], "policy strategy mismatch")
        answers[state] = chat["json"]["answer"]

    require(len(set(answers.values())) == 4, "state-specific dummy answers were not visibly different")
    require("Core idea" in answers["confusion"], "confusion answer did not use step-oriented structure")
    require("Simplest version first" in answers["frustration"], "frustration answer did not simplify")
    require("One-sentence takeaway" in answers["boredom"], "boredom answer did not re-engage concisely")
    require("Technical explanation" in answers["engagement"], "engagement answer did not go deeper")

    frame = app.test_request("POST", "/api/emotion/frame", {"image": "data:image/jpeg;base64,AAAA"})
    require(frame["status"] == 200, f"frame endpoint failed: {frame}")
    require("state" in frame["json"], "frame endpoint did not return state")

    state_payload = app.test_request("GET", "/api/emotion/state")
    require(state_payload["status"] == 200, f"emotion state failed: {state_payload}")
    require(state_payload["json"]["history"], "emotion state history is empty")
    require(state_payload["json"]["probabilities"], "emotion state probabilities missing")

    speech = app.test_request("POST", "/api/speech/transcribe")
    require(speech["status"] == 200, f"speech endpoint failed: {speech}")

    log_path = app.state.session.logger.path
    require(log_path.exists() and log_path.stat().st_size > 0, "JSONL log was not written")
    last_log = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    for key in [
        "document_id",
        "document_type",
        "page_number",
        "highlight_id",
        "selected_text_preview",
        "selected_passage_length",
        "learning_state",
        "trend",
        "strategy",
        "model_name",
        "response_length",
        "passage_type",
    ]:
        require(key in last_log, f"log missing {key}")
    require(last_log["highlight_id"] == highlight_id, "log did not include latest highlight id")

    print("WEB SMOKE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
