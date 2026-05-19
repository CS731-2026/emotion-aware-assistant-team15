# Demo Script

## Demo Goal

Show that the system supports:

- paper upload and reading
- text highlights and area selections
- RAG-grounded baseline explanations
- live learning signal and reaction-window monitoring
- pedagogical strategy recommendation
- strategy-conditioned explanations
- camera/model transparency
- prompt snapshot comparison across LLMs

## Pre-Demo Setup

Before the demo:

- Start the server with `python -u main.py --mode web`.
- Open `/settings`.
- Confirm provider/API status is configured for the models you plan to use.
- Confirm the `answer_model`, `strategy_planner_model`, and `embedding_model` roles.
- Confirm the emotion checkpoint is configured if model inference is part of the demo.
- Confirm OpenFace is configured if using live camera analysis.
- Open `/camera-debug` and capture one frame to verify landmarks, crop preview, and model input.
- Open `/pdf-chat`.
- Confirm a sample paper is available in the paper library, or prepare a PDF to upload.
- Optionally use a fresh highlight if you want a clean conversation thread.

## Demo Part 1: Settings

1. Open `/settings`.
2. Show provider credential status for Gemini, OpenRouter, and OpenAI-compatible endpoints.
3. Show role-based model settings:
   - answer model
   - strategy planner model
   - embedding model
4. Show comparison model profiles.
5. Explain that API keys are stored locally in `.env.local` and are hidden from the browser UI.

## Demo Part 2: Camera Debug

1. Open `/camera-debug`.
2. Start the camera.
3. Capture one frame.
4. Show the last analyzed frame.
5. Show OpenFace landmarks and bboxes.
6. Show the face crop preview.
7. Show the 224x224 model input preview.
8. Show the active model mode:
   - academic-state mode, or
   - raw-emotion mode if an 8-class checkpoint is configured.
9. Show the raw/mapped panel.
10. Explain that `/pdf-chat` uses only the compact learning signal, while `/camera-debug` is for transparency and troubleshooting.

## Demo Part 3: PDF Chat

1. Open `/pdf-chat`.
2. Open an existing paper or upload a new PDF.
3. Highlight a passage or select an area.
4. Click **Explain**.
5. Show the baseline explanation.
6. Explain that this is Stage A: the RAG baseline explanation prompt.
7. Wait for the reaction window to complete.
8. Show the strategy candidates under the baseline turn.
9. Explain that this is Stage B: the strategy planner prompt.
10. Select a strategy and click **Explain with this strategy**.
11. Show the strategy-conditioned answer.
12. Explain that this is Stage C: the strategy-conditioned answer prompt.
13. If time allows, refresh the page or click the highlight again to show that the conversation is persisted.

## Demo Part 4: LLM Compare

1. Open `/llm-compare`.
2. Select a prompt snapshot.
3. Use the stage filter to show:
   - RAG baseline
   - Strategy planner
   - Strategy-conditioned answer
4. Preview the saved prompt messages and context summary.
5. Choose or edit comparison models.
6. Run the comparison.
7. Show outputs side by side.
8. For strategy planner snapshots, show JSON validity and candidate checks.
9. Show manual scoring fields.
10. Export JSON or Markdown if needed for reporting.

## Demo Part 5: PDF Debug

1. Open `/pdf-test`.
2. Explain that it is a standalone PDF/RAG/highlight debug workspace.
3. Use it only for inspecting PDF parsing, matching, retrieval, and selection behavior.
4. Do not present it as the main reading assistant.

## Common Failure Recovery

- If embedding preparation fails, open `/settings` and check the embedding provider, model, and API key. Keyword retrieval can still be used.
- If Gemini calls fail, check `/settings`, the Gemini key, selected model, and quota.
- If OpenRouter calls fail, check the OpenRouter API key and exact model ID.
- If an OpenAI-compatible endpoint fails, check the base URL, API key, and model ID.
- If the camera does not start, check browser camera permission and retry in `/camera-debug`.
- If OpenFace is not available, run `python scripts/diagnose_openface.py`.
- If no prompt snapshots appear in `/llm-compare`, run a new explanation in `/pdf-chat` first.
- If a conversation does not appear after reopening, check `runtime_uploads/documents/<document_id>/threads/`.
- If the emotion model is not available, inspect it with `python scripts/inspect_emotion_checkpoint.py --checkpoint /path/to/best.pt`.

## Suggested Talk Track

### 30 seconds: Project Overview

This is an academic paper reading assistant. The user reads a PDF, highlights a passage or figure, and receives an explanation grounded in the paper. The system also monitors a local learning signal and uses it to choose a suitable pedagogical strategy for a follow-up explanation.

### 1 minute: Camera and Model Transparency

Open `/camera-debug`. Show that the system analyzes one captured frame at a time, displays the exact analyzed frame, overlays OpenFace landmarks, and shows the crop and 224x224 model input used for inference. Emphasize that this is a local transparency page, not the main user interface.

### 2 minutes: PDF Reading and Adaptive Strategy

Open `/pdf-chat`. Select a passage and click **Explain**. Describe the first answer as the RAG baseline explanation. After the reaction window, show the strategy candidates and explain that the planner combines the selected passage, paper context, baseline answer, and learning signal. Select one strategy and generate the adaptive explanation.

### 1 minute: LLM Comparison

Open `/llm-compare`. Select a saved snapshot from one of the three prompt stages, run it across configured models, and compare outputs. Show that the same real prompt can be evaluated across providers and exported for later analysis.

### Closing Summary

The project combines PDF-grounded explanation, local learning-signal estimation, adaptive pedagogy, and prompt-level model comparison. `/pdf-chat` is the final reading assistant, while `/settings`, `/camera-debug`, `/llm-compare`, and `/pdf-test` support configuration, transparency, and debugging.
