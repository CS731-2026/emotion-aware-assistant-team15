# Emotion-Aware Academic Assistant for Paper Reading Support

## Overview

This project is a local web-based assistant for reading academic papers. Users can upload PDFs, open them in a paper reading workspace, select passages or visual areas, and receive paper-grounded explanations supported by parsed PDF context and retrieval.

The assistant combines a reading workflow with a local webcam/model learning-signal pipeline. A typical interaction starts with a RAG baseline explanation, observes a short reaction window, recommends pedagogical strategies, and then generates a strategy-conditioned follow-up explanation. The project also includes transparency tools for camera/model inspection and prompt-based LLM comparison.

## Key Features

- Paper library and PDF upload.
- PDF parsing into Markdown, blocks, page context, and paper profile data.
- Keyword retrieval and Gemini embedding retrieval when configured.
- Text highlights and area selections.
- Baseline paper-grounded explanation.
- Conversation persistence per highlight.
- Local webcam learning signal and reaction-window monitoring.
- OpenFace-based face detection and landmark-derived face cropping when configured.
- Academic-state model inference for boredom, confusion, engagement, and frustration.
- Drop-in support for future 8-class raw-emotion checkpoints.
- Pedagogical strategy recommendation.
- Strategy-conditioned adaptive explanation.
- Prompt snapshots for baseline, strategy planner, and strategy-conditioned prompts.
- LLM comparison workflow using real saved prompts.
- Local settings page for provider credentials and model roles.
- Debug pages for camera/model and PDF/RAG inspection.

## Main Routes

- `/pdf-chat`: Final user-facing paper reading assistant.

- `/settings`: Local provider, API key, model role, and comparison profile configuration.

- `/camera-debug`: Camera, OpenFace, crop, model, raw/mapped emotion, and learning-signal transparency page.

- `/llm-compare`: Compare real prompt snapshots saved from `/pdf-chat` across multiple LLMs.

- `/pdf-test`: Standalone PDF/RAG/highlight debug workspace. It is intentionally independent from `/pdf-chat`.

## System Architecture

The main workflow is:

1. PDF ingestion: PDF upload -> text/layout extraction -> parsed blocks -> paper profile -> keyword index -> optional embedding index.
2. User interaction: text highlight or area selection -> retrieved context -> baseline explanation.
3. Learning signal: webcam frame -> local backend -> OpenFace face analysis/crop -> 224x224 model input -> emotion or academic-state model -> reaction window summary.
4. Strategy planning: selected evidence + paper context + baseline explanation + reaction window -> strategy candidates.
5. Adaptive response: selected strategy + paper context + prior explanation -> strategy-conditioned answer.
6. Comparison and debugging: prompt snapshots feed `/llm-compare`; camera/model internals are inspected in `/camera-debug`; PDF/RAG internals are inspected in `/pdf-test`.

The web app is served by `python -u main.py --mode web`. The server uses Python standard-library HTTP handling with route logic in `emotion_aware_assistant/web`.

## PDF Reading Pipeline

Uploaded documents are stored locally under:

```text
runtime_uploads/documents/<document_id>/
```

A prepared document may contain:

```text
meta.json
original.pdf
parsed/
rag/
highlights/
threads/
prompt_snapshots/
logs/
```

PDF preparation writes parsed files such as `parsed/document.md`, `parsed/content_list.json`, and `parsed/blocks_index.json`. The RAG preparation path builds `rag/paper_profile.json`, `rag/keyword_index.json`, and, when embedding configuration is available, `rag/embeddings.json`.

Highlights are saved under `highlights/`. Conversations are saved per highlight under `threads/<highlight_id>.json`. Prompt snapshots are saved separately under `prompt_snapshots/` so full prompts are not embedded in thread messages.

The retrieval path uses selected text, page number, matched parsed block, nearby context, related blocks, global RAG chunks, and paper profile information to ground explanations. If embeddings are unavailable, keyword retrieval remains available.

## Camera and Emotion Recognition Pipeline

Browser webcam frames are sent only to the local backend. They are not sent to external LLM providers. Raw frames are not persisted by default.

When OpenFace is configured, face analysis uses the `FeatureExtraction` binary to produce 68 landmarks. `/camera-debug` shows the exact analyzed frame, landmarks, landmark bbox, crop bbox, crop preview, and final 224x224 pre-normalization model input. `/pdf-chat` uses the same detector/crop/model pipeline internally but only shows a compact learning signal.

The current local checkpoint may be a 4-class academic-state model with these classes:

```text
boredom, confusion, engagement, frustration
```

The emotion pipeline also supports future 8-class raw-emotion checkpoints with these classes:

```text
anger, contempt, disgust, fear, happy, neutral, sad, surprise
```

For an 8-class checkpoint, raw probabilities are mapped to academic states:

```text
sad + anger + disgust -> frustration
fear + surprise -> confusion
contempt -> boredom
happy + neutral -> engagement
```

For a 4-class academic-state checkpoint, raw emotion is unavailable and mapping is bypassed. In both modes, `EmotionBuffer` smooths recent academic states by majority vote.

## Prompt and LLM Pipeline

The system has three main LLM prompt stages.

### Stage A: RAG Baseline Explanation Prompt

Purpose: explain the selected passage or selected visual area using PDF context.

Inputs include:

- selected text or area metadata
- page number
- matched parsed block
- nearby context
- retrieved RAG chunks
- paper profile
- optional user question
- answer style and grounding rules

Output: the first baseline explanation.

### Stage B: Strategy Planner Prompt

Purpose: generate pedagogical strategy candidates after the user reads the baseline explanation.

Inputs include:

- selected evidence
- paper context
- baseline explanation
- reaction window summary
- support cue
- academic-state distribution
- allowed strategy families
- recent conversation
- previous strategy metadata when available

Output: strict JSON strategy candidates. Each candidate is expected to include:

- `strategy_id`
- `strategy_family`
- `pedagogical_move`
- `context_focus`
- `title`
- `short_description`
- `why_recommended`
- `prompt_instruction`
- `expected_answer_shape`
- `recommended`
- `recommended_score`

Support cue values include:

- `sustained_clarification`
- `reduce_load`
- `re_engagement`
- `deepening`
- `clarify_and_reengage`
- `gentle_clarification`
- `neutral_or_uncertain`

The learning signal guides the likely support need, while the selected paper context determines the content focus. The strategy planner is instructed not to diagnose the user and not to mention webcam or face detection.

### Stage C: Strategy-Conditioned Answer Prompt

Purpose: generate a new explanation according to the selected pedagogical strategy.

Inputs include:

- selected passage or area
- RAG context
- baseline explanation
- selected strategy
- expected answer shape
- recent conversation
- grounding and safety rules

Output: a strategy-conditioned explanation.

Prompt snapshots for all three stages are saved under:

```text
runtime_uploads/documents/<document_id>/prompt_snapshots/
```

`/llm-compare` uses these real prompt snapshots for model comparison.

## LLM Configuration and Comparison

Use `/settings` to configure provider credentials and role-specific models. Supported provider types are:

- Gemini
- OpenRouter
- OpenAI-compatible endpoints

Configured roles are:

- `answer_model`: baseline explanations, strategy-conditioned answers, and follow-ups.
- `strategy_planner_model`: strategy candidate generation.
- `embedding_model`: document embedding and RAG retrieval when supported.

API keys are stored locally in `.env.local` and are not stored in browser localStorage or sessionStorage. Non-secret comparison model profiles are stored in:

```text
runtime_uploads/config/llm_profiles.json
```

`/llm-compare` can compare saved prompt snapshots for:

- RAG baseline prompts
- strategy planner prompts
- strategy-conditioned answer prompts

OpenRouter model IDs are editable, so paid or custom model IDs can be tested if the local OpenRouter key has access.

## Setup

1. Clone or enter the project directory.

2. Create and activate a Python environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install frontend dependencies:

   ```bash
   npm install
   ```

5. Configure LLM/API access after starting the server by opening `/settings`, or configure Gemini from the terminal:

   ```bash
   python scripts/configure_api_key.py
   ```

6. Configure an emotion model checkpoint if needed:

   ```bash
   python scripts/configure_emotion_checkpoint.py --checkpoint /path/to/best.pt --mode auto
   ```

   Model weights are local files and should not be committed.

7. Configure OpenFace if using the OpenFace detector:

   ```bash
   python scripts/diagnose_openface.py
   python scripts/configure_openface.py --bin /path/to/FeatureExtraction
   ```

8. Start the web app:

   ```bash
   python -u main.py --mode web
   ```

## Running the App

Start the local server:

```bash
python -u main.py --mode web
```

By default the server tries `127.0.0.1:8000` and then the next available ports if needed. Open:

```text
http://127.0.0.1:8000/pdf-chat
```

The legacy terminal mode is still available:

```bash
python main.py --mode terminal
```

## Testing and Diagnostics

Run the main checks:

```bash
python -m unittest
npm run build:pdf-workspace
git diff --check
```

Useful diagnostics:

```bash
python scripts/diagnose_environment.py
python scripts/inspect_emotion_checkpoint.py --checkpoint /path/to/best.pt
python scripts/diagnose_openface.py
python scripts/test_openface_feature_extraction.py --image /path/to/image.jpg
```

## Changing the Emotion Model

Place checkpoints locally and do not commit them. The default candidate paths are:

```text
models/emotion_model/raw_8class_best.pt
models/emotion_model/best_model.pt
```

Configuration supports:

- `RAW_EMOTION_CHECKPOINT_PATH`
- `EMOTION_CHECKPOINT_PATH`
- `EMOTION_MODEL_MODE` with `auto`, `raw_emotion`, or `academic_state`

Use:

```bash
python scripts/configure_emotion_checkpoint.py --checkpoint /path/to/best.pt --mode auto
```

For a 4-class academic-state model, classes should be:

```text
boredom, confusion, engagement, frustration
```

For an 8-class raw-emotion model, classes should be:

```text
anger, contempt, disgust, fear, happy, neutral, sad, surprise
```

The system inspects checkpoint metadata and detects the mode. `/camera-debug` shows whether raw detection is available, and `/pdf-chat` consumes the mapped or direct academic state.

## Changing LLM Providers or Models

Use `/settings` to configure:

- Gemini credentials and models
- OpenRouter credentials and default model
- OpenAI-compatible credentials, base URL, and default model
- answer model role
- strategy planner model role
- embedding model role
- comparison model profiles

The embedding path currently supports Gemini embeddings. If another embedding provider is selected, the app reports a warning and falls back to keyword retrieval where possible.

Use `/llm-compare` to test saved baseline, strategy planner, and strategy-conditioned prompts across configured models.

## Privacy and Local Data

- Webcam frames are processed locally and are not sent to LLM providers.
- Raw webcam frames are not persisted by default.
- API keys are stored locally in `.env.local`.
- Uploaded PDFs, parsed files, highlights, conversations, prompt snapshots, and comparison results live under `runtime_uploads/`.
- Model weights and OpenFace binaries are local operational files and should not be shared publicly.

Do not publicly share:

- `.env.local`
- `runtime_uploads/`
- model weights
- OpenFace binaries
- API keys
- raw frames or private uploaded papers

## Repository Notes

The source code, scripts, tests, and documentation are intended to be tracked. Local runtime data and sensitive operational artifacts should remain untracked.

Do not commit:

- `.env.local`
- model weights such as `.pt`, `.pth`, `.ckpt`, `.onnx`, or `.engine`
- OpenFace binaries or external build outputs
- runtime uploads and prompt snapshots
- API keys or provider secrets
- raw webcam frames
