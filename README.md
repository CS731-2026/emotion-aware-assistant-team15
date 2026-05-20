# Emotion-Aware Academic Assistant for Paper Reading Support

> **COMPSYS 731 — Human-Robot Interaction Group Project**  
> **Team 15 — Emotion-Aware Academic Paper Reading Assistant**  
> University of Auckland · Semester 1, 2026

[![Course](https://img.shields.io/badge/COMPSYS%20731-Human--Robot%20Interaction-blue)](https://www.auckland.ac.nz/)
[![Topic](https://img.shields.io/badge/Topic-Emotion--Aware%20Paper%20Reading%20Assistant-purple)](#)
[![Python](https://img.shields.io/badge/Python-3.9%2F3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/Version%20Control-GitHub-black)](https://github.com/)

<p align="center">
  <img src="https://images.unsplash.com/photo-1522202176988-66273c2fd55f?auto=format&fit=crop&w=1600&q=60" width="820" alt="Banner" />
</p>

This repository contains the final runtime system for an **Emotion-Aware Academic Assistant**. The application helps users read academic papers by combining PDF-grounded explanation, local camera-based learning-signal estimation, pedagogical strategy planning, and LLM response generation.

The final user-facing workflow is implemented as a local web app. A user uploads or opens a PDF, highlights a passage or selects a visual area, receives an initial paper-grounded explanation, and then receives a recommended adaptive follow-up strategy based on a compact learning signal derived from the local emotion/academic-state pipeline.

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Final System Summary](#2-final-system-summary)
3. [Main User Routes](#3-main-user-routes)
4. [Repository Structure](#4-repository-structure)
5. [Core Architecture](#5-core-architecture)
6. [PDF Reading and RAG Pipeline](#6-pdf-reading-and-rag-pipeline)
7. [Camera and Emotion / Academic-State Pipeline](#7-camera-and-emotion--academic-state-pipeline)
8. [Emotion-to-Academic-State Mapping](#8-emotion-to-academic-state-mapping)
9. [Final Emotion Model and Training Results](#9-final-emotion-model-and-training-results)
10. [LLM and Prompt Pipeline](#10-llm-and-prompt-pipeline)
11. [Installation](#11-installation)
12. [Configuration](#12-configuration)
13. [Running the Application](#13-running-the-application)
14. [Demo Workflow](#14-demo-workflow)
15. [Testing and Diagnostics](#15-testing-and-diagnostics)
16. [Local Runtime Data and Privacy](#16-local-runtime-data-and-privacy)
17. [Known Limitations](#17-known-limitations)
18. [Useful Handoff Notes](#18-useful-handoff-notes)
19. [References](#19-references)

---

## 1. Project Goal

Academic paper reading is cognitively demanding. Users may become confused by complex methods, frustrated by dense writing, bored by long explanations, or engaged and ready for deeper discussion. A standard chatbot cannot observe these learning states unless the user explicitly reports them.

This project addresses that gap by creating a local assistant that:

- reads and parses academic PDFs;
- retrieves relevant paper context for selected text or selected visual areas;
- detects a compact local learning signal from the camera/model pipeline;
- maps the signal into learning-centered academic states;
- plans a suitable pedagogical support strategy;
- generates adaptive explanations using an LLM.

The key design idea is:

```text
Paper context + user question + learning signal -> adaptive academic support
```

The assistant does **not** diagnose the user. It uses only a lightweight learning-support signal to choose a response style such as clarification, simplification, re-engagement, or deeper expansion.

---

## 2. Final System Summary

The final codebase provides:

- **Web-based PDF reading assistant** at `/pdf-chat`.
- **PDF upload and parsing** for text, blocks, layout, page-level context, and paper profile information.
- **Highlight and area selection workflow** for text passages and visual regions.
- **RAG-grounded baseline explanation** using selected evidence, nearby context, retrieved chunks, and paper profile data.
- **Local camera/model transparency page** at `/camera-debug`.
- **4-class academic-state model support** for `boredom`, `confusion`, `engagement`, and `frustration`.
- **Schema-compatible 8-class raw-emotion support** for future checkpoints.
- **Academic-state smoothing** through a short rolling buffer.
- **Reaction-window summary** to help select pedagogical strategies.
- **Strategy planner prompt stage** that proposes learning-support strategies.
- **Strategy-conditioned answer generation** for adaptive explanations.
- **LLM provider settings page** at `/settings`.
- **Prompt snapshot comparison workflow** at `/llm-compare`.
- **PDF/RAG debug workspace** at `/pdf-test`.
- **Local JSONL logging** for interaction analysis.

Important implementation note:

> The final runtime repository focuses on application integration and model inference. Model training was performed externally. The trained model weights are not committed to Git and must be installed locally under `models/emotion_model/`.

---

## 3. Main User Routes

| Route           | Purpose                                                                                                                                  | Intended audience             |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `/pdf-chat`     | Final paper reading assistant. Upload/open a paper, highlight/select content, ask questions, receive baseline and adaptive explanations. | Main demo / end user          |
| `/settings`     | Configure LLM providers, API keys, role-specific models, comparison models, face detector, OpenFace, and emotion checkpoint.             | Developer / local operator    |
| `/camera-debug` | Inspect webcam frame analysis, OpenFace/face detection output, crop preview, model input, raw/mapped state, and reaction summary.        | Developer / transparency demo |
| `/llm-compare`  | Compare saved prompt snapshots across multiple LLMs and export comparison results.                                                       | Evaluation / analysis         |
| `/pdf-test`     | Standalone PDF, RAG, parsing, matching, and highlight debug workspace.                                                                   | Developer debugging           |

The final presentation should primarily show `/pdf-chat`, with `/camera-debug`, `/settings`, and `/llm-compare` as supporting transparency and evaluation pages.

---

## 4. Repository Structure

The final repository is organized around one Python package and several runtime/configuration folders.

```text
emotion-aware-assistant-team15-master/
├── README.md
├── main.py
├── pyproject.toml
├── requirements.txt
├── package.json
├── package-lock.json
├── docs/
│   ├── demo_script.md
│   └── archive/
│       └── README_github_original.md
├── emotion_aware_assistant/
│   ├── app.py
│   ├── cli.py
│   ├── core/
│   │   ├── config.py
│   │   ├── errors.py
│   │   ├── llm_config.py
│   │   ├── logging_utils.py
│   │   └── types.py
│   ├── emotion/
│   │   ├── affective_trend_tracker.py
│   │   ├── camera_worker.py
│   │   ├── dummy_emotion.py
│   │   ├── emotion_buffer.py
│   │   ├── emotion_interface.py
│   │   ├── face_detector.py
│   │   ├── labels.py
│   │   ├── manual_emotion.py
│   │   ├── raw_emotion_pipeline.py
│   │   ├── state_mapper.py
│   │   └── teammate_emotion_adapter.py
│   ├── evaluation/
│   │   ├── evaluation_schema.py
│   │   └── interaction_logger.py
│   ├── llm/
│   │   ├── dummy_llm.py
│   │   ├── llm_interface.py
│   │   ├── model_registry.py
│   │   ├── openrouter_client.py
│   │   ├── prompt_builder.py
│   │   ├── providers.py
│   │   └── response_policy.py
│   ├── paper/
│   │   ├── document.py
│   │   ├── paper_rag.py
│   │   ├── passage_analyzer.py
│   │   ├── pdf_loader.py
│   │   ├── pdf_parse_pipeline.py
│   │   ├── retriever.py
│   │   └── text_chunker.py
│   ├── speech/
│   │   ├── dummy_speech.py
│   │   ├── faster_whisper_adapter.py
│   │   └── speech_interface.py
│   ├── ui/
│   │   ├── gui_app.py
│   │   ├── main_window.py
│   │   ├── styles.qss
│   │   └── workers.py
│   └── web/
│       ├── routes.py
│       ├── schemas.py
│       ├── server.py
│       ├── state.py
│       └── static/
│           ├── index.html
│           ├── pdf_chat.html
│           ├── pdf_test.html
│           ├── camera_debug.html
│           ├── local_settings.html
│           ├── llm_compare.html
│           └── pdf-workspace/
├── models/
│   ├── emotion_model/
│   │   └── README.md
│   └── face_detector/
│       └── README.md
├── sample_data/
│   ├── README.md
│   └── sample_paper.txt
├── scripts/
│   ├── configure_api_key.py
│   ├── configure_emotion_checkpoint.py
│   ├── configure_openface.py
│   ├── create_sample_data.py
│   ├── diagnose_environment.py
│   ├── diagnose_openface.py
│   ├── find_face_detector_weights.py
│   ├── inspect_emotion_checkpoint.py
│   ├── install_emotion_checkpoint.py
│   ├── smoke_check.py
│   ├── test_emotion_adapter.py
│   ├── test_openface_feature_extraction.py
│   └── web_smoke_check.py
└── tests/
    ├── test_camera_debug.py
    ├── test_core_flow.py
    ├── test_emotion_checkpoint_scripts.py
    ├── test_llm_compare.py
    ├── test_pdf_chat_backend.py
    ├── test_pdf_parse_pipeline.py
    ├── test_pdf_rag.py
    ├── test_teammate_emotion_adapter.py
    └── test_web_api.py
```

Runtime folders such as `runtime_uploads/`, `logs/`, and local model weights are intentionally ignored by Git.

---

## 5. Core Architecture

The implemented product uses the following end-to-end flow:

```text
PDF Upload / Sample Paper
        |
        v
PDF Parsing and RAG Preparation
        |
        v
User Highlight or Area Selection
        |
        v
Stage A: RAG Baseline Explanation
        |
        v
Local Camera / Emotion Model Signal
        |
        v
Reaction Window Summary
        |
        v
Stage B: Pedagogical Strategy Planner
        |
        v
User Chooses Recommended Strategy
        |
        v
Stage C: Strategy-Conditioned Adaptive Explanation
        |
        v
Saved Prompt Snapshots + Conversation Thread + Evaluation Logs
```

The backend is a local Python web server built with `http.server.ThreadingHTTPServer`; it does not require FastAPI at runtime. Route dispatch is implemented in:

```text
emotion_aware_assistant/web/server.py
emotion_aware_assistant/web/routes.py
emotion_aware_assistant/web/state.py
```

The user-facing pages are static HTML/JS assets under:

```text
emotion_aware_assistant/web/static/
```

---

## 6. PDF Reading and RAG Pipeline

### 6.1 Document storage

Uploaded and generated document artifacts are saved locally under:

```text
runtime_uploads/documents/<document_id>/
```

A prepared document can contain:

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

### 6.2 Parsing outputs

The PDF parsing pipeline is implemented mainly in:

```text
emotion_aware_assistant/paper/pdf_loader.py
emotion_aware_assistant/paper/pdf_parse_pipeline.py
emotion_aware_assistant/paper/paper_rag.py
```

Typical parsed artifacts include:

```text
parsed/document.md
parsed/content_list.json
parsed/blocks_index.json
rag/paper_profile.json
rag/keyword_index.json
rag/embeddings.json          # only when embedding configuration is available
```

### 6.3 Retrieval behavior

The explanation pipeline can use:

- selected text;
- page number;
- matched parsed block;
- nearby useful context;
- candidate captions for area selections;
- paper profile information;
- keyword retrieval results;
- optional Gemini embedding retrieval results.

If embeddings are unavailable, the system still falls back to keyword retrieval.

### 6.4 Highlights and conversations

Highlights are saved under:

```text
runtime_uploads/documents/<document_id>/highlights/
```

Conversation threads are saved per highlight under:

```text
runtime_uploads/documents/<document_id>/threads/<highlight_id>.json
```

Prompt snapshots are stored separately from conversation messages under:

```text
runtime_uploads/documents/<document_id>/prompt_snapshots/
```

This makes `/llm-compare` possible because it can reuse the exact prompts generated during real interactions.

---

## 7. Camera and Emotion / Academic-State Pipeline

The camera pipeline is local. Browser frames are sent to the local backend only; they are not sent to external LLM providers.

### 7.1 Face analysis

The implementation supports multiple face-analysis paths:

| Path                       | File / config                                      | Notes                                                                                             |
| -------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| OpenFace FeatureExtraction | `emotion_aware_assistant/emotion/face_detector.py` | Preferred transparency path when OpenFace is configured. Provides landmarks, bbox, pose, and AUs. |
| YOLO face detector         | `models/face_detector/yolov8n-face.pt`             | Optional local weights. Not committed.                                                            |
| OpenCV Haar fallback       | OpenCV built-in fallback                           | Used when optional model weights are absent and OpenCV is available.                              |

`/camera-debug` displays the exact analyzed frame, landmarks, bounding box, face crop, and 224x224 model input preview.

### 7.2 Current final checkpoint mode

The current final checkpoint is a **4-class academic-state model**, not an 8-class raw facial emotion model.

Current output classes:

```text
boredom
confusion
engagement
frustration
```

In this mode:

- raw 8-class emotion is unavailable;
- the model directly predicts academic states;
- mapping is bypassed;
- `/camera-debug` shows `raw_emotion_available = false`;
- the output is still smoothed and used as the learning signal.

### 7.3 Future raw-emotion checkpoint mode

The code is schema-compatible with future 8-class raw-emotion checkpoints using:

```text
anger / angry
contempt
disgust
fear
happy
neutral
sad
surprise
```

In raw-emotion mode, the system can show both:

```text
Raw Detection -> Mapped Academic State -> Smoothed State -> Response Strategy
```

### 7.4 Smoothing and trend tracking

The implementation includes:

- `EmotionBuffer` for majority-vote smoothing;
- `AffectiveTrendTracker` for short-window trend and hysteresis;
- confidence thresholds and high-confidence switching rules in default config.

Default configuration from `emotion_aware_assistant/core/config.py`:

```text
buffer_size = 10
confidence_threshold = 0.35
trend_window_sec = 6
trend_update_interval_sec = 0.5
hysteresis_updates = 3
high_confidence_switch_threshold = 0.80
```

---

## 8. Emotion-to-Academic-State Mapping

The final deployed checkpoint directly predicts four academic states. However, the repository still contains the raw-emotion mapping layer for compatibility with an 8-class checkpoint.

The implemented raw-emotion probability aggregation in `raw_emotion_pipeline.py` is:

| Raw emotion(s)              | Mapped academic state |
| --------------------------- | --------------------- |
| `sad` + `anger` + `disgust` | `frustration`         |
| `fear` + `surprise`         | `confusion`           |
| `contempt`                  | `boredom`             |
| `happy` + `neutral`         | `engagement`          |

Equivalent rule format:

```python
frustration = P(sad) + P(anger) + P(disgust)
confusion   = P(fear) + P(surprise)
boredom     = P(contempt)
engagement  = P(happy) + P(neutral)
```

The single-label mapping is:

```text
sad      -> frustration
anger    -> frustration
disgust  -> frustration
fear     -> confusion
surprise -> confusion
contempt -> boredom
happy    -> engagement
neutral  -> engagement
```

This mapping is used only when a raw 8-class checkpoint is installed. With the current 4-class checkpoint, the mapping rule is:

```text
bypassed: checkpoint directly predicts academic states
```

### Response strategies

The mapped or directly predicted academic state is converted into a response strategy:

| Academic state | Strategy                   |
| -------------- | -------------------------- |
| `confusion`    | Step-by-step clarification |
| `frustration`  | Supportive simplification  |
| `boredom`      | Concise re-engagement      |
| `engagement`   | Deeper academic expansion  |
| `uncertain`    | Neutral adaptive support   |

---

## 9. Final Emotion Model and Training Results

### 9.1 Final selected model

The final selected academic-state model is:

```text
Model: ConvNeXt-Tiny
Architecture: convnext_tiny.fb_in22k_ft_in1k
Framework: PyTorch + timm
Output classes: boredom, confusion, engagement, frustration
Input size: 224 x 224
Training epochs: 25
Batch size: 64
Learning rate: 5e-5
Best epoch: 19
Best validation accuracy: 80.67%
Test accuracy: 79.94%
```

Recommended runtime checkpoint installation target:

```text
models/emotion_model/best_model.pt
models/emotion_model/metadata.json
```

Use the installer:

```bash
python scripts/install_emotion_checkpoint.py --source /path/to/convnext_best_checkpoint_or_folder
```

or configure an existing checkpoint directly:

```bash
python scripts/configure_emotion_checkpoint.py --checkpoint /path/to/best.pt --mode auto
```

### 9.2 Six-model comparison

The final six model comparison used 25 training epochs and pretrained `timm` backbones. ResNet50 was retained as a strong CNN baseline, while the other models were treated as final candidate architectures.

| Rank | Model             | `timm` architecture                             | Epochs | Batch |     LR | Best epoch | Highest val acc | Final epoch val acc |   Test acc | Role                    |
| ---: | ----------------- | ----------------------------------------------- | -----: | ----: | -----: | ---------: | --------------: | ------------------: | ---------: | ----------------------- |
|    1 | **ConvNeXt-Tiny** | `convnext_tiny.fb_in22k_ft_in1k`                |     25 |    64 | `5e-5` |         19 |      **80.67%** |              80.19% | **79.94%** | Final selected model    |
|    2 | RegNetY-800MF     | `regnety_008_tv.tv2_in1k`                       |     25 |    64 | `1e-4` |         24 |          79.00% |              78.04% |     78.59% | Efficient CNN candidate |
|    3 | Swin-Tiny         | `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` |     25 |    64 | `3e-5` |         21 |          78.72% |              78.68% |     78.43% | Transformer candidate   |
|    4 | MobileNetV3-Large | `mobilenetv3_large_100.miil_in21k_ft_in1k`      |     25 |    64 | `3e-4` |         22 |          78.68% |              77.64% |     76.13% | Lightweight candidate   |
|    5 | ResNet50          | `resnet50.a1_in1k`                              |     25 |    64 | `1e-4` |         21 |          78.32% |              77.52% |     76.49% | Baseline                |
|    6 | EfficientNet-B4   | `tf_efficientnet_b4.ns_jft_in1k`                |     25 |    32 | `5e-5` |         22 |          75.33% |              74.77% |     74.62% | Not selected            |

### 9.3 Best checkpoint paths used during training

Training artifacts were kept outside Git. The best checkpoints were:

```text
emotion_recognition/checkpoints/convnext_tiny_4state_25ep_b64_lr5e5/best.pt
emotion_recognition/checkpoints/regnety_008_4state_25ep_b64_lr1e4/best.pt
emotion_recognition/checkpoints/swin_tiny_4state_25ep_b64_lr3e5/best.pt
emotion_recognition/checkpoints/mobilenetv3_large_4state_25ep_b64_lr3e4/best.pt
emotion_recognition/checkpoints/resnet50_a1_4state_25ep_b64_lr1e4/best.pt
emotion_recognition/checkpoints/efficientnet_b4_4state_25ep_b32_lr5e5/best.pt
```

For the final application, copy/install only the selected ConvNeXt-Tiny best checkpoint into `models/emotion_model/best_model.pt` and keep all weights untracked.

### 9.4 Why ConvNeXt-Tiny was selected

ConvNeXt-Tiny was selected because it achieved:

- the highest validation accuracy: **80.67%**;
- the highest test accuracy: **79.94%**;
- stable performance across later epochs;
- better performance than ResNet50 baseline and all other candidates.

---

## 10. LLM and Prompt Pipeline

The system uses three major prompt stages.

### Stage A — RAG Baseline Explanation

Purpose: generate the first paper-grounded explanation.

Main inputs:

- selected text or area metadata;
- page number;
- matched parsed block;
- nearby context;
- retrieved chunks;
- paper profile;
- optional user question;
- grounding rules.

Output:

```text
Baseline explanation grounded in the selected paper content.
```

### Stage B — Strategy Planner

Purpose: recommend pedagogical strategy candidates after the baseline explanation.

Main inputs:

- selected evidence;
- paper context;
- baseline answer;
- recent learning-signal window;
- academic-state distribution;
- support cue;
- recent conversation;
- allowed strategy families.

Expected output is strict JSON with fields such as:

```text
strategy_id
strategy_family
pedagogical_move
context_focus
title
short_description
why_recommended
prompt_instruction
expected_answer_shape
recommended
recommended_score
```

Support cue values include:

```text
sustained_clarification
reduce_load
re_engagement
deepening
clarify_and_reengage
gentle_clarification
neutral_or_uncertain
```

### Stage C — Strategy-Conditioned Answer

Purpose: generate an adaptive explanation using the selected strategy.

Main inputs:

- selected passage or visual area;
- RAG context;
- baseline explanation;
- selected strategy;
- expected answer shape;
- recent conversation;
- grounding and safety rules.

Output:

```text
Adaptive academic explanation conditioned on the selected pedagogical strategy.
```

### LLM providers and roles

Use `/settings` to configure local provider credentials and role-specific models.

Supported provider types:

- Gemini;
- OpenRouter;
- OpenAI-compatible endpoints.

Main roles:

| Role                     | Purpose                                                                        |
| ------------------------ | ------------------------------------------------------------------------------ |
| `answer_model`           | Baseline explanation, strategy-conditioned explanation, and follow-up answers. |
| `strategy_planner_model` | JSON strategy candidate generation.                                            |
| `embedding_model`        | RAG embedding generation when supported.                                       |

Secrets are stored in `.env.local`; non-secret comparison model profiles are stored in:

```text
runtime_uploads/config/llm_profiles.json
```

---

## 11. Installation

### 11.1 Recommended Python version

The package requires Python >= 3.10. Python 3.11 was used successfully during final integration.

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For editable local development:

```bash
pip install -e .
```

### 11.2 GPU / PyTorch note

If running on a modern Blackwell GPU, use a PyTorch build that supports the GPU compute capability. In our final training environment, PyTorch CUDA 12.8 was used successfully:

```text
torch: 2.11.0+cu128
cuda: 12.8
```

For general CPU or older GPU runtime, install a PyTorch build appropriate for the host machine.

### 11.3 Frontend dependencies

Install Node dependencies:

```bash
npm install
```

Build the PDF workspace bundle:

```bash
npm run build:pdf-workspace
```

The build command is defined in `package.json` and uses Vite.

---

## 12. Configuration

### 12.1 Local environment file

The app reads local secrets and runtime configuration from:

```text
.env.local
```

`.env.local` is ignored by Git.

Common keys:

```bash
LLM_PROVIDER=gemini
LLM_MODEL=gemini-flash-latest
STRATEGY_PLANNER_PROVIDER=gemini
STRATEGY_PLANNER_MODEL=gemini-flash-latest
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-001
GEMINI_API_KEY=your_key_here

OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_SITE_URL=http://127.0.0.1:8000
OPENROUTER_SITE_NAME=Emotion-Aware Academic Assistant

OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=your_model_here
```

### 12.2 Configure Gemini from terminal

```bash
python scripts/configure_api_key.py
```

### 12.3 Configure emotion checkpoint

Install the teammate/final checkpoint:

```bash
python scripts/install_emotion_checkpoint.py --source /path/to/best.pt
```

Inspect a checkpoint:

```bash
python scripts/inspect_emotion_checkpoint.py --checkpoint /path/to/best.pt
```

Configure a checkpoint path:

```bash
python scripts/configure_emotion_checkpoint.py --checkpoint /path/to/best.pt --mode auto
```

Important environment keys:

```bash
EMOTION_CHECKPOINT_PATH=/absolute/path/to/best.pt
RAW_EMOTION_CHECKPOINT_PATH=/absolute/path/to/raw_8class_best.pt
EMOTION_MODEL_MODE=auto        # auto | academic_state | raw_emotion
```

### 12.4 Configure face detector / OpenFace

Diagnose OpenFace:

```bash
python scripts/diagnose_openface.py
```

Configure an existing OpenFace `FeatureExtraction` binary:

```bash
python scripts/configure_openface.py --bin /path/to/FeatureExtraction
```

Optional local build helper:

```bash
python scripts/build_openface_local.py --configure-project
```

Optional YOLO face weights can be placed under:

```text
models/face_detector/yolov8n-face.pt
```

---

## 13. Running the Application

### 13.1 Start web mode

```bash
python -u main.py --mode web
```

By default, the app tries:

```text
http://127.0.0.1:8000
```

If port 8000 is already in use, the server automatically tries the next available ports in the range 8000–8019.

Open:

```text
http://127.0.0.1:8000/pdf-chat
```

### 13.2 Start with custom host/port

```bash
python -u main.py --mode web --host 0.0.0.0 --port 8000
```

### 13.3 Terminal mode

A legacy terminal mode remains available:

```bash
python main.py --mode terminal
```

### 13.4 GUI mode

A PyQt GUI entry point exists, but the final demo should use the web app:

```bash
python main.py --mode gui
```

---

## 14. Demo Workflow

A recommended final demonstration sequence is:

1. Open `/settings`.
   - Show provider status.
   - Show answer model, strategy planner model, and embedding model roles.
   - Confirm that secrets are stored locally.

2. Open `/camera-debug`.
   - Capture one frame.
   - Show detected face, landmarks, crop, and 224x224 model input.
   - Show model mode: academic-state mode or raw-emotion mode.
   - Show the current learning signal and smoothing output.

3. Open `/pdf-chat`.
   - Upload or open a sample paper.
   - Highlight a difficult passage or select an area.
   - Click **Explain** to generate the Stage A baseline explanation.
   - Wait for the reaction window.
   - Show strategy candidates.
   - Select a strategy and generate the Stage C adaptive answer.

4. Open `/llm-compare`.
   - Select a saved prompt snapshot.
   - Compare the prompt across configured models.
   - Export comparison output if needed.

5. Optionally open `/pdf-test`.
   - Show parsing, retrieval, matching, and debug information.

A detailed demonstration script is available at:

```text
docs/demo_script.md
```

---

## 15. Testing and Diagnostics

### 15.1 Python tests

```bash
python -m unittest
```

### 15.2 Web smoke check

```bash
python scripts/web_smoke_check.py
```

### 15.3 Full smoke check

```bash
python scripts/smoke_check.py
```

### 15.4 Emotion adapter test

```bash
python scripts/test_emotion_adapter.py
```

### 15.5 Environment diagnosis

```bash
python scripts/diagnose_environment.py
```

### 15.6 OpenFace diagnosis

```bash
python scripts/diagnose_openface.py
python scripts/test_openface_feature_extraction.py --image /path/to/image.jpg
```

### 15.7 Frontend build check

```bash
npm run build:pdf-workspace
```

### 15.8 Whitespace check before commit

```bash
git diff --check
```

---

## 16. Local Runtime Data and Privacy

The system is designed as a local prototype.

### 16.1 Local-only data

These are stored locally and should not be committed:

```text
.env.local
runtime_uploads/
logs/*.jsonl
logs/uploads/
models/emotion_model/*.pt
models/emotion_model/*.pth
models/emotion_model/*.ckpt
models/face_detector/*.pt
models/face_detector/*.onnx
external/OpenFace/
```

### 16.2 Webcam privacy

- Webcam frames are processed locally.
- Raw frames are not persisted by default.
- External LLM providers receive only text prompts and selected context, not raw webcam images.
- `/camera-debug` is for local transparency and troubleshooting.

### 16.3 API key privacy

- API keys are stored in `.env.local`.
- Keys are masked in the settings page.
- Keys are not stored in browser localStorage or sessionStorage.
- `.env.local` is ignored by Git.

### 16.4 Uploaded papers

Uploaded PDFs and derived artifacts are stored under `runtime_uploads/`. Do not publicly share this folder if it contains private papers or prompt snapshots.

---

## 17. Known Limitations

- The current final checkpoint is a **4-class academic-state model**. It does not provide raw 8-class facial emotion output.
- Raw-emotion display requires an 8-class checkpoint to be installed.
- Emotion inference quality depends on lighting, camera angle, face visibility, and the configured detector.
- OpenFace must be installed/configured separately for full landmark transparency.
- If no LLM API key is configured, the app can fall back to a dummy LLM for testing but not for final-quality explanations.
- If embedding configuration is unavailable, RAG falls back to keyword retrieval.
- The system should be described as a learning-support assistant, not as a psychological diagnosis system.
- Model weights and private runtime uploads are intentionally excluded from Git and must be installed separately.

---

## 18. Useful Handoff Notes

### 18.1 Files most relevant for emotion/model handoff

```text
emotion_aware_assistant/emotion/labels.py
emotion_aware_assistant/emotion/raw_emotion_pipeline.py
emotion_aware_assistant/emotion/state_mapper.py
emotion_aware_assistant/emotion/emotion_buffer.py
emotion_aware_assistant/emotion/affective_trend_tracker.py
emotion_aware_assistant/emotion/teammate_emotion_adapter.py
models/emotion_model/README.md
scripts/install_emotion_checkpoint.py
scripts/inspect_emotion_checkpoint.py
scripts/configure_emotion_checkpoint.py
```

### 18.2 Files most relevant for PDF/RAG handoff

```text
emotion_aware_assistant/paper/pdf_loader.py
emotion_aware_assistant/paper/pdf_parse_pipeline.py
emotion_aware_assistant/paper/paper_rag.py
emotion_aware_assistant/paper/retriever.py
emotion_aware_assistant/paper/passage_analyzer.py
emotion_aware_assistant/web/static/pdf_chat.html
emotion_aware_assistant/web/static/pdf-workspace/
```

### 18.3 Files most relevant for LLM handoff

```text
emotion_aware_assistant/llm/providers.py
emotion_aware_assistant/llm/prompt_builder.py
emotion_aware_assistant/llm/response_policy.py
emotion_aware_assistant/core/llm_config.py
emotion_aware_assistant/web/static/llm_compare.html
```

### 18.4 Final model configuration summary

```text
Checkpoint target: models/emotion_model/best_model.pt
Metadata target:   models/emotion_model/metadata.json
Architecture:      convnext_tiny.fb_in22k_ft_in1k
Output type:       academic_state
Classes:           boredom, confusion, engagement, frustration
Best val acc:      80.67%
Test acc:          79.94%
```

---

## 19. References

The emotion-to-academic-state design is supported by research on basic emotions, affective dimensions, achievement emotions, and learning-centered affective states.

1. P. Ekman, “Basic Emotions,” in _Handbook of Cognition and Emotion_, 1999.
2. J. A. Russell, “A Circumplex Model of Affect,” _Journal of Personality and Social Psychology_, vol. 39, no. 6, pp. 1161–1178, 1980.
3. R. Pekrun, “The Control-Value Theory of Achievement Emotions: Assumptions, Corollaries, and Implications for Educational Research and Practice,” _Educational Psychology Review_, vol. 18, pp. 315–341, 2006.
4. S. K. D’Mello and A. C. Graesser, “Dynamics of Affective States During Complex Learning,” _Learning and Instruction_, vol. 22, no. 2, pp. 145–157, 2012.
5. R. S. J. d. Baker, S. K. D’Mello, M. M. T. Rodrigo, and A. C. Graesser, “Better to Be Frustrated than Bored: The Incidence, Persistence, and Impact of Learners’ Cognitive-Affective States During Interactions with Three Different Computer-Based Learning Environments,” _International Journal of Human-Computer Studies_, vol. 68, no. 4, pp. 223–241, 2010.

---

## Final Submission Checklist

Before submission or demo, confirm:

- [ ] `python -m unittest` passes.
- [ ] `npm run build:pdf-workspace` passes.
- [ ] `/settings` shows the intended LLM provider configured.
- [ ] The final ConvNeXt-Tiny checkpoint is installed locally.
- [ ] `python scripts/test_emotion_adapter.py` reports the academic-state checkpoint status.
- [ ] `/camera-debug` can analyze a frame or clearly shows the configured fallback status.
- [ ] `/pdf-chat` can upload/open a paper, create a highlight, generate a baseline explanation, show strategy candidates, and produce a strategy-conditioned answer.
- [ ] `/llm-compare` can load prompt snapshots produced by `/pdf-chat`.
- [ ] No private keys, model weights, raw data, or runtime uploads are staged for Git.
