# 🔥 Emotion-Aware Academic Paper Reading Assistant

> **COMPSYS 731 — Human-Robot Interaction Group Project**  
> **Team 15 — Emotion-Aware Academic Assistant for Paper Reading Support**  
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

This repository contains the final runtime system for an **emotion-aware academic paper reading assistant**. The system combines PDF-grounded explanation, local camera-based learning-signal estimation, emotion / academic-state modelling, pedagogical strategy planning, and LLM-based adaptive response generation.

The final demo is a local web application. A user uploads or opens a PDF, highlights a passage or selects an area in the paper, receives a paper-grounded baseline explanation, and can then receive an adaptive follow-up response based on the system's local learning-state signal.

---

## 1. Project Summary

Academic paper reading is difficult for many students because research papers often contain dense terminology, complex methodology, unfamiliar notation, and long technical explanations. During reading, students may become **confused**, **frustrated**, **bored**, or **engaged**. A standard chatbot can answer a text question, but it usually does not adapt its teaching style to the learner's affective or academic state.

This project addresses that gap by building an assistant that:

1. parses academic PDFs and retrieves paper-grounded context;
2. estimates a local learning signal from a webcam / emotion-recognition pipeline;
3. converts that signal into a pedagogical response strategy;
4. generates adaptive explanations through an LLM.

The high-level system idea is:

```text
PDF context + user question + learning signal
        -> pedagogical strategy
        -> adaptive academic support
```

The system **does not diagnose the user**. The learning signal is used only to adapt explanation style, such as providing step-by-step clarification, supportive simplification, concise re-engagement, or deeper academic expansion.

---

## 2. Final Demo Configuration

The repository supports both a direct 4-state academic-state checkpoint and an 8-class raw facial-emotion checkpoint. The final demo was configured to use the **8-class raw-emotion mode**. In this mode, the checkpoint outputs raw emotion probabilities, which are mapped into academic-state evidence and interpreted through reaction-window logic before being used as a learning-support cue.

The 4-state academic-state checkpoint remains available as a baseline, fallback, or comparison model. It is useful for direct academic-state classification experiments, but it was not the active final demo runtime model.

| Component | Role | Result |
| --- | --- | ---: |
| **8-class raw-emotion model** | Final demo runtime input model | Best Val Acc **73.46%**, Test Acc **72.80%** |
| **4-state academic-state model** | Baseline / fallback / comparison model | Best Val Acc **80.67%**, Test Acc **79.94%** |
| **Mapping + reaction-window layer** | Converts raw probabilities into process-aware academic-state support cues | Runtime adaptation layer |

---

## 3. Final System Features

The final implementation includes:

- a local web app for academic paper reading and chatbot interaction;
- PDF upload, parsing, page/block indexing, highlighting, and area selection;
- RAG-style retrieval from selected paper content and nearby context;
- local camera / emotion model integration;
- support for both 4-class academic-state and 8-class raw-emotion checkpoints;
- learning-state smoothing through a rolling emotion buffer;
- reaction-window monitoring and summary for strategy planning;
- a three-stage LLM pipeline: baseline explanation, strategy planning, and strategy-conditioned answer generation;
- local LLM provider configuration;
- debugging pages for PDF/RAG, camera/emotion, and LLM comparison;
- local runtime logs for analysis and evaluation.

---

## 4. Main User Routes

| Route           | Purpose                                                                                                                                   | Recommended Use          |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `/pdf-chat`     | Final paper reading assistant. Upload/open PDF, highlight text or select an area, ask questions, receive baseline and adaptive responses. | Main demo page           |
| `/settings`     | Configure API keys, LLM providers, emotion checkpoint, face detector, and local runtime settings.                                         | Setup / developer        |
| `/camera-debug` | Inspect camera frame, detected face crop, model input, raw/mapped state, confidence, and reaction window.                                 | Transparency / debugging |
| `/llm-compare`  | Compare saved prompt snapshots across different LLM models.                                                                               | Evaluation / analysis    |
| `/pdf-test`     | Debug PDF parsing, retrieval, matching, area selection, and highlight behaviour.                                                          | Developer debugging      |

---

## 5. Repository Structure

The current final project is organised as follows:

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
│   │   ├── response_policy.py
│   │   └── teammate_llm_adapter.py
│   ├── paper/
│   │   ├── document.py
│   │   ├── paper_rag.py
│   │   ├── passage_analyzer.py
│   │   ├── pdf_loader.py
│   │   ├── pdf_parse_pipeline.py
│   │   ├── retriever.py
│   │   └── text_chunker.py
│   ├── speech/
│   ├── ui/
│   └── web/
│       ├── routes.py
│       ├── schemas.py
│       ├── server.py
│       ├── state.py
│       └── static/
│           ├── camera_debug.html
│           ├── llm_compare.html
│           ├── local_settings.html
│           ├── pdf_chat.html
│           └── pdf_test.html
├── models/
│   ├── emotion_model/
│   │   └── README.md
│   └── face_detector/
│       └── README.md
├── sample_data/
├── scripts/
└── tests/
```

Runtime folders such as `runtime_uploads/`, local settings, generated logs, and large model weights are intentionally ignored by Git unless they are explicitly tracked through Git LFS.

---

## 6. Core Architecture

The final system follows this end-to-end pipeline:

```text
PDF Upload / Sample Paper
        |
        v
PDF Parsing + RAG Preparation
        |
        v
User Highlight / Area Selection
        |
        v
Stage A: Paper-Grounded Baseline Explanation
        |
        v
Local Camera / Emotion or Academic-State Signal
        |
        v
Reaction Window + State Smoothing
        |
        v
Stage B: Pedagogical Strategy Planner
        |
        v
User Selects or Applies Recommended Strategy
        |
        v
Stage C: Strategy-Conditioned Adaptive Answer
        |
        v
Prompt Snapshots + Conversation Threads + Evaluation Logs
```

The local backend uses `http.server.ThreadingHTTPServer`. The main route logic is implemented in:

```text
emotion_aware_assistant/web/server.py
emotion_aware_assistant/web/routes.py
emotion_aware_assistant/web/state.py
```

---

## 7. PDF and RAG Pipeline

The PDF pipeline is implemented mainly in:

```text
emotion_aware_assistant/paper/pdf_loader.py
emotion_aware_assistant/paper/pdf_parse_pipeline.py
emotion_aware_assistant/paper/paper_rag.py
emotion_aware_assistant/paper/retriever.py
emotion_aware_assistant/paper/passage_analyzer.py
emotion_aware_assistant/paper/text_chunker.py
```

Uploaded or sample documents are stored locally under:

```text
runtime_uploads/documents/<document_id>/
```

A prepared document may contain:

```text
meta.json
original.pdf
parsed/document.md
parsed/content_list.json
parsed/blocks_index.json
rag/paper_profile.json
rag/keyword_index.json
rag/embeddings.json
highlights/
threads/
prompt_snapshots/
logs/
```

The explanation pipeline can use selected text, page number, nearby context, matched parsed block, retrieved chunks, candidate captions, and paper profile data. If embedding retrieval is unavailable, the system falls back to keyword retrieval.

---

## 8. Emotion and Academic-State Pipeline

The emotion pipeline is implemented mainly in:

```text
emotion_aware_assistant/emotion/raw_emotion_pipeline.py
emotion_aware_assistant/emotion/labels.py
emotion_aware_assistant/emotion/emotion_buffer.py
emotion_aware_assistant/emotion/affective_trend_tracker.py
emotion_aware_assistant/emotion/camera_worker.py
emotion_aware_assistant/emotion/face_detector.py
emotion_aware_assistant/emotion/teammate_emotion_adapter.py
```

Browser webcam frames are sent only to the local backend. They are not sent to external LLM providers. Raw frames are not persisted by default.

The final demo used the OpenFace-supported detection/crop path. When OpenFace is configured, face analysis uses the `FeatureExtraction` binary to produce facial landmarks. `/camera-debug` shows the analyzed frame, landmarks, face bounding box, crop preview, and final `224 x 224` model input. `/pdf-chat` uses the same pipeline internally but only shows a compact learning signal.

### 8.1 Final raw-emotion runtime mode

This is the final demo runtime mode for chatbot adaptation.

```text
Input image / crop -> ConvNeXt-Tiny -> 8 raw emotion probabilities -> academic-state evidence -> reaction-window support cue
```

In this mode:

- the checkpoint predicts raw facial-emotion probabilities;
- raw probabilities are mapped into engagement, confusion, frustration, and boredom evidence;
- reaction-window logic summarises the signal over time;
- the resulting learning-support cue is passed to the strategy planner.

Expected final demo checkpoint path:

```text
models/emotion_model/raw_8class_best.pt
```

### 8.2 Supported 4-state academic-state mode

The repository also supports a direct 4-state academic-state checkpoint.

```text
Input image / crop -> ConvNeXt-Tiny -> boredom / confusion / engagement / frustration
```

This mode remains useful as a baseline, fallback, or comparison model.

### 8.3 Smoothing and trend tracking

The implementation includes:

- `EmotionBuffer` for majority-vote smoothing;
- `AffectiveTrendTracker` for short-window trend and hysteresis;
- confidence thresholds and high-confidence switching rules in the default configuration.

Default configuration values include:

```text
buffer_size = 10
confidence_threshold = 0.35
trend_window_sec = 6
trend_update_interval_sec = 0.5
hysteresis_updates = 3
high_confidence_switch_threshold = 0.80
```

---

## 9. Raw Emotion to Academic-State Mapping

The implemented mapping is defined in:

```text
emotion_aware_assistant/emotion/labels.py
emotion_aware_assistant/emotion/raw_emotion_pipeline.py
```

The probability-level mapping is:

| Raw emotion probabilities        | Academic state |
| -------------------------------- | -------------- |
| `P(sad) + P(anger) + P(disgust)` | `frustration`  |
| `P(fear) + P(surprise)`          | `confusion`    |
| `P(contempt)`                    | `boredom`      |
| `P(happy) + P(neutral)`          | `engagement`   |

The single-label mapping is:

| Raw emotion       | Academic state |
| ----------------- | -------------- |
| `sad`             | `frustration`  |
| `anger` / `angry` | `frustration`  |
| `disgust`         | `frustration`  |
| `fear`            | `confusion`    |
| `surprise`        | `confusion`    |
| `contempt`        | `boredom`      |
| `happy`           | `engagement`   |
| `neutral`         | `engagement`   |

Important note:

> In the final demo configuration, the system uses the 8-class raw-emotion checkpoint as low-level facial evidence. Raw probabilities are mapped into academic-state evidence and interpreted through reaction-window logic before being used as a learning-support cue rather than a user diagnosis. The direct 4-state checkpoint remains supported as a baseline, fallback, or comparison model.

---

## 10. Response Strategy Mapping

The academic state is converted into a response strategy:

| Academic state | Chatbot response strategy  |
| -------------- | -------------------------- |
| `confusion`    | Step-by-step clarification |
| `frustration`  | Supportive simplification  |
| `boredom`      | Concise re-engagement      |
| `engagement`   | Deeper academic expansion  |
| `uncertain`    | Neutral adaptive support   |

This design ensures that the emotion-recognition output influences the chatbot's teaching style rather than merely being displayed as a label.

---

## 11. Model Training and Results

Training was conducted outside the runtime repository. The runtime repository is designed to load the selected checkpoints from `models/emotion_model/`.

### 11.1 4-state academic-state comparison model

The 4-state academic-state model remains supported as a baseline, fallback, and comparison model:

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
Final epoch validation accuracy: 80.19%
Test accuracy using best checkpoint: 79.94%
```

The 4-state model was produced by first constructing a mapped dataset:

```text
8-class raw emotion labels
        -> mapping rules
        -> data/processed_4state
        -> ConvNeXt-Tiny 4-class training
```

This means the 4-state model is not an 8-class model with mapping applied at prediction time. It is directly trained as a 4-class academic-state classifier.

Training command used in the external training environment:

```bash
python emotion_recognition/train.py \
  --arch convnext_tiny.fb_in22k_ft_in1k \
  --data-root data/processed_4state \
  --epochs 25 \
  --batch-size 64 \
  --lr 5e-5 \
  --num-workers 4 \
  --save-dir checkpoints/convnext_tiny_4state_25ep_b64_lr5e5
```

### 11.2 Final 8-class raw-emotion runtime model

A second ConvNeXt-Tiny model was trained for raw emotion detection. This checkpoint was used as the final demo runtime input model.

```text
Model: ConvNeXt-Tiny
Architecture: convnext_tiny.fb_in22k_ft_in1k
Output classes: anger, contempt, disgust, fear, happy, neutral, sad, surprise
Epochs: 25
Batch size: 64
Learning rate: 5e-5
Best validation accuracy: 73.46%
Final epoch validation accuracy: 72.30%
Test accuracy: 72.80%
```

This result is expected to be lower than the 4-state task because raw emotion detection is more fine-grained. For example, `anger`, `disgust`, and `contempt` can be visually similar, while the mapped academic-state task groups related emotional evidence into broader learning states.

### 11.3 Six-model comparison for the 4-state task

| Rank | Model             | Architecture                                    | Epochs | Batch |     LR | Best epoch | Highest val acc | Final val acc |   Test acc | Role                       |
| ---: | ----------------- | ----------------------------------------------- | -----: | ----: | -----: | ---------: | --------------: | ------------: | ---------: | -------------------------- |
|    1 | **ConvNeXt-Tiny** | `convnext_tiny.fb_in22k_ft_in1k`                |     25 |    64 | `5e-5` |         19 |      **80.67%** |        80.19% | **79.94%** | Best 4-state comparison model |
|    2 | RegNetY-800MF     | `regnety_008_tv.tv2_in1k`                       |     25 |    64 | `1e-4` |         24 |          79.00% |        78.04% |     78.59% | Efficient CNN candidate    |
|    3 | Swin-Tiny         | `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` |     25 |    64 | `3e-5` |         21 |          78.72% |        78.68% |     78.43% | Transformer candidate      |
|    4 | MobileNetV3-Large | `mobilenetv3_large_100.miil_in21k_ft_in1k`      |     25 |    64 | `3e-4` |         22 |          78.68% |        77.64% |     76.13% | Lightweight candidate      |
|    5 | ResNet50          | `resnet50.a1_in1k`                              |     25 |    64 | `1e-4` |         21 |          78.32% |        77.52% |     76.49% | Baseline                   |
|    6 | EfficientNet-B4   | `tf_efficientnet_b4.ns_jft_in1k`                |     25 |    32 | `5e-5` |         22 |          75.33% |        74.77% |     74.62% | Not selected               |

### 11.4 Why ConvNeXt-Tiny was used for the 4-state comparison

ConvNeXt-Tiny was used as the strongest 4-state comparison model because it achieved:

- the highest 4-state validation accuracy: **80.67%**;
- the highest 4-state test accuracy: **79.94%**;
- stable performance across later epochs;
- better performance than the ResNet50 baseline and all other candidates.

### 11.5 Evaluation figures and training artifacts

Recommended training artifacts, if included in the repository, should be placed under:

```text
docs/model_training/
├── model_training_summary.md
├── results/
│   ├── final6_model_summary.csv
│   ├── final6_per_epoch_metrics.csv
│   └── convnext_tiny_4state_val/
│       ├── confusion_matrix_val.csv
│       ├── overall_metrics_val.json
│       └── per_class_metrics_val.csv
├── figures/
│   ├── final6_validation_accuracy_curve_only.png
│   ├── confusion_matrix_val_exemplar2_style.png
│   ├── before_after_mapping_emotion_detection_comparison_simple.png
│   ├── dataset_distribution_before_mapping_8emotion.png
│   ├── dataset_distribution_after_mapping_4state.png
│   ├── raw_emotion_to_academic_state_mapping_diagram.png
│   └── chatbot_response_strategy_diagram.png
└── logs/
    ├── convnext_tiny_4state_train.log
    └── convnext_tiny_8emotion_train.log
```

Full training datasets and intermediate epoch checkpoints should not be committed.

---

## 12. Model Checkpoint Installation

The complete source code is available in this GitHub Classroom repository. Because the trained emotion checkpoints are large binary files and are not stored directly in the normal Git repository, the checkpoint files used by the final demo are provided separately through [REANNZ FileSender](https://filesender.reannz.co.nz/?s=download&token=d1fcd0a7-d7e4-4c8b-87c5-7c1380827c74). The archive contains the final 8-class raw-emotion runtime checkpoint, `raw_8class_best.pt`, and the supported 4-state academic-state checkpoint, `best_model.pt`. The expected model paths and installation steps are documented below, so the application can be reproduced by downloading the archive and placing the checkpoint files in the corresponding model directory.

Expected runtime files:

```text
models/emotion_model/raw_8class_best.pt     # final demo raw-emotion runtime checkpoint
models/emotion_model/best_model.pt          # supported 4-state baseline / fallback / comparison checkpoint
models/emotion_model/metadata.json          # model metadata and metrics
```

Configure the final demo raw-emotion checkpoint:

```bash
python scripts/configure_emotion_checkpoint.py --checkpoint models/emotion_model/raw_8class_best.pt --mode raw_emotion
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
EMOTION_CHECKPOINT_PATH=/absolute/path/to/best_model.pt
RAW_EMOTION_CHECKPOINT_PATH=/absolute/path/to/raw_8class_best.pt
EMOTION_MODEL_MODE=auto        # auto | academic_state | raw_emotion
```

### Optional Git LFS policy

By default, `.pt`, `.pth`, and `.ckpt` files under `models/emotion_model/` are ignored. If the submission requires checkpoints to be stored in GitHub, use Git LFS and update the ignore policy accordingly:

```bash
git lfs install
git lfs track "models/emotion_model/*.pt"
```

Then ensure checkpoint files are either force-added or the relevant ignore rules are adjusted before committing. For normal development, keep checkpoints outside Git and install them locally.

---

## 13. LLM and Prompt Pipeline

The system uses three major prompt stages.

### Stage A — RAG Baseline Explanation

Purpose: generate the first paper-grounded explanation.

Main inputs:

- selected text or selected visual area metadata;
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

---

## 14. LLM Providers and Configuration

Use `/settings` to configure local provider credentials and role-specific models.

Supported provider types:

- Gemini;
- OpenRouter;
- OpenAI-compatible endpoints;
- dummy providers for offline testing.

Based on the final LLM comparison, Claude was selected for the final demo because it achieved the best overall quality and latency balance. The final OpenRouter model ID used successfully was `anthropic/claude-opus-4.7-fast`.

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

Example `.env.local` keys:

```bash
LLM_PROVIDER=openrouter
LLM_MODEL=anthropic/claude-opus-4.7-fast
STRATEGY_PLANNER_PROVIDER=openrouter
STRATEGY_PLANNER_MODEL=anthropic/claude-opus-4.7-fast
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-001
GEMINI_API_KEY=your_key_here

OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=anthropic/claude-opus-4.7-fast
OPENROUTER_SITE_URL=http://127.0.0.1:8000
OPENROUTER_SITE_NAME=Emotion-Aware Academic Assistant

OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=your_model_here
```

Final LLM comparison evidence is stored in:

```text
docs/evaluation/llm_comparison_final/2026-05-25-result.xlsx
docs/evaluation/llm_comparison_final/test-result-final-02.zip
docs/evaluation/llm_comparison_final/llm_comparison_summary.md
```

The final LLM comparison used an LLM-as-a-judge evaluation with human supervision. Responses were scored against fixed rubrics for answer quality, strategy quality, safety, and latency.

Configure Gemini from terminal:

```bash
python scripts/configure_api_key.py
```

---

## 15. Installation

### 15.1 Recommended Python version

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

### 15.2 GPU / PyTorch note

If running on a modern Blackwell GPU, use a PyTorch build that supports the GPU compute capability. In our final training environment, PyTorch CUDA 12.8 was used successfully:

```text
torch: 2.11.0+cu128
cuda: 12.8
```

For general CPU or older GPU runtime, install a PyTorch build appropriate for the host machine.

### 15.3 Frontend dependencies

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

## 16. Face Detector / OpenFace Setup

The final demo used the OpenFace-supported detection/crop path, while the implementation keeps fallback detector options for configuration and debugging.

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

## 17. Running the Application

### 17.1 Start web mode

```bash
python -u main.py --mode web
```

If port 8000 is already in use, the server automatically tries the next available ports in the range 8000–8019.

Open:

```text
http://127.0.0.1:8000/pdf-chat
```

### 17.2 Start with custom host/port

```bash
python -u main.py --mode web --host 0.0.0.0 --port 8000¸¸¸
```

### 17.3 Terminal mode

A legacy terminal mode remains available:

```bash
python main.py --mode terminal
```

### 17.4 GUI mode

A PyQt GUI entry point exists, but the final demo should use the web app:

```bash
python main.py --mode gui
```

---

## 18. Demo Workflow

A recommended final demonstration sequence is:

1. Open `/settings`.
   - Show provider status.
   - Show answer model, strategy planner model, and embedding model roles.
   - Confirm that secrets are stored locally.

2. Open `/camera-debug`.
   - Capture one frame.
   - Show the exact analyzed frame, landmarks, face crop, and model input.
   - Show that the active checkpoint is raw-emotion mode.
   - Show raw detection, mapped academic-state evidence, and the reaction-window support cue.

3. Open `/pdf-chat`.
   - Upload or open a paper.
   - Highlight a passage or select an area.
   - Click **Explain**.
   - Show the Stage A baseline explanation.
   - Wait for the reaction window.
   - Show Stage B strategy candidates.
   - Select a strategy and generate the Stage C adaptive explanation.

4. Open `/llm-compare`.
   - Select a saved prompt snapshot.
   - Compare outputs across configured models.
   - Show JSON validity for strategy planner snapshots if available.

5. Open `/pdf-test` only if needed.
   - Use it to inspect PDF parsing, selection, and retrieval behaviour.

---

## 19. Testing

Run the test suite with:

```bash
pytest
```

Current tests cover:

- core workflow;
- web routes and API contracts;
- PDF parsing and RAG;
- PDF chat backend and page behaviour;
- camera-debug route;
- emotion checkpoint scripts;
- LLM comparison workflow;
- local environment configuration;
- OpenFace helper scripts;
- product hardening checks.

Representative test files include:

```text
tests/test_core_flow.py
tests/test_web_api.py
tests/test_pdf_chat_backend.py
tests/test_pdf_chat_page.py
tests/test_pdf_rag.py
tests/test_camera_debug.py
tests/test_llm_compare.py
tests/test_emotion_checkpoint_scripts.py
tests/test_openface_scripts.py
```

---

## 20. Privacy and Responsible AI Notes

- Browser camera frames are processed locally by the backend.
- Raw camera frames are not persisted by default.
- API keys are stored locally in `.env.local` and are not shown in plaintext in the browser UI.
- The learning signal is used to adapt explanation style, not to diagnose the user.
- Prompts should avoid saying that the user is definitively frustrated, confused, bored, or engaged.
- Paper context controls factual content; the learning signal controls teaching style.
- The system should not claim psychological certainty from facial cues.

---

## 21. Common Troubleshooting

| Problem                                    | Check                                                                                             |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `/pdf-chat` does not load                  | Confirm server is running and the port is correct.                                                |
| PDF parsing fails                          | Check `pymupdf` installation and file permissions.                                                |
| Embedding preparation fails                | Check embedding provider, model name, and API key in `/settings`.                                 |
| LLM response fails                         | Check provider credentials and selected model.                                                    |
| Camera does not start                      | Check browser camera permission and use `/camera-debug`.                                          |
| OpenFace unavailable                       | Run `python scripts/diagnose_openface.py`.                                                        |
| Emotion checkpoint missing                 | Install or configure `models/emotion_model/raw_8class_best.pt` for the final demo mode.           |
| Raw emotion not shown                      | Check that `EMOTION_MODEL_MODE=raw_emotion` and `RAW_EMOTION_CHECKPOINT_PATH` are configured.     |
| Prompt snapshots missing in `/llm-compare` | Run a new explanation in `/pdf-chat` first.                                                       |

---

## 22. Files Not Intended for Git

Do not commit:

```text
.env
.env.local
runtime_uploads/
logs/*.jsonl
logs/assistant.log
models/emotion_model/*.pt
models/emotion_model/*.pth
models/emotion_model/*.ckpt
models/face_detector/*.pt
models/face_detector/*.pth
data/raw/
data/processed/
data/processed_4state/
external/OpenFace/
__pycache__/
node_modules/
```

If checkpoint files must be submitted through GitHub, use Git LFS and adjust `.gitignore` intentionally.

---

## 23. Final Presentation Alignment

The final presentation should describe the emotion-recognition component as follows:

1. **Method**: build an 8-class raw emotion dataset and a mapped 4-state academic-state dataset.
2. **Model comparison**: compare six pretrained models on the 4-state task.
3. **Result**: ConvNeXt-Tiny performs best on the 4-state comparison task.
4. **Runtime decision**: the final demo uses the 8-class raw-emotion checkpoint as low-level evidence.
5. **Adaptation chain**: raw probabilities are mapped into academic-state evidence, interpreted through reaction-window logic, and passed to the strategy planner as a learning-support cue.
6. **Supported fallback**: the 4-state academic-state checkpoint remains available as a baseline, fallback, and comparison model.

This wording keeps the README, final code, and presentation consistent.

---

## 24. Suggested Citation / Reference Themes

The emotion-to-academic-state design is motivated by three related ideas:

- basic emotion recognition for facial emotion labels;
- valence/arousal and affective dimensional theory for grouping emotions;
- educational affect research showing that confusion, frustration, boredom, and engagement are meaningful learning-related states.

These references can be discussed in the report or presentation:

```text
Ekman, P. (1999). Basic Emotions.
Russell, J. A. (1980). A Circumplex Model of Affect.
Pekrun, R. (2006). The Control-Value Theory of Achievement Emotions.
D'Mello, S. K., & Graesser, A. C. (2012). Dynamics of Affective States During Complex Learning.
Baker, R. S. J. d., D'Mello, S. K., Rodrigo, M. M. T., & Graesser, A. C. (2010). Better to Be Frustrated than Bored.
```

---

## 25. Current Project Status

The project currently includes:

- final local web application;
- PDF reading and RAG explanation workflow;
- camera-debug transparency workflow;
- LLM comparison workflow;
- academic-state and raw-emotion checkpoint compatibility;
- final demo ConvNeXt-Tiny 8-class raw emotion runtime mode;
- supported ConvNeXt-Tiny 4-state baseline / fallback / comparison results;
- tests for key backend and frontend workflows;
- demo script and configuration scripts.

The core final model decision is:

```text
Final demo emotion runtime model: ConvNeXt-Tiny 8-class raw-emotion model
Raw emotion test accuracy: 72.80%
Supported comparison model: ConvNeXt-Tiny 4-state academic-state model
4-state best validation accuracy: 80.67%
4-state test accuracy: 79.94%
```
