# Emotion-Aware Academic Assistant for Paper Reading Support

> **COMPSYS 731 — Human-Robot Interaction**
> University of Auckland · Group Project · Semester 1, 2026

[![Course](https://img.shields.io/badge/COMPSYS%20731-Human--Robot%20Interaction-blue)](https://www.auckland.ac.nz/)
[![Topic](https://img.shields.io/badge/Topic-Emotion--Aware%20Paper%20Reading%20Assistant-purple)](#)
[![Python](https://img.shields.io/badge/Python-3.9%2F3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/Version%20Control-GitHub-black)](https://github.com/)

An emotion-aware assistant that supports academic paper reading. It detects learners’ emotions and reading states during dialogue and adapts responses accordingly (empathize, encourage, clarify, scaffold), improving comprehension, engagement, and study outcomes.

<p align="center">
  <img src="https://images.unsplash.com/photo-1522202176988-66273c2fd55f?auto=format&fit=crop&w=1600&q=60" width="820" alt="Banner" />
</p>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Team Members](#2-team-members)
3. [System Architecture](#3-system-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Environment Setup](#5-environment-setup)
6. [Dataset Preparation](#6-dataset-preparation)
7. [Training Emotion Recognition Models](#7-training-emotion-recognition-models)
8. [Running the Application](#8-running-the-application)
9. [Model Comparison](#9-model-comparison)
10. [LLM Comparison](#10-llm-comparison)
11. [Evaluation](#11-evaluation)
12. [Code Quality & Version Control](#12-code-quality--version-control)
13. [Known Issues & Limitations](#13-known-issues--limitations)
14. [References](#14-references)

---

## 1. Project Overview

This project implements an **Emotion-Aware Academic Assistant** designed to support researchers and graduate students during paper reading sessions. The system uses a webcam to continuously detect the user's facial emotion in real time. The detected emotion is then mapped to an **academic affective state** (Confusion, Frustration, Engagement, or Boredom) and used to adapt the responses of a large language model (LLM)-powered chatbot, providing context-sensitive academic assistance.

**Core use case:** A user is reading a dense methodology section of a research paper. The system detects signs of frustration or confusion from their facial expression and proactively offers a simplified explanation, alternative resources, or an encouraging message — without the user needing to explicitly ask.

**Key technical contributions:**

- Real-time facial emotion recognition pipeline using YOLOv8n-face detection and six trained deep learning architectures (no pre-trained emotion classifiers used; all models trained from scratch on AffectNet-HQ + RAF-DB).
- A **10-frame sliding window majority-vote buffer** (`collections.deque` + `statistics.mode`) that eliminates transient expression noise and produces stable emotion labels.
- An **Emotion Mapping Layer** that translates Ekman basic emotions into academic affective states (e.g., Fear + Surprise → Confusion; Sadness + Anger → Frustration), as recommended by the course instructor.
- Integration with six large language models via the OpenRouter API, with systematic comparison of response quality, empathy, latency, and cost.
- A structured prompt template encoding the user's emotional state and the current paper context (PDF section) to generate adaptive academic responses. PDF integration is implemented as an extra-credit feature.

> **Instructor Feedback (25 Mar 2026):** "I think this application is a good example for this project. It is a type of education robot... Confusion is usually not considered as primary human emotions... So you should find a clue how to detect confusion from FER... your robot should have access to the pdf, and connect it to LLM or other solutions. It is out of scope of this project, but you can add it, and get extra points if it works."

---

## 2. Team Members

| Name | Student ID | Role |
| :--- | :--- | :--- |
| Xincheng Li | 168277355 | Emotion Recognition Pipeline, Model Training |
| Xiyu Huang | XXXXXXXX | LLM Integration, Prompt Engineering, Emotion Mapping |
| Ruoxuan Li | XXXXXXXX | Frontend UI, Evaluation, Dataset Preparation |

> **Note:** Replace the above with your actual names and student IDs before submission.

---

## 3. System Architecture

The system consists of eight sequential modules connected in a real-time pipeline:

```
Webcam Input
    │  (30 FPS BGR frames)
    ▼
Face Detection (YOLOv8n-face / MTCNN)
    │  (224×224 aligned face crop)
    ▼
Emotion Recognition Model
    │  (softmax probabilities over 8 Ekman classes)
    ▼
10-Frame Emotion Buffer
    │  (majority-vote stable label + confidence)
    ▼
Emotion Mapping Layer          ← [Responds to instructor guidance]
    │  (Ekman basic emotions → Academic Affective States)
    │  fear+surprise → Confusion
    │  sadness+anger → Frustration
    │  happiness/neutral → Engagement
    │  sustained neutral → Boredom
    ▼
Context Builder
    │  (emotion state + PDF section + user query → structured prompt)
    ▼
LLM Integration (OpenRouter API)
    │  (streaming response)
    ▼
Academic Assistant Terminal UI
```

### Module Descriptions

| Module | Technology | Description |
| :--- | :--- | :--- |
| **Webcam Input** | OpenCV `cv2.VideoCapture` | Captures 30 FPS video stream from the default webcam. |
| **Face Detection** | YOLOv8n-face (primary) / MTCNN (high-accuracy) | Detects and crops face regions. MTCNN provides 5-point landmark alignment for improved accuracy. |
| **Emotion Recognition** | PyTorch + `timm` | Classifies aligned face crops into 8 emotion categories. Six architectures trained and compared. |
| **Emotion Buffer** | `collections.deque(maxlen=10)` | Maintains a sliding window of the last 10 frames. Final label is `statistics.mode()` of the window. |
| **Emotion Mapping** | Rule-based mapping table | Translates Ekman basic emotions to academic affective states per instructor guidance. |
| **Context Builder** | PyMuPDF (`fitz`) | Parses the current PDF section and constructs a structured prompt with emotion state and paper context. |
| **LLM Integration** | OpenRouter API | Sends the structured prompt to one of six LLMs and receives a streaming response. |
| **Terminal UI** | Python `rich` | Displays chatbot response, current emotion indicator, and paper context in a formatted terminal interface. |

---

## 4. Repository Structure

```
emotion-aware-assistant/
│
├── README.md                                   # This file
├── requirements_cnn.txt                        # CNN env dependencies (cs731, Python 3.9)
├── requirements_chatbot.txt                    # Chatbot env dependencies (openrouter, Python 3.11)
├── .env.example                                # Environment variable template
├── .gitignore                                  # Excludes checkpoints/, data/raw/, __pycache__/
│
│
├── ── [Module 1: Dataset Preparation] ─────────────────────────────
│
├── data/
│   ├── raw/                                    # ⚠ NOT committed to Git
│   │   ├── AffectNet-HQ/                       # Provided by course: AffectNet-HQ images
│   │   └── RAF-DB/                             # Provided by course: RAF-DB images
│   │
│   ├── processed/                              # Cleaned + split dataset
│   │   ├── train/                              # 19,719 images (70%)
│   │   │   ├── anger/
│   │   │   ├── contempt/
│   │   │   ├── disgust/
│   │   │   ├── fear/
│   │   │   ├── happy/
│   │   │   ├── neutral/
│   │   │   ├── sad/
│   │   │   └── surprise/
│   │   ├── val/                                # 4,223 images (15%)
│   │   │   └── [same 8 classes]
│   │   └── test/                               # 4,233 images (15%)
│   │       └── [same 8 classes]
│   │
│   └── prepare_dataset.py                      # Merge AffectNet-HQ + RAF-DB,
│                                               # stratified 70/15/15 split
│
│
├── ── [Module 2: Emotion Recognition CNN] ──────────────────────────
│
├── emotion_recognition/
│   │
│   ├── dataset.py                              # ★ PROVIDED BY INSTRUCTOR
│   │                                           # EmotionDataset class: image loading,
│   │                                           # augmentation, label mapping (8 classes)
│   │
│   ├── train.py                                # ★ PROVIDED BY INSTRUCTOR (extended)
│   │                                           # Original: ConvNeXtV2-Pico
│   │                                           # Extended: --arch flag for 6 architectures
│   │
│   ├── inference.py                            # ★ PROVIDED BY INSTRUCTOR
│   │                                           # Inferencer(model_path).predict(image)
│   │                                           # Returns (predicted_class, confidence)
│   │
│   ├── yolov8_face.py                          # ★ PROVIDED BY INSTRUCTOR
│   │                                           # YOLOv8n-face detection + Inferencer
│   │                                           # Draws emotion labels on live webcam feed
│   │
│   ├── simple_cnn_and_yolov8.py                # ★ PROVIDED BY INSTRUCTOR
│   │                                           # Minimal YOLOv8 single-image test example
│   │
│   ├── architectures/                          # [NEW] 6 model architecture configs
│   │   ├── convnextv2_pico.py                  # Baseline (instructor default)
│   │   ├── efficientnet_b0.py                  # Comparison model 1
│   │   ├── swin_tiny.py                        # Comparison model 2 (Transformer)
│   │   ├── convnext_tiny.py                    # Comparison model 3
│   │   ├── resnet50.py                         # Comparison model 4 (classic baseline)
│   │   └── mobilenetv3.py                      # Comparison model 5 (lightweight)
│   │
│   ├── evaluate.py                             # [NEW] Evaluation script
│   │                                           # Outputs: Accuracy, F1, confusion matrix, ROC
│   │
│   ├── compare_models.py                       # [NEW] 6-model comparison experiment
│   │                                           # Output: results/model_comparison.csv
│   │
│   └── checkpoints/                            # ★ PROVIDED BY INSTRUCTOR (partial)
│       ├── 0.pt ~ 3.pt                         # Epoch checkpoints from instructor training
│       ├── 8.pt                                # Best model weight (ConvNeXtV2-Pico)
│       └── [arch_name]_best.pt                 # [NEW] Best weights per architecture
│
│
├── ── [Module 3: Emotion-Aware Pipeline] ───────────────────────────
│
├── pipeline/
│   ├── face_detector.py                        # Face detection wrapper
│   │                                           # Backend 1: YOLOv8n-face (fast, real-time)
│   │                                           # Backend 2: MTCNN (high-accuracy + alignment)
│   │                                           # Unified interface: detect(frame) → List[BBox]
│   │
│   ├── emotion_recogniser.py                   # Emotion inference wrapper
│   │                                           # Wraps Inferencer from inference.py
│   │                                           # Interface: recognise(face_crop)
│   │                                           #   → (label, confidence, probs_dict)
│   │
│   ├── emotion_buffer.py                       # [CORE INNOVATION] 10-frame buffer
│   │                                           # collections.deque(maxlen=10)
│   │                                           # statistics.mode() for majority vote
│   │                                           # Interface: push(label) → stable_label
│   │
│   ├── emotion_mapper.py                       # [NEW — responds to instructor guidance]
│   │                                           # Maps Ekman emotions → academic states:
│   │                                           #   fear + surprise  → Confusion
│   │                                           #   sadness + anger  → Frustration
│   │                                           #   happiness/neutral → Engagement
│   │                                           #   sustained neutral → Boredom
│   │
│   ├── context_builder.py                      # [EXTRA CREDIT — instructor confirmed]
│   │                                           # Feature 1: PDF parsing (PyMuPDF)
│   │                                           # Feature 2: Current page text extraction
│   │                                           # Feature 3: Build LLM prompt with
│   │                                           #   emotion state + paper context
│   │
│   └── llm_client.py                           # OpenRouter LLM client
│                                               # Supports 6 LLMs via --model flag
│                                               # Supports streaming output
│                                               # Interface: chat(prompt, model_id) → str
│
│
├── ── [Module 4: Main Application] ─────────────────────────────────
│
├── app/
│   ├── main.py                                 # Main entry point (terminal UI)
│   │                                           # Integrates: webcam → face detection
│   │                                           # → emotion recognition → buffer
│   │                                           # → mapping → context → LLM response
│   │                                           # CLI args: --model, --arch, --pdf
│   │
│   └── config.py                               # Global config (API keys, paths, params)
│
│
├── ── [Module 5: LLM Comparison] ───────────────────────────────────
│
├── llm_comparison/
│   ├── chatbot.py                              # ★ PROVIDED BY INSTRUCTOR
│   │                                           # Original OpenRouter chatbot (no emotion)
│   │
│   ├── emotion_chatbot.py                      # [NEW] Emotion-aware chatbot
│   │                                           # Extends chatbot.py: injects emotion
│   │                                           # state into system prompt
│   │
│   └── benchmark.py                            # 6-LLM comparison benchmark
│                                               # Models tested:
│                                               #   openai/gpt-4o-mini
│                                               #   anthropic/claude-3-5-haiku
│                                               #   deepseek/deepseek-chat
│                                               #   google/gemini-2.0-flash-001
│                                               #   meta-llama/llama-3.1-8b-instruct
│                                               #   mistralai/mistral-7b-instruct
│                                               # Metrics: latency, relevance, empathy
│
│
├── ── [Module 6: Evaluation] ───────────────────────────────────────
│
├── evaluation/
│   ├── user_study.py                           # User study (5-point Likert scale)
│   │                                           # Dimensions: usefulness, accuracy,
│   │                                           #   satisfaction, naturalness
│   │
│   ├── model_benchmark.py                      # CNN benchmark (6-arch comparison)
│   │                                           # Outputs: Accuracy/F1/speed/model size
│   │
│   ├── daisee_validation.py                    # [NEW] DAiSEE zero-shot validation
│   │                                           # Validates emotion → academic state
│   │                                           # mapping on real e-learning videos
│   │
│   └── results/                                # Experiment output directory
│       ├── model_comparison.csv
│       ├── llm_comparison.csv
│       ├── user_study_results.csv
│       ├── confusion_matrix_[arch].png
│       └── daisee_validation_report.md
│
│
└── ── [Module 7: Documentation] ────────────────────────────────────

    └── docs/
        ├── design_presentation.pdf             # Design Presentation (Week 6)
        ├── final_presentation.pdf              # Final Presentation (Week 11)
        ├── architecture_diagram.png            # System architecture diagram
        ├── emotion_mapping_table.md            # Emotion → academic state mapping table
        └── dataset_analysis/
            ├── dataset_report.md               # Dataset analysis report
            └── dataset_distribution.png        # Class distribution chart
```

---

## 5. Environment Setup

This project uses **two separate Conda environments** as specified in the course environment setup instructions.

### Prerequisites

- Anaconda or Miniconda installed
- CUDA-capable GPU (recommended: ≥ 8 GB VRAM for training; CPU inference is supported)
- Webcam (required for live demo)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-org>/emotion-aware-assistant.git
cd emotion-aware-assistant
```

### Step 2 — Create the CNN Environment (`cs731`, Python 3.9)

This environment is used for all code in `emotion_recognition/` and `pipeline/`.

```bash
conda create -n cs731 python=3.9
conda activate cs731
pip install -r requirements_cnn.txt
```

Key packages in `requirements_cnn.txt`:

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `torch` | ≥ 2.2.0 | Deep learning framework |
| `torchvision` | ≥ 0.17.0 | Image transforms and datasets |
| `timm` | ≥ 0.9.16 | Model architecture library (`convnextv2`, `swin`, etc.) |
| `ultralytics` | ≥ 8.0.0 | YOLOv8n-face detection |
| `facenet-pytorch` | ≥ 2.5.3 | MTCNN face detection (alternative backend) |
| `opencv-python` | ≥ 4.9.0 | Webcam capture and image processing |
| `scikit-learn` | ≥ 1.4.0 | Evaluation metrics (F1, confusion matrix) |

### Step 3 — Create the Chatbot Environment (`openrouter`, Python 3.11)

This environment is used for all code in `llm_comparison/` and `app/`.

```bash
conda create -n openrouter python=3.11
conda activate openrouter
pip install -r requirements_chatbot.txt
```

Key packages in `requirements_chatbot.txt`:

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `requests` | ≥ 2.31.0 | OpenRouter API calls |
| `python-dotenv` | ≥ 1.0.0 | Load `.env` configuration |
| `pymupdf` | ≥ 1.24.0 | PDF parsing (extra-credit feature) |
| `rich` | ≥ 13.7.0 | Terminal UI formatting |

### Step 4 — Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# OpenRouter API key — obtain from https://openrouter.ai/
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxx

# Default LLM model
DEFAULT_LLM_MODEL=openai/gpt-4o-mini

# Path to the best trained emotion recognition checkpoint
EMOTION_MODEL_PATH=emotion_recognition/checkpoints/efficientnet_b0_best.pt

# Emotion recognition architecture
EMOTION_ARCH=efficientnet_b0
```

> **Security:** Never commit your `.env` file. It is listed in `.gitignore` by default.

---

## 6. Dataset Preparation

The project uses the **AffectNet-HQ + RAF-DB** combined dataset provided by the course, containing **28,175 facial images** across **8 emotion categories**: Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

### Dataset Statistics

| Emotion | Total | Train (70%) | Val (15%) | Test (15%) |
| :--- | ---: | ---: | ---: | ---: |
| Surprise | 4,616 | 3,231 | 692 | 693 |
| Happy | 4,336 | 3,035 | 650 | 651 |
| Anger | 3,608 | 2,525 | 541 | 542 |
| Disgust | 3,472 | 2,430 | 520 | 522 |
| Contempt | 3,244 | 2,270 | 486 | 488 |
| Fear | 3,043 | 2,130 | 456 | 457 |
| Sad | 2,995 | 2,096 | 449 | 450 |
| Neutral | 2,861 | 2,002 | 429 | 430 |
| **Total** | **28,175** | **19,719** | **4,223** | **4,233** |

### Preparing the Dataset

Place the raw course-provided files in `data/raw/` and run the preparation script:

```bash
conda activate cs731
python data/prepare_dataset.py \
    --raw_dir data/raw \
    --output_dir data/processed \
    --split 0.70 0.15 0.15 \
    --seed 42
```

This script performs **stratified splitting** to maintain class proportions across all three splits. **Class-weighted cross-entropy loss** is used during training to handle the 1.61× class imbalance (Surprise: 4,616 vs. Neutral: 2,861).

### Note on Confusion Detection

As directed by the course instructor, **Confusion is not treated as a primary emotion** and is therefore not a training label. Instead, the `pipeline/emotion_mapper.py` module infers Confusion from the combination of Fear and Surprise outputs at inference time. This design is consistent with D'Mello & Graesser's cognitive-affective learning theory and the instructor's guidance.

---

## 7. Training Emotion Recognition Models

All six architectures are trained using the extended `emotion_recognition/train.py` script (based on the instructor-provided original). Training uses **class-weighted cross-entropy loss** and the **AdamW optimiser** with cosine annealing learning rate scheduling.

### Training a Single Model

```bash
conda activate cs731
python emotion_recognition/train.py \
    --arch efficientnet_b0 \
    --data_dir data/processed \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --output_dir emotion_recognition/checkpoints
```

### Training All Six Models (Sequential)

```bash
conda activate cs731
for arch in convnextv2_pico efficientnet_b0 swin_tiny convnext_tiny resnet50 mobilenetv3; do
    python emotion_recognition/train.py \
        --arch $arch \
        --data_dir data/processed \
        --epochs 50 \
        --batch_size 64 \
        --lr 1e-4 \
        --output_dir emotion_recognition/checkpoints
done
```

### Model Architectures Compared

| Architecture | Type | Parameters | Notes |
| :--- | :--- | ---: | :--- |
| ConvNeXtV2-Pico | Modern CNN | 3.7M | Instructor baseline model |
| EfficientNet-B0 | CNN | 5.3M | Lightweight, fast inference |
| Swin-Tiny | Transformer | 28M | Hierarchical shifted-window attention |
| ConvNeXt-Tiny | Modern CNN | 28M | CNN with Transformer design principles |
| ResNet-50 | Classic CNN | 25.6M | Strong traditional baseline |
| MobileNetV3-Large | Lightweight CNN | 5.4M | Optimised for real-time on CPU |

### Using the Instructor-Provided Checkpoint

The instructor-provided checkpoint `emotion_recognition/checkpoints/8.pt` (ConvNeXtV2-Pico) can be used directly for inference without retraining:

```bash
conda activate cs731
python emotion_recognition/inference.py
# Edit inference.py: model_path = "emotion_recognition/checkpoints/8.pt"
```

### Evaluating a Trained Model

```bash
conda activate cs731
python emotion_recognition/evaluate.py \
    --arch efficientnet_b0 \
    --checkpoint emotion_recognition/checkpoints/efficientnet_b0_best.pt \
    --data_dir data/processed/test \
    --output_dir evaluation/results
```

---

## 8. Running the Application

### Quick Start

Ensure your `.env` file is configured (see §5), then activate both environments as needed. The main application runs in the `openrouter` environment but calls the CNN inference module from `cs731`.

```bash
conda activate openrouter
python app/main.py
```

### Command-Line Options

```bash
python app/main.py \
    --arch efficientnet_b0 \
    --checkpoint emotion_recognition/checkpoints/efficientnet_b0_best.pt \
    --llm openai/gpt-4o-mini \
    --pdf path/to/your/paper.pdf \
    --buffer_size 10 \
    --camera_id 0
```

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--arch` | `convnextv2_pico` | Emotion recognition architecture |
| `--checkpoint` | From `.env` | Path to trained model weights (`.pt` file) |
| `--llm` | `openai/gpt-4o-mini` | LLM model identifier (OpenRouter format) |
| `--pdf` | `None` | Path to the PDF paper being read (extra-credit feature) |
| `--buffer_size` | `10` | Number of frames in the emotion smoothing buffer |
| `--camera_id` | `0` | Webcam device index |

### Runtime Controls

| Key | Action |
| :--- | :--- |
| `Enter` | Send a message to the chatbot |
| `e` | Display current detected emotion and academic state |
| `m` | Switch LLM model |
| `q` | Quit the application |

### Live Demo Mode (for Final Presentation)

The `yolov8_face.py` script (instructor-provided) can be run independently for a visual real-time demo:

```bash
conda activate cs731
python emotion_recognition/yolov8_face.py
```

This displays a live webcam feed with bounding boxes and emotion labels overlaid on detected faces.

---

## 9. Model Comparison

After training all six architectures, run the comparison benchmark:

```bash
conda activate cs731
python emotion_recognition/compare_models.py \
    --archs convnextv2_pico efficientnet_b0 swin_tiny convnext_tiny resnet50 mobilenetv3 \
    --checkpoints_dir emotion_recognition/checkpoints \
    --data_dir data/processed/test \
    --output evaluation/results/model_comparison.csv
```

The benchmark reports the following metrics for each architecture:

| Metric | Description |
| :--- | :--- |
| **Accuracy** | Overall classification accuracy on the test set |
| **Macro F1** | Unweighted mean F1-score across all 8 classes |
| **Weighted F1** | F1-score weighted by class support |
| **Inference Time (ms)** | Average time per frame on CPU and GPU |
| **Parameters (M)** | Total trainable parameters in millions |
| **Model Size (MB)** | Checkpoint file size on disk |

---

## 10. LLM Comparison

The system compares six large language models accessed via the **OpenRouter API** (unified endpoint: `https://openrouter.ai/api/v1/chat/completions`). All models receive identical prompts for fair comparison.

### Models Compared

| Model ID | Provider | Input Cost (per 1M tokens) | Key Strength |
| :--- | :--- | :--- | :--- |
| `openai/gpt-4o-mini` | OpenAI | $0.15 | High-frequency, low latency |
| `anthropic/claude-3-5-haiku` | Anthropic | $0.80 | Academic text comprehension |
| `deepseek/deepseek-chat` | DeepSeek | $0.27 | Mathematical formula reasoning |
| `google/gemini-2.0-flash-001` | Google | $0.10 | Long context, multimodal |
| `meta-llama/llama-3.1-8b-instruct` | Meta | $0.05 | Open-source, fast |
| `mistralai/mistral-7b-instruct` | Mistral | $0.03 | Lightweight, low cost |

### Running the LLM Benchmark

```bash
conda activate openrouter
python llm_comparison/benchmark.py \
    --test_prompts evaluation/test_prompts.json \
    --output evaluation/results/llm_comparison.csv
```

### Evaluation Metrics for LLMs

| Metric | Method |
| :--- | :--- |
| **Response Quality** | Human-rated 1–5 Likert scale (academic accuracy, relevance) |
| **Empathy Score** | Human-rated 1–5 Likert scale (tone appropriateness for detected emotion) |
| **Latency (ms)** | Time-to-first-token via OpenRouter streaming API |
| **Cost (USD)** | Actual token cost per response from API response headers |

---

## 11. Evaluation

### User Study

A within-subjects user study with **N ≥ 5 participants** (graduate students) evaluates the system's effectiveness. Participants complete a 20-minute paper reading session with and without the emotion-aware assistant.

```bash
conda activate openrouter
python evaluation/user_study.py
```

Participants complete a **pre/post Likert-scale questionnaire** (5-point scale) covering perceived reading comprehension, cognitive load (NASA-TLX adapted), satisfaction with chatbot responses, and perceived empathy of the system.

### DAiSEE Zero-Shot Validation

To validate the emotion-to-academic-state mapping, the trained model is tested on the DAiSEE dataset (IIT Hyderabad, 2016) — an e-learning video dataset labelled with Boredom, Confusion, Engagement, and Frustration. This validates whether the Ekman-to-academic-state mapping holds in real academic scenarios.

```bash
conda activate cs731
python evaluation/daisee_validation.py \
    --daisee_dir path/to/DAiSEE \
    --checkpoint emotion_recognition/checkpoints/efficientnet_b0_best.pt \
    --output evaluation/results/daisee_validation_report.md
```

### Running All Evaluations

```bash
# CNN model comparison
conda activate cs731
python emotion_recognition/compare_models.py --output_dir evaluation/results

# LLM comparison
conda activate openrouter
python llm_comparison/benchmark.py --output evaluation/results/llm_comparison.csv

# User study
python evaluation/user_study.py

# DAiSEE validation
conda activate cs731
python evaluation/daisee_validation.py --output_dir evaluation/results
```

---

## 12. Code Quality & Version Control

### Code Standards

All code follows **PEP 8** style guidelines and is documented with **Google-style docstrings**. Every function includes a description of its parameters, return values, and any raised exceptions. Inline comments explain non-obvious logic, particularly in the model training loop, the 10-frame buffer, and the prompt construction pipeline.

The five instructor-provided source files (`dataset.py`, `train.py`, `inference.py`, `yolov8_face.py`, `simple_cnn_and_yolov8.py`) are preserved with their original structure; only additional comments and minor extensions (e.g., `--arch` flag in `train.py`) are added.

### Git Workflow

The repository uses a **feature-branch workflow**:

```
main          ← stable, reviewed code only
├── dev       ← integration branch
│   ├── feature/emotion-recognition
│   ├── feature/llm-integration
│   ├── feature/emotion-mapping
│   ├── feature/pdf-context
│   └── feature/evaluation
```

Commit messages follow the **Conventional Commits** specification (e.g., `feat: add 10-frame emotion buffer`, `feat: add emotion mapping layer`, `fix: resolve YOLOv8 face detection edge case`).

### Pre-commit Checks

```bash
pip install pre-commit
pre-commit install
```

The pre-commit configuration runs `flake8` (linting) and `black` (formatting) on every commit.

---

## 13. Known Issues & Limitations

- **Lighting sensitivity:** YOLOv8n-face and MTCNN detection performance degrades under low-light conditions. Ensure adequate frontal lighting during the live demo.
- **Single-face constraint:** The current pipeline processes only the first detected face. Multi-user scenarios are not supported.
- **Confusion inference limitation:** Confusion is inferred from a combination of Fear and Surprise probabilities rather than directly classified, as it is not a primary Ekman emotion. This heuristic mapping may not generalise to all users.
- **PDF dependency:** The context builder requires a machine-readable PDF. Scanned PDFs (image-only) are not supported without additional OCR preprocessing.
- **API rate limits:** OpenRouter imposes per-minute rate limits that may cause delays during the LLM benchmark if all six models are queried in rapid succession.
- **Contempt class accuracy:** The Contempt class has the third-lowest sample count (3,244) and is frequently confused with Disgust and Neutral. This is a known limitation of the AffectNet-HQ dataset annotation.
- **Two-environment complexity:** The project requires two separate Conda environments (`cs731` for CNN, `openrouter` for chatbot). Inter-process communication between the two environments is handled via a shared JSON file or socket; see `app/config.py` for details.

---

## 14. References

The following references informed the design and implementation of this system. Full citations with DOIs are provided in the individual final report.

- Ekman, P. (1992). An argument for basic emotions. *Cognition & Emotion*, 6(3–4), 169–200.
- D'Mello, S., & Graesser, A. (2012). Dynamics of affective states during complex learning. *Learning and Instruction*, 22(2), 145–157.
- Gupta, A., et al. (2016). DAiSEE: Towards user engagement recognition in the wild. *arXiv:1609.01885*.
- Wang, W., et al. (2020). AffectNet: A database for facial expression, valence, and arousal computing in the wild. *IEEE Transactions on Affective Computing*, 11(1), 15–28.
- Li, S., & Deng, W. (2020). Deep facial expression recognition: A survey. *IEEE Transactions on Affective Computing*, 13(3), 1195–1215.
- Liu, Z., et al. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. *ICCV 2021*.
- Liu, Z., et al. (2022). A ConvNet for the 2020s. *CVPR 2022*.
- Tan, M., & Le, Q. (2021). EfficientNetV2: Smaller models and faster training. *ICML 2021*.
- Zhang, K., et al. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. *IEEE Signal Processing Letters*, 23(10), 1499–1503.
- Redmon, J., & Farhadi, A. (2018). YOLOv3: An incremental improvement. *arXiv:1804.02767*.

---

## Contributors

- Team 15

## Acknowledgements

- COMPSYS 731 course and teaching team

---

*Last updated: April 2026 · COMPSYS 731 Group Project · University of Auckland*
