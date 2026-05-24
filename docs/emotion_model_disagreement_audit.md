# Emotion Model Disagreement Audit

Date: 2026-05-21

## Current Conclusion

The direct 4-class checkpoint and the raw 8-class checkpoint both contain usable class metadata. The output label order is verified from checkpoint metadata and is not the main unresolved risk.

The unresolved risk is the training-label mapping used to create `models/emotion_model/best_model.pt`. The teammate's current `inference.py` / `emotion_mapper.py` style pipeline proves the current raw 8-class runtime mapping, but it does not prove that the older 4-class checkpoint was trained from labels generated with the same mapping.

The suspicious example remains:

```text
raw 8-class: sad 84%, disgust 16%
required mapping: sad + anger + disgust -> frustration = 100%
direct 4-class: boredom 92%
```

A high-confidence `sad -> boredom` result would be explainable if the 4-class training data was generated with a legacy rule where sadness belonged to boredom. This must be confirmed with the actual training script, label CSV, 4-class dataset folders, or dataset-level audit.

## Checkpoint Inspection

4-class checkpoint:

```text
checkpoint path: models/emotion_model/best_model.pt
arch: convnext_tiny.fb_in22k_ft_in1k
num_classes: 4
classes: ['boredom', 'confusion', 'engagement', 'frustration']
class_to_idx: {'boredom': 0, 'confusion': 1, 'engagement': 2, 'frustration': 3}
epoch: 19
val_acc: 80.66958947787964
head.fc.weight shape: (4, 768)
output_type: academic_state
metadata source: checkpoint_metadata
class order source: checkpoint_class_to_idx
```

8-class checkpoint:

```text
checkpoint path: models/emotion_model/raw_8class_best.pt
arch: convnext_tiny.fb_in22k_ft_in1k
num_classes: 8
classes: ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
class_to_idx: {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 'neutral': 5, 'sad': 6, 'surprise': 7}
epoch: 12
val_acc: 73.45555998405739
head.fc.weight shape: (8, 768)
output_type: raw_emotion
metadata source: checkpoint_metadata
class order source: checkpoint_class_to_idx
```

Both checkpoints have internally consistent class order and classifier output dimensions. This rules out the simple "runtime read class index 0 as the wrong label" hypothesis. It does not prove how the 4-class training labels were constructed.

## Training-Label Mapping Evidence

Focused repository searches for `boredom`, `confusion`, `engagement`, `frustration`, `sad`, `sadness`, `contempt`, `mapping`, `class_to_idx`, `ImageFolder`, `train`, and `dataset` found no active 4-class training script, no `emotion_recognition/train.py`, no `emotion_recognition/dataset.py`, no label CSV, and no tracked 4-class dataset folders. A focused file search for `*.csv`, `*labels*`, `*mapping*`, `*train*.py`, and `*dataset*.py` outside dependencies/runtime uploads found only runtime label code.

The archived README references training/data files that are not present in the working tree:

```text
emotion_recognition/dataset.py
emotion_recognition/train.py
data/processed/train
data/processed/val
data/processed/test
```

The active runtime mapping is:

```text
sad + anger + disgust -> frustration
fear + surprise -> confusion
contempt -> boredom
happy + neutral -> engagement
```

But `docs/archive/README_github_original.md` contains a conflicting older architecture diagram:

```text
Surprise -> Confusion
Anger + Disgust + Contempt -> Frustration
Happy + Neutral -> Engagement
Fear + Sadness -> Boredom
```

Later in the same archived document it lists the current-style mapping:

```text
fear + surprise -> Confusion
sad + anger + disgust -> Frustration
happiness/neutral -> Engagement
contempt -> Boredom
```

This conflict is the strongest local root-cause lead. If the 4-class checkpoint used the older diagram's conversion rule, then a raw sad face being classified as 4-class boredom is not a model mystery; it is a training-label mismatch.

## Preprocessing Evidence

Current runtime preprocessing reported by checkpoint inspection:

```text
input_size: 224
color_space: RGB
tensor_layout: NCHW
resize_method: PIL.Image.resize default
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]
checkpoint preprocessing metadata: not stored
```

The user-provided teammate inference code reportedly uses:

```text
Resize((224, 224))
ToTensor()
Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
```

That is a real preprocessing discrepancy to test. I did not change runtime normalization because neither checkpoint stores preprocessing metadata and a normalization change should be validated with a dataset-level audit or a known validation sample set. Both current runtime paths use the same ImageNet/timm normalization, so preprocessing mismatch is more likely a runtime-vs-training risk than a direct-vs-raw runtime inconsistency.

## Available Diagnostics

Inspect a checkpoint with metadata, head shape, and current runtime preprocessing:

```bash
python scripts/inspect_emotion_checkpoint.py \
  --checkpoint models/emotion_model/best_model.pt
```

Scan a 4-class dataset folder for raw-label tokens preserved in filenames or paths:

```bash
python scripts/audit_emotion_model_alignment.py \
  --scan-4class-dataset /path/to/4class_dataset \
  --output-dir diagnostics
```

This writes:

```text
diagnostics/emotion_4class_dataset_scan.json
```

It counts images per academic class and reports whether sad/sadness-like source paths appear under `boredom` or `frustration`. This is path-token evidence only; it is useful when original raw labels are preserved in filenames or parent paths.

Run the full dataset-level model alignment audit on raw 8-class data:

```bash
python scripts/audit_emotion_model_alignment.py \
  --dataset-root /path/to/8class_dataset \
  --raw-checkpoint models/emotion_model/raw_8class_best.pt \
  --academic-checkpoint models/emotion_model/best_model.pt \
  --max-images 1000
```

Or with a CSV:

```bash
python scripts/audit_emotion_model_alignment.py \
  --csv /path/to/labels.csv \
  --image-root /path/to/images \
  --raw-checkpoint models/emotion_model/raw_8class_best.pt \
  --academic-checkpoint models/emotion_model/best_model.pt \
  --max-images 1000
```

This writes:

```text
diagnostics/emotion_model_alignment_report.json
diagnostics/emotion_model_alignment_examples.csv
```

It reports direct 4-class accuracy against required mapped ground truth, raw-8+mapping accuracy, direct-vs-mapped agreement, confusion matrices, prediction distributions, likely label permutation, class bias, and direct accuracy against the archived legacy mapping as a diagnostic.

For one suspicious crop:

```bash
python scripts/compare_emotion_models_on_crop.py \
  --image path/to/image.png \
  --raw-label sad \
  --raw-checkpoint models/emotion_model/raw_8class_best.pt \
  --academic-checkpoint models/emotion_model/best_model.pt
```

## Answers To Current Questions

1. Was the 4-class checkpoint trained using the same 8-to-4 mapping? Not verified from local artifacts.
2. Where is the 4-class training label mapping defined? Not found in active repository files.
3. Was it trained directly from a 4-folder dataset or remapped from 8-class CSV? Unknown; neither source is present in the tracked workspace.
4. Was sad/sadness mapped to frustration or boredom? Unknown for the 4-class checkpoint; the archived README contains both possibilities.
5. Is class order verified? Yes, from checkpoint `class_to_idx`.
6. Does metadata prove training-label mapping? No.
7. Is preprocessing consistent with teammate inference code? Current runtime uses ImageNet/timm normalization; teammate inference reportedly uses 0.5/0.5 normalization. This is unresolved and should be tested before changing runtime.

## Recommendation

Do not declare the mismatch solved yet. Keep the 4-class result visible in `/camera-debug` as a diagnostic comparison, but treat the raw 8-class plus required probability aggregation as the active teacher-facing chain when raw mode is configured.

The implemented final chain is process-aware rather than single-frame direct classification:

```text
Camera frame -> Raw 8-class facial emotion inference -> Academic-state probability mapping -> Rolling buffer / reaction-window summary -> Learning process context -> Learning signal package -> Strategy planner -> LLM prompt builder -> Adaptive explanation
```

In `raw_emotion` mode, `direct_4class_diagnostic.used_for_strategy` must remain `false`. The direct 4-class checkpoint is retained for diagnostic/baseline/fallback comparison only.

The next evidence needed is one of:

- the actual 4-class training script or label-conversion script
- the 4-class label CSV
- the 4-class dataset folders, scanned for source-label path evidence
- a dataset-level audit showing whether the direct 4-class model aligns with the required mapping or with the archived legacy mapping
