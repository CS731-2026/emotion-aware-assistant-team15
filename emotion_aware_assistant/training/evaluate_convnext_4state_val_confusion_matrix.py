from pathlib import Path
import csv
import json
import numpy as np
import torch
import timm
import matplotlib.pyplot as plt

from torchvision import datasets
from torch.utils.data import DataLoader
from timm.data import resolve_data_config, create_transform

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    balanced_accuracy_score,
)

# ============================================================
# Config
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

ARCH = "convnext_tiny.fb_in22k_ft_in1k"

CKPT_PATH = (
    ROOT
    / "emotion_recognition"
    / "checkpoints"
    / "convnext_tiny_4state_25ep_b64_lr5e5"
    / "best.pt"
)

DATA_ROOT = ROOT / "data" / "processed_4state"
SPLIT = "val"

OUT_DIR = ROOT / "evaluation" / "results" / "convnext_tiny_4state_val"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
NUM_WORKERS = 4

def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        major, minor = torch.cuda.get_device_capability(0)
        sm = f"sm_{major}{minor}"
        arch_list = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]

        if sm not in arch_list:
            def _sm_to_int(s: str) -> int:
                return int(s.split("_", 1)[1])

            current = _sm_to_int(sm)
            max_supported = max((_sm_to_int(a) for a in arch_list), default=None)
            if max_supported is not None and current > max_supported:
                print(
                    f"[Warning] GPU compute capability {sm} is newer than this PyTorch build "
                    f"(max supported: sm_{max_supported}). Falling back to CPU."
                )
                return torch.device("cpu")

            print(
                f"[Warning] GPU compute capability {sm} is not in torch.cuda.get_arch_list()={arch_list}. "
                "Falling back to CPU."
            )
            return torch.device("cpu")
    except Exception as e:
        print(f"[Warning] CUDA is available but device capability check failed ({e}). Using CUDA anyway.")

    return torch.device("cuda")


DEVICE = _select_device()
PIN_MEMORY = DEVICE.type == "cuda"

EXPECTED_CLASSES = [
    "boredom",
    "confusion",
    "engagement",
    "frustration",
]

print("=" * 80)
print("ConvNeXt-Tiny 4-State Validation Evaluation")
print("=" * 80)
print("Architecture:", ARCH)
print("Checkpoint:", CKPT_PATH)
print("Dataset:", DATA_ROOT / SPLIT)
print("Device:", DEVICE.type)

# ============================================================
# Dataset
# ============================================================
model_for_config = timm.create_model(
    ARCH,
    pretrained=False,
    num_classes=len(EXPECTED_CLASSES),
)

data_config = resolve_data_config({}, model=model_for_config)
transform = create_transform(**data_config, is_training=False)

dataset = datasets.ImageFolder(DATA_ROOT / SPLIT, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

print("Detected classes:", class_names)
print("Number of validation samples:", len(dataset))

if class_names != EXPECTED_CLASSES:
    print("\n[Warning] Dataset class order is different from EXPECTED_CLASSES.")
    print("Expected:", EXPECTED_CLASSES)
    print("Detected:", class_names)
    print("The script will use detected ImageFolder class order.\n")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

# ============================================================
# Load Model
# ============================================================
model = timm.create_model(
    ARCH,
    pretrained=False,
    num_classes=num_classes,
)

try:
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
except TypeError:
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

if isinstance(ckpt, dict):
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
else:
    raise RuntimeError("Unsupported checkpoint format.")

clean_state_dict = {}

for k, v in state_dict.items():
    if k.startswith("module."):
        k = k[len("module."):]
    clean_state_dict[k] = v

missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

model.to(DEVICE)
model.eval()

# ============================================================
# Inference
# ============================================================
all_true = []
all_pred = []
all_prob = []

with torch.no_grad():
    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=PIN_MEMORY)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_true.extend(targets.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy().tolist())

y_true = np.array(all_true)
y_pred = np.array(all_pred)
y_prob = np.array(all_prob)

# ============================================================
# Metrics
# ============================================================
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(num_classes)),
)

val_acc = accuracy_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)

precision_macro = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

recall_macro = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

f1_macro = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

precision_weighted = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

recall_weighted = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

f1_weighted = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

# Specificity per class
specificities = []
sensitivities = []

for i in range(num_classes):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - TP - FN - FP

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    sensitivities.append(sensitivity)
    specificities.append(specificity)

specificity_macro = float(np.mean(specificities))
sensitivity_macro = float(np.mean(sensitivities))

summary = {
    "model": "ConvNeXt-Tiny",
    "architecture": ARCH,
    "checkpoint": str(CKPT_PATH),
    "split": SPLIT,
    "num_samples": int(len(dataset)),
    "validation_accuracy": float(val_acc),
    "balanced_accuracy": float(balanced_acc),
    "precision_macro": float(precision_macro),
    "recall_macro_sensitivity_macro": float(recall_macro),
    "sensitivity_macro_manual": float(sensitivity_macro),
    "specificity_macro": float(specificity_macro),
    "f1_macro": float(f1_macro),
    "precision_weighted": float(precision_weighted),
    "recall_weighted": float(recall_weighted),
    "f1_weighted": float(f1_weighted),
}

print("\n" + "=" * 80)
print("Overall Validation Metrics")
print("=" * 80)

for k, v in summary.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

print("\n" + "=" * 80)
print("Classification Report")
print("=" * 80)

report_text = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0,
)

print(report_text)

# ============================================================
# Save CSV / JSON outputs
# ============================================================
cm_csv = OUT_DIR / "confusion_matrix_val.csv"

with cm_csv.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["True \\ Pred"] + class_names)

    for i, row in enumerate(cm):
        writer.writerow([class_names[i]] + row.tolist())

summary_json = OUT_DIR / "overall_metrics_val.json"
summary_json.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

per_class_csv = OUT_DIR / "per_class_metrics_val.csv"

with per_class_csv.open("w", newline="", encoding="utf-8-sig") as f:
    fieldnames = [
        "class",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1_score",
        "support",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i, cls in enumerate(class_names):
        writer.writerow({
            "class": cls,
            "precision": report_dict[cls]["precision"],
            "recall_sensitivity": report_dict[cls]["recall"],
            "specificity": specificities[i],
            "f1_score": report_dict[cls]["f1-score"],
            "support": report_dict[cls]["support"],
        })

pred_csv = OUT_DIR / "predictions_val.csv"

with pred_csv.open("w", newline="", encoding="utf-8-sig") as f:
    fieldnames = [
        "image_path",
        "true_label",
        "pred_label",
        "correct",
        "confidence",
    ] + [f"prob_{c}" for c in class_names]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for idx, (path, target) in enumerate(dataset.samples):
        pred = int(y_pred[idx])
        probs = y_prob[idx]

        row = {
            "image_path": path,
            "true_label": class_names[target],
            "pred_label": class_names[pred],
            "correct": int(target == pred),
            "confidence": float(probs[pred]),
        }

        for j, c in enumerate(class_names):
            row[f"prob_{c}"] = float(probs[j])

        writer.writerow(row)

# ============================================================
# Plot Confusion Matrix: Counts
# ============================================================
fig_path = OUT_DIR / "confusion_matrix_val.png"

plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("ConvNeXt-Tiny 4-State Confusion Matrix (Validation)")
plt.xlabel("Predicted Academic State")
plt.ylabel("True Academic State")
plt.xticks(
    np.arange(num_classes),
    class_names,
    rotation=30,
    ha="right",
)
plt.yticks(np.arange(num_classes), class_names)
plt.colorbar()

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
        )

plt.tight_layout()
plt.savefig(fig_path, dpi=200)
plt.close()

# ============================================================
# Plot Normalized Confusion Matrix
# ============================================================
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

fig_norm_path = OUT_DIR / "confusion_matrix_val_normalized.png"

plt.figure(figsize=(7, 6))
plt.imshow(cm_norm, interpolation="nearest", vmin=0, vmax=1)
plt.title("ConvNeXt-Tiny 4-State Confusion Matrix Normalized (Validation)")
plt.xlabel("Predicted Academic State")
plt.ylabel("True Academic State")
plt.xticks(
    np.arange(num_classes),
    class_names,
    rotation=30,
    ha="right",
)
plt.yticks(np.arange(num_classes), class_names)
plt.colorbar()

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j,
            i,
            f"{cm_norm[i, j]:.2f}",
            ha="center",
            va="center",
        )

plt.tight_layout()
plt.savefig(fig_norm_path, dpi=200)
plt.close()

print("\n" + "=" * 80)
print("Saved Files")
print("=" * 80)
print("Confusion matrix CSV:", cm_csv)
print("Confusion matrix PNG:", fig_path)
print("Normalized confusion matrix PNG:", fig_norm_path)
print("Per-class metrics CSV:", per_class_csv)
print("Overall metrics JSON:", summary_json)
print("Prediction details CSV:", pred_csv)
