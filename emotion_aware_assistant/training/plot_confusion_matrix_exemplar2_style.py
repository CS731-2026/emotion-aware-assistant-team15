from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

cm_csv = ROOT / "evaluation" / "results" / "convnext_tiny_4state_val" / "confusion_matrix_val.csv"
out_dir = ROOT / "evaluation" / "results" / "convnext_tiny_4state_val"
out_dir.mkdir(parents=True, exist_ok=True)

out_png = out_dir / "confusion_matrix_val_exemplar2_style.png"
out_norm_png = out_dir / "confusion_matrix_val_exemplar2_style_normalized.png"

# ------------------------------------------------------------
# Load confusion matrix CSV
# ------------------------------------------------------------
df = pd.read_csv(cm_csv)

# First column is usually "True \\ Pred"
class_names = df.iloc[:, 0].tolist()
cm = df.iloc[:, 1:].to_numpy(dtype=float)

# Row-normalized percentage
row_sum = cm.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

# ------------------------------------------------------------
# Plot 1: Exemplar-style count + percentage matrix
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 7), dpi=220)

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

ax.set_title("ConvNeXt-Tiny 4-State Confusion Matrix on Validation Set", fontsize=14, pad=14)
ax.set_xlabel("Predicted Academic State", fontsize=12)
ax.set_ylabel("Actual Academic State", fontsize=12)

ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
ax.set_yticklabels(class_names, fontsize=10)

# White grid lines like presentation heatmap
ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

# Add count + row percentage
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = int(cm[i, j])
        pct = cm_norm[i, j] * 100

        text_color = "white" if cm_norm[i, j] >= 0.55 else "#333333"

        ax.text(
            j,
            i,
            f"{count}\n{pct:.1f}%",
            ha="center",
            va="center",
            fontsize=10,
            color=text_color,
            fontweight="bold" if i == j else "normal",
        )

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Row-normalized ratio", fontsize=10)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(out_png, bbox_inches="tight", facecolor="white")
plt.close()

# ------------------------------------------------------------
# Plot 2: normalized-only version
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 7), dpi=220)

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

ax.set_title("ConvNeXt-Tiny 4-State Confusion Matrix Normalized (Validation)", fontsize=14, pad=14)
ax.set_xlabel("Predicted Academic State", fontsize=12)
ax.set_ylabel("Actual Academic State", fontsize=12)

ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
ax.set_yticklabels(class_names, fontsize=10)

ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        text_color = "white" if cm_norm[i, j] >= 0.55 else "#333333"
        ax.text(
            j,
            i,
            f"{cm_norm[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=11,
            color=text_color,
            fontweight="bold" if i == j else "normal",
        )

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Normalized value", fontsize=10)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(out_norm_png, bbox_inches="tight", facecolor="white")
plt.close()

print("Saved:")
print(out_png)
print(out_norm_png)
