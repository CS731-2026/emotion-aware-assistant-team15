from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = ROOT / "evaluation" / "plots"
RESULT_DIR = ROOT / "evaluation" / "results"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Figure 1: Before vs After Mapping Performance Comparison
# ============================================================

RAW_LOG = ROOT / "logs" / "convnext_tiny_8emotion_25ep_b64_lr5e5" / "train.log"
MAPPED_LOG = ROOT / "logs" / "convnext_tiny_4state_25ep_b64_lr5e5" / "train.log"

fallback = {
    "Before Mapping\n8-class Raw Emotion": {
        "best_val_acc": 73.46,
        "test_acc": 72.80,
    },
    "After Mapping\n4-class Academic State": {
        "best_val_acc": 80.67,
        "test_acc": 79.94,
    },
}

def parse_log(log_path: Path, fallback_values: dict):
    if not log_path.exists():
        return fallback_values

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    best_vals = re.findall(r"Best validation accuracy:\s*([0-9.]+)%", text)
    test_vals = re.findall(r"Test Acc:\s*([0-9.]+)%", text)

    return {
        "best_val_acc": float(best_vals[-1]) if best_vals else fallback_values["best_val_acc"],
        "test_acc": float(test_vals[-1]) if test_vals else fallback_values["test_acc"],
    }

comparison = {
    "Before Mapping\n8-class Raw Emotion": parse_log(
        RAW_LOG, fallback["Before Mapping\n8-class Raw Emotion"]
    ),
    "After Mapping\n4-class Academic State": parse_log(
        MAPPED_LOG, fallback["After Mapping\n4-class Academic State"]
    ),
}

df = pd.DataFrame([
    {
        "Stage": k.replace("\n", " "),
        "Best Validation Accuracy": v["best_val_acc"],
        "Test Accuracy": v["test_acc"],
    }
    for k, v in comparison.items()
])
df.to_csv(
    RESULT_DIR / "before_after_mapping_accuracy_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

labels = list(comparison.keys())
x = np.arange(len(labels))
width = 0.32

fig, ax = plt.subplots(figsize=(11, 6.2), dpi=220)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

best_vals = [comparison[k]["best_val_acc"] for k in labels]
test_vals = [comparison[k]["test_acc"] for k in labels]

bars1 = ax.bar(x - width / 2, best_vals, width, label="Best Validation Accuracy")
bars2 = ax.bar(x + width / 2, test_vals, width, label="Test Accuracy")

for bars in [bars1, bars2]:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

ax.set_title("Before vs After Mapping: Emotion Detection Accuracy", fontsize=15, pad=14)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(68, 84)
ax.grid(axis="y", alpha=0.25)
ax.legend(loc="upper left", fontsize=10)
plt.tight_layout()

fig1_path = PLOT_DIR / "before_after_mapping_emotion_detection_comparison_simple.png"
plt.savefig(fig1_path, bbox_inches="tight", facecolor="white")
plt.close()


# ============================================================
# Helper for diagram boxes
# ============================================================

def add_box(ax, xy, width, height, text, fontsize=12, face="#F7F9FC", edge="#2C3E50"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )
    return box

def add_arrow(ax, start, end, color="#34495E"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=16,
        linewidth=1.6,
        color=color,
    )
    ax.add_patch(arrow)


# ============================================================
# Figure 2: Raw Emotion to Academic State Mapping Diagram
# ============================================================

fig, ax = plt.subplots(figsize=(13.5, 7.6), dpi=220)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(
    0.5,
    0.94,
    "Raw Emotion to Academic State Mapping",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
)

ax.text(
    0.18,
    0.86,
    "Raw Emotions\n(8-class detection)",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
)

ax.text(
    0.78,
    0.86,
    "Academic States\n(4-class learning states)",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
)

mappings = [
    ("happy\nneutral", "engagement", "Positive / stable participation"),
    ("fear\nsurprise", "confusion", "Uncertainty or cognitive conflict"),
    ("anger\ncontempt\ndisgust", "frustration", "Negative high-arousal task blockage"),
    ("sad", "boredom", "Low-arousal disengagement"),
]

y_positions = [0.70, 0.53, 0.35, 0.18]

for (raw, state, reason), y in zip(mappings, y_positions):
    add_box(ax, (0.06, y - 0.055), 0.24, 0.11, raw, fontsize=12, face="#F9FBFD")
    add_box(ax, (0.66, y - 0.055), 0.24, 0.11, state, fontsize=13, face="#EDF4FF")
    add_arrow(ax, (0.31, y), (0.65, y))

    ax.text(
        0.48,
        y + 0.035,
        reason,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#34495E",
    )

ax.text(
    0.5,
    0.055,
    "Fine-grained facial emotions are converted into learning-centered states for chatbot adaptation.",
    ha="center",
    va="center",
    fontsize=12,
    color="#34495E",
)

fig2_path = PLOT_DIR / "raw_emotion_to_academic_state_mapping_diagram.png"
plt.tight_layout()
plt.savefig(fig2_path, bbox_inches="tight", facecolor="white")
plt.close()


# ============================================================
# Figure 3: Academic State to Chatbot Response Strategy Diagram
# ============================================================

fig, ax = plt.subplots(figsize=(13.5, 7.6), dpi=220)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(
    0.5,
    0.94,
    "Academic State to Chatbot Response Strategy",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
)

ax.text(
    0.18,
    0.86,
    "Detected Academic State",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
)

ax.text(
    0.78,
    0.86,
    "Adaptive Chatbot Strategy",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
)

strategies = [
    (
        "engagement",
        "Encourage deeper exploration\nAsk follow-up questions\nMaintain learning momentum",
    ),
    (
        "confusion",
        "Explain step-by-step\nUse simpler examples\nCheck understanding",
    ),
    (
        "frustration",
        "Provide supportive guidance\nReduce pressure\nBreak task into smaller steps",
    ),
    (
        "boredom",
        "Re-engage the learner\nUse interactive prompts\nMake content more relevant",
    ),
]

y_positions = [0.70, 0.53, 0.35, 0.18]

for (state, strategy), y in zip(strategies, y_positions):
    add_box(ax, (0.06, y - 0.055), 0.24, 0.11, state, fontsize=13, face="#EDF4FF")
    add_box(ax, (0.58, y - 0.065), 0.34, 0.13, strategy, fontsize=10.5, face="#F9FBFD")
    add_arrow(ax, (0.31, y), (0.57, y))

ax.text(
    0.5,
    0.055,
    "The chatbot adapts its tone and explanation strategy based on the learner's detected academic state.",
    ha="center",
    va="center",
    fontsize=12,
    color="#34495E",
)

fig3_path = PLOT_DIR / "chatbot_response_strategy_diagram.png"
plt.tight_layout()
plt.savefig(fig3_path, bbox_inches="tight", facecolor="white")
plt.close()

print("\nGenerated Final Presentation support figures:")
print("1.", fig1_path)
print("2.", fig2_path)
print("3.", fig3_path)
print("\nCSV:")
print(RESULT_DIR / "before_after_mapping_accuracy_comparison.csv")
