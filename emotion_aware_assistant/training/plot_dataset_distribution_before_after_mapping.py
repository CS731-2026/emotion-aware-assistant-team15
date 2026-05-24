from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_ROOT = ROOT / "data" / "processed"

OUT_DIR = ROOT / "evaluation" / "plots"
RESULT_DIR = ROOT / "evaluation" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RAW_CLASSES = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

# 我们当前采用的映射关系
EMOTION_TO_ACADEMIC_STATE = {
    "happy": "engagement",
    "neutral": "engagement",

    "surprise": "confusion",
    "fear": "confusion",

    "anger": "frustration",
    "contempt": "frustration",
    "disgust": "frustration",

    "sad": "boredom",
}

ACADEMIC_STATES = [
    "engagement",
    "confusion",
    "frustration",
    "boredom",
]

SPLITS = ["train", "val", "test"]


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


# ------------------------------------------------------------
# 1. Count raw 8-emotion dataset
# ------------------------------------------------------------
raw_rows = []

for split in SPLITS:
    for cls in RAW_CLASSES:
        cls_dir = RAW_DATA_ROOT / split / cls
        raw_rows.append({
            "split": split,
            "raw_emotion": cls,
            "count": count_images(cls_dir),
        })

raw_df = pd.DataFrame(raw_rows)

raw_total = (
    raw_df.groupby("raw_emotion", as_index=False)["count"]
    .sum()
    .sort_values("raw_emotion")
)

# ------------------------------------------------------------
# 2. Aggregate into 4 academic states
# ------------------------------------------------------------
mapped_df = raw_df.copy()
mapped_df["academic_state"] = mapped_df["raw_emotion"].map(EMOTION_TO_ACADEMIC_STATE)

mapped_split = (
    mapped_df.groupby(["split", "academic_state"], as_index=False)["count"]
    .sum()
)

mapped_total = (
    mapped_df.groupby("academic_state", as_index=False)["count"]
    .sum()
)

# 保证顺序固定
raw_total["raw_emotion"] = pd.Categorical(
    raw_total["raw_emotion"],
    categories=RAW_CLASSES,
    ordered=True
)
raw_total = raw_total.sort_values("raw_emotion")

mapped_total["academic_state"] = pd.Categorical(
    mapped_total["academic_state"],
    categories=ACADEMIC_STATES,
    ordered=True
)
mapped_total = mapped_total.sort_values("academic_state")


# ------------------------------------------------------------
# 3. Save CSV files
# ------------------------------------------------------------
raw_df.to_csv(
    RESULT_DIR / "dataset_distribution_raw_8emotion_by_split.csv",
    index=False,
    encoding="utf-8-sig"
)

raw_total.to_csv(
    RESULT_DIR / "dataset_distribution_raw_8emotion_total.csv",
    index=False,
    encoding="utf-8-sig"
)

mapped_split.to_csv(
    RESULT_DIR / "dataset_distribution_mapped_4state_by_split.csv",
    index=False,
    encoding="utf-8-sig"
)

mapped_total.to_csv(
    RESULT_DIR / "dataset_distribution_mapped_4state_total.csv",
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 4. Plot raw 8-emotion total distribution
# ------------------------------------------------------------
plt.figure(figsize=(11, 6))
plt.bar(raw_total["raw_emotion"], raw_total["count"])
plt.title("Dataset Distribution Before Mapping: 8 Raw Emotions", fontsize=14)
plt.xlabel("Raw Emotion Class", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)
plt.xticks(rotation=35, ha="right")

for i, v in enumerate(raw_total["count"]):
    plt.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(
    OUT_DIR / "dataset_distribution_before_mapping_8emotion.png",
    dpi=200
)
plt.close()

# ------------------------------------------------------------
# 5. Plot mapped 4-state total distribution
# ------------------------------------------------------------
plt.figure(figsize=(9, 6))
plt.bar(mapped_total["academic_state"], mapped_total["count"])
plt.title("Dataset Distribution After Mapping: 4 Academic States", fontsize=14)
plt.xlabel("Academic State", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)

for i, v in enumerate(mapped_total["count"]):
    plt.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(
    OUT_DIR / "dataset_distribution_after_mapping_4state.png",
    dpi=200
)
plt.close()

# ------------------------------------------------------------
# 6. Plot split-level grouped bar charts
# ------------------------------------------------------------
def plot_grouped(df, category_col, categories, title, xlabel, out_name):
    pivot = (
        df.pivot_table(
            index=category_col,
            columns="split",
            values="count",
            aggfunc="sum",
            fill_value=0
        )
        .reindex(categories)
    )

    x = np.arange(len(categories))
    width = 0.25

    plt.figure(figsize=(11, 6))

    for idx, split in enumerate(SPLITS):
        values = pivot[split].values if split in pivot.columns else [0] * len(categories)
        plt.bar(x + (idx - 1) * width, values, width, label=split)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(x, categories, rotation=35, ha="right")
    plt.legend(title="Split")
    plt.tight_layout()
    plt.savefig(OUT_DIR / out_name, dpi=200)
    plt.close()


plot_grouped(
    raw_df,
    "raw_emotion",
    RAW_CLASSES,
    "Dataset Distribution Before Mapping by Split",
    "Raw Emotion Class",
    "dataset_distribution_before_mapping_8emotion_by_split.png"
)

plot_grouped(
    mapped_split,
    "academic_state",
    ACADEMIC_STATES,
    "Dataset Distribution After Mapping by Split",
    "Academic State",
    "dataset_distribution_after_mapping_4state_by_split.png"
)

# ------------------------------------------------------------
# 7. Print summary
# ------------------------------------------------------------
print("\nRaw 8-emotion total distribution:")
print(raw_total.to_string(index=False))

print("\nMapped 4-state total distribution:")
print(mapped_total.to_string(index=False))

print("\nSaved figures:")
print(OUT_DIR / "dataset_distribution_before_mapping_8emotion.png")
print(OUT_DIR / "dataset_distribution_after_mapping_4state.png")
print(OUT_DIR / "dataset_distribution_before_mapping_8emotion_by_split.png")
print(OUT_DIR / "dataset_distribution_after_mapping_4state_by_split.png")

print("\nSaved CSV files:")
print(RESULT_DIR / "dataset_distribution_raw_8emotion_by_split.csv")
print(RESULT_DIR / "dataset_distribution_raw_8emotion_total.csv")
print(RESULT_DIR / "dataset_distribution_mapped_4state_by_split.csv")
print(RESULT_DIR / "dataset_distribution_mapped_4state_total.csv")
