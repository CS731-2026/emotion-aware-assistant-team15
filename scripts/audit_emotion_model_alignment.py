from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emotion_aware_assistant.emotion.raw_emotion_pipeline import (  # noqa: E402
    ACADEMIC_STATE_CLASSES,
    RAW_EMOTION_CLASSES,
    EmotionMapper,
    RawEmotionInferencer,
    inspect_checkpoint_file,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ACADEMIC_LABELS = list(ACADEMIC_STATE_CLASSES)
RAW_LABELS = list(RAW_EMOTION_CLASSES)
RAW_SOURCE_LABEL_PATTERNS = {
    "anger": re.compile(r"(?<![a-z0-9])(anger|angry)(?![a-z0-9])", re.IGNORECASE),
    "contempt": re.compile(r"(?<![a-z0-9])contempt(?![a-z0-9])", re.IGNORECASE),
    "disgust": re.compile(r"(?<![a-z0-9])disgust(?![a-z0-9])", re.IGNORECASE),
    "fear": re.compile(r"(?<![a-z0-9])fear(?![a-z0-9])", re.IGNORECASE),
    "happy": re.compile(r"(?<![a-z0-9])(happy|happiness)(?![a-z0-9])", re.IGNORECASE),
    "neutral": re.compile(r"(?<![a-z0-9])neutral(?![a-z0-9])", re.IGNORECASE),
    "sad": re.compile(r"(?<![a-z0-9])(sad|sadness)(?![a-z0-9])", re.IGNORECASE),
    "surprise": re.compile(r"(?<![a-z0-9])surprise(?![a-z0-9])", re.IGNORECASE),
}

REQUIRED_RAW_TO_ACADEMIC = {
    "sad": "frustration",
    "anger": "frustration",
    "disgust": "frustration",
    "fear": "confusion",
    "surprise": "confusion",
    "contempt": "boredom",
    "happy": "engagement",
    "neutral": "engagement",
}

# Kept only as a diagnostic because docs/archive/README_github_original.md contains
# this older mapping in the architecture diagram. It is not the active requirement.
LEGACY_ARCHIVED_RAW_TO_ACADEMIC = {
    "anger": "frustration",
    "disgust": "frustration",
    "contempt": "frustration",
    "surprise": "confusion",
    "happy": "engagement",
    "neutral": "engagement",
    "fear": "boredom",
    "sad": "boredom",
}


@dataclass(frozen=True)
class DatasetSample:
    path: Path
    raw_label: str


def canonical_raw_label(label: Any) -> str:
    text = str(label or "").strip().lower()
    aliases = {
        "angry": "anger",
        "happiness": "happy",
        "sadness": "sad",
    }
    return aliases.get(text, text)


def map_raw_label_to_academic(label: Any) -> str:
    canonical = canonical_raw_label(label)
    try:
        return REQUIRED_RAW_TO_ACADEMIC[canonical]
    except KeyError as exc:
        raise ValueError(f"Unsupported raw emotion label: {label}") from exc


def map_raw_label_to_legacy_archived_academic(label: Any) -> str:
    canonical = canonical_raw_label(label)
    try:
        return LEGACY_ARCHIVED_RAW_TO_ACADEMIC[canonical]
    except KeyError as exc:
        raise ValueError(f"Unsupported raw emotion label: {label}") from exc


def collect_dataset_root_samples(dataset_root: str | Path) -> list[DatasetSample]:
    root = Path(dataset_root).expanduser()
    samples: list[DatasetSample] = []
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    for class_dir in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        raw_label = canonical_raw_label(class_dir.name)
        if raw_label not in RAW_LABELS:
            continue
        for image_path in sorted(class_dir.rglob("*"), key=lambda item: str(item).lower()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(DatasetSample(path=image_path, raw_label=raw_label))
    return samples


def collect_csv_samples(csv_path: str | Path, image_root: str | Path | None = None) -> list[DatasetSample]:
    label_path = Path(csv_path).expanduser()
    if not label_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {label_path}")
    root = Path(image_root).expanduser() if image_root else label_path.parent
    with label_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return []
    header = [cell.strip().lower() for cell in rows[0]]
    image_col = _first_column_index(header, ("image_path", "path", "filepath", "file", "filename", "image"))
    label_col = _first_column_index(header, ("raw_label", "label", "emotion", "class", "class_name"))
    has_header = image_col is not None and label_col is not None
    data_rows = rows[1:] if has_header else rows
    if image_col is None:
        image_col = 0
    if label_col is None:
        label_col = 1
    samples: list[DatasetSample] = []
    for row in data_rows:
        if len(row) <= max(image_col, label_col):
            continue
        raw_path = row[image_col].strip()
        raw_label = canonical_raw_label(row[label_col])
        if not raw_path or raw_label not in RAW_LABELS:
            continue
        image_path = Path(raw_path).expanduser()
        if not image_path.is_absolute():
            image_path = root / image_path
        samples.append(DatasetSample(path=image_path, raw_label=raw_label))
    return samples


def scan_4class_dataset_for_raw_source_labels(dataset_root: str | Path, max_examples: int = 25) -> dict[str, Any]:
    root = Path(dataset_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"4-class dataset root does not exist: {root}")
    class_counts = {label: 0 for label in ACADEMIC_LABELS}
    mentions = {academic: {raw: 0 for raw in RAW_LABELS} for academic in ACADEMIC_LABELS}
    examples = {academic: {raw: [] for raw in RAW_LABELS} for academic in ACADEMIC_LABELS}
    unknown_class_files = 0
    for image_path in sorted(root.rglob("*"), key=lambda item: str(item).lower()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        academic_label = _academic_label_from_path(root, image_path)
        if not academic_label:
            unknown_class_files += 1
            continue
        class_counts[academic_label] += 1
        haystack = " ".join(part.lower() for part in image_path.relative_to(root).parts)
        for raw_label, pattern in RAW_SOURCE_LABEL_PATTERNS.items():
            if pattern.search(haystack):
                mentions[academic_label][raw_label] += 1
                if len(examples[academic_label][raw_label]) < max_examples:
                    examples[academic_label][raw_label].append(str(image_path))
    return {
        "dataset_root": str(root),
        "class_counts": class_counts,
        "raw_label_mentions_by_academic_class": mentions,
        "raw_label_examples_by_academic_class": examples,
        "sad_like_source_paths_under_boredom": examples["boredom"]["sad"],
        "sad_like_source_paths_under_frustration": examples["frustration"]["sad"],
        "unknown_class_image_files": unknown_class_files,
        "interpretation": (
            "This scan only uses filenames/path parts. It is evidence only when source raw labels are preserved in paths. "
            "Sad-like paths under boredom support a legacy sad/fear -> boredom training-label hypothesis; "
            "sad-like paths under frustration support the current sad/anger/disgust -> frustration mapping."
        ),
    }


def accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    total = min(len(predictions), len(ground_truth))
    if total <= 0:
        return 0.0
    correct = sum(1 for index in range(total) if predictions[index] == ground_truth[index])
    return correct / total


def prediction_distribution(predictions: list[str]) -> dict[str, int]:
    counts = Counter(predictions)
    return {label: int(counts[label]) for label in sorted(counts)}


def distribution_summary(values: list[str], labels: list[str]) -> dict[str, dict[str, float | int]]:
    counts = Counter(values)
    total = max(1, len(values))
    return {
        label: {
            "count": int(counts.get(label, 0)),
            "fraction": round(float(counts.get(label, 0)) / total, 6),
        }
        for label in labels
    }


def confusion_matrix(ground_truth: list[str], predictions: list[str], labels: list[str]) -> dict[str, dict[str, int]]:
    matrix = {truth: {predicted: 0 for predicted in labels} for truth in labels}
    for truth, predicted in zip(ground_truth, predictions):
        if truth not in matrix:
            matrix[truth] = {label: 0 for label in labels}
        if predicted not in matrix[truth]:
            matrix[truth][predicted] = 0
        matrix[truth][predicted] += 1
    return matrix


def detect_best_label_permutation(
    predictions: list[str],
    ground_truth: list[str],
    labels: list[str],
) -> dict[str, Any]:
    sample_count = min(len(predictions), len(ground_truth))
    direct_accuracy = accuracy(predictions, ground_truth)
    best_accuracy = direct_accuracy
    best_mapping = {label: label for label in labels}
    for permuted in itertools.permutations(labels):
        mapping = {source: target for source, target in zip(labels, permuted)}
        remapped = [mapping.get(prediction, prediction) for prediction in predictions]
        score = accuracy(remapped, ground_truth)
        if score > best_accuracy:
            best_accuracy = score
            best_mapping = mapping
    improvement = best_accuracy - direct_accuracy
    suspected = bool(sample_count >= len(labels) and best_accuracy >= 0.65 and improvement >= 0.20)
    return {
        "sample_count": sample_count,
        "direct_accuracy": round(direct_accuracy, 6),
        "best_accuracy": round(best_accuracy, 6),
        "improvement": round(improvement, 6),
        "best_mapping": best_mapping,
        "suspected_permutation": suspected,
    }


def audit_alignment(
    samples: list[DatasetSample],
    raw_checkpoint: str | Path,
    academic_checkpoint: str | Path,
    max_images: int | None = None,
    device: str = "cpu",
    top_k: int = 25,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_samples = samples[: max_images or None]
    raw_inferencer = RawEmotionInferencer(checkpoint_path=raw_checkpoint, device=device)
    academic_inferencer = RawEmotionInferencer(checkpoint_path=academic_checkpoint, device=device)
    raw_status = raw_inferencer.load()
    academic_status = academic_inferencer.load()
    if raw_status.get("loading_error") or raw_status.get("model_output_type") != "raw_emotion":
        raise RuntimeError(f"Raw 8-class checkpoint could not be loaded as raw_emotion: {raw_status.get('loading_error')}")
    if academic_status.get("loading_error") or academic_status.get("model_output_type") != "academic_state":
        raise RuntimeError(f"4-class checkpoint could not be loaded as academic_state: {academic_status.get('loading_error')}")

    mapper = EmotionMapper()
    examples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for sample in selected_samples:
        try:
            example = _evaluate_sample(sample, raw_inferencer, academic_inferencer, mapper)
            examples.append(example)
        except Exception as exc:
            errors.append({"image_path": str(sample.path), "raw_label": sample.raw_label, "error": str(exc)})

    ground_truth = [item["ground_truth_academic_label"] for item in examples]
    legacy_ground_truth = [item["legacy_archived_ground_truth_academic_label"] for item in examples]
    mapped_predictions = [item["mapped_from_8class_state"] for item in examples]
    direct_predictions = [item["direct_4class_state"] for item in examples]
    disagreement_examples = [
        item
        for item in examples
        if item["direct_4class_state"] != item["mapped_from_8class_state"]
        or item["direct_4class_state"] != item["ground_truth_academic_label"]
    ]
    disagreement_examples.sort(
        key=lambda item: (
            item.get("direct_4class_confidence", 0.0)
            + item.get("mapped_from_8class_confidence", 0.0)
        ),
        reverse=True,
    )

    report = {
        "sample_count_requested": len(selected_samples),
        "sample_count_evaluated": len(examples),
        "sample_count_failed": len(errors),
        "errors": errors[:50],
        "raw_checkpoint": _checkpoint_report(raw_checkpoint, raw_status),
        "academic_checkpoint": _checkpoint_report(academic_checkpoint, academic_status),
        "required_mapping": REQUIRED_RAW_TO_ACADEMIC,
        "legacy_archived_mapping_diagnostic": {
            "mapping": LEGACY_ARCHIVED_RAW_TO_ACADEMIC,
            "source": "docs/archive/README_github_original.md architecture diagram",
            "note": "Diagnostic only; the active requirement maps sad/anger/disgust to frustration.",
        },
        "metrics": {
            "direct_4class_accuracy_against_required_mapped_ground_truth": round(accuracy(direct_predictions, ground_truth), 6),
            "raw8_mapped_accuracy_against_required_mapped_ground_truth": round(accuracy(mapped_predictions, ground_truth), 6),
            "agreement_rate_direct_vs_raw8_mapped": round(accuracy(direct_predictions, mapped_predictions), 6),
            "direct_4class_accuracy_against_legacy_archived_mapping": round(accuracy(direct_predictions, legacy_ground_truth), 6),
            "raw8_mapped_accuracy_against_legacy_archived_mapping": round(accuracy(mapped_predictions, legacy_ground_truth), 6),
        },
        "confusion_matrices": {
            "direct_4class_vs_required_mapped_ground_truth": confusion_matrix(ground_truth, direct_predictions, ACADEMIC_LABELS),
            "raw8_mapped_vs_required_mapped_ground_truth": confusion_matrix(ground_truth, mapped_predictions, ACADEMIC_LABELS),
        },
        "prediction_distributions": {
            "required_mapped_ground_truth": distribution_summary(ground_truth, ACADEMIC_LABELS),
            "legacy_archived_ground_truth": distribution_summary(legacy_ground_truth, ACADEMIC_LABELS),
            "direct_4class_predictions": distribution_summary(direct_predictions, ACADEMIC_LABELS),
            "mapped_raw8_predictions": distribution_summary(mapped_predictions, ACADEMIC_LABELS),
        },
        "label_permutation_check": detect_best_label_permutation(direct_predictions, ground_truth, ACADEMIC_LABELS),
        "class_bias_check": _class_bias_check(direct_predictions, ACADEMIC_LABELS),
        "top_disagreement_examples": disagreement_examples[:top_k],
    }
    return report, disagreement_examples[:top_k]


def save_outputs(
    report: dict[str, Any],
    examples: list[dict[str, Any]],
    output_dir: str | Path = "diagnostics",
) -> tuple[Path, Path]:
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "emotion_model_alignment_report.json"
    csv_path = directory / "emotion_model_alignment_examples.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_examples_csv(csv_path, examples)
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit alignment between raw 8-class emotion and direct 4-class academic-state checkpoints.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root", help="Folder dataset with 8-class subfolders.")
    source.add_argument("--csv", dest="csv_path", help="CSV containing image path and raw 8-class label.")
    source.add_argument("--scan-4class-dataset", help="Scan a 4-class academic-state dataset for raw-label tokens preserved in paths.")
    parser.add_argument("--image-root", help="Root folder for relative image paths in CSV files.")
    parser.add_argument("--raw-checkpoint", default="models/emotion_model/raw_8class_best.pt", help="Raw 8-class checkpoint path.")
    parser.add_argument("--academic-checkpoint", default="models/emotion_model/best_model.pt", help="Direct 4-class checkpoint path.")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum images to evaluate. 0 means all images.")
    parser.add_argument("--top-k", type=int, default=25, help="Number of disagreement examples to save in the report and CSV.")
    parser.add_argument("--output-dir", default="diagnostics", help="Directory for JSON and CSV diagnostic outputs.")
    parser.add_argument("--device", default="cpu", help="Torch device to use. Defaults to cpu.")
    args = parser.parse_args()

    try:
        if args.scan_4class_dataset:
            scan = scan_4class_dataset_for_raw_source_labels(args.scan_4class_dataset)
            output_dir = Path(args.output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "emotion_4class_dataset_scan.json"
            output_path.write_text(json.dumps(scan, indent=2, sort_keys=True), encoding="utf-8")
            print(f"4-class dataset: {scan['dataset_root']}")
            print(f"class counts: {json.dumps(scan['class_counts'], sort_keys=True)}")
            print(
                "sad-like source paths: "
                f"boredom={len(scan['sad_like_source_paths_under_boredom'])}, "
                f"frustration={len(scan['sad_like_source_paths_under_frustration'])}"
            )
            print(f"report: {output_path}")
            return 0
        samples = (
            collect_dataset_root_samples(args.dataset_root)
            if args.dataset_root
            else collect_csv_samples(args.csv_path, image_root=args.image_root)
        )
        report, examples = audit_alignment(
            samples=samples,
            raw_checkpoint=args.raw_checkpoint,
            academic_checkpoint=args.academic_checkpoint,
            max_images=args.max_images or None,
            device=args.device,
            top_k=args.top_k,
        )
        json_path, csv_path = save_outputs(report, examples, args.output_dir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    metrics = report["metrics"]
    print(f"samples evaluated: {report['sample_count_evaluated']} / {report['sample_count_requested']}")
    print(f"direct 4-class accuracy vs required mapped GT: {metrics['direct_4class_accuracy_against_required_mapped_ground_truth']:.6f}")
    print(f"raw 8-class + mapping accuracy vs required mapped GT: {metrics['raw8_mapped_accuracy_against_required_mapped_ground_truth']:.6f}")
    print(f"agreement rate direct vs raw8+mapping: {metrics['agreement_rate_direct_vs_raw8_mapped']:.6f}")
    print(f"direct 4-class accuracy vs archived legacy mapping: {metrics['direct_4class_accuracy_against_legacy_archived_mapping']:.6f}")
    permutation = report["label_permutation_check"]
    print(f"permutation suspected: {permutation['suspected_permutation']} (best accuracy {permutation['best_accuracy']:.6f})")
    print(f"report: {json_path}")
    print(f"examples: {csv_path}")
    return 0


def _evaluate_sample(
    sample: DatasetSample,
    raw_inferencer: RawEmotionInferencer,
    academic_inferencer: RawEmotionInferencer,
    mapper: EmotionMapper,
) -> dict[str, Any]:
    from PIL import Image  # type: ignore

    image = Image.open(sample.path).convert("RGB")
    raw_prediction = _predict(raw_inferencer, image)
    academic_prediction = _predict(academic_inferencer, image)
    raw_probs = _float_probabilities(raw_prediction.get("probabilities") or {})
    direct_probs = _float_probabilities(academic_prediction.get("probabilities") or {})
    raw_predicted_label = max(raw_probs, key=raw_probs.get)
    direct_state = max(direct_probs, key=direct_probs.get)
    mapped_state, mapped_scores = mapper.map_probs_to_state(raw_probs)
    mapped_scores = {label: round(float(mapped_scores.get(label, 0.0)), 6) for label in ("frustration", "confusion", "boredom", "engagement")}
    return {
        "image_path": str(sample.path),
        "ground_truth_raw_label": sample.raw_label,
        "ground_truth_academic_label": map_raw_label_to_academic(sample.raw_label),
        "legacy_archived_ground_truth_academic_label": map_raw_label_to_legacy_archived_academic(sample.raw_label),
        "raw_8class_predicted_label": raw_predicted_label,
        "raw_8class_confidence": round(float(raw_probs.get(raw_predicted_label, 0.0)), 6),
        "raw_8class_probabilities": _rounded_probabilities(raw_probs),
        "mapped_from_8class_state": mapped_state,
        "mapped_from_8class_confidence": round(float(mapped_scores.get(mapped_state, 0.0)), 6),
        "mapped_4class_scores_from_8class": mapped_scores,
        "direct_4class_state": direct_state,
        "direct_4class_confidence": round(float(direct_probs.get(direct_state, 0.0)), 6),
        "direct_4class_probabilities": _rounded_probabilities(direct_probs),
    }


def _predict(inferencer: RawEmotionInferencer, image: Any) -> dict[str, Any]:
    prediction = inferencer.predict_probabilities(image)
    if prediction.get("error"):
        raise RuntimeError(str(prediction["error"]))
    probabilities = prediction.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        raise RuntimeError("Model did not return probabilities.")
    return prediction


def _checkpoint_report(path: str | Path, status: dict[str, Any]) -> dict[str, Any]:
    try:
        inspection = inspect_checkpoint_file(path)
    except Exception as exc:
        inspection = {"inspection_error": str(exc)}
    return {
        "path": str(Path(path).expanduser()),
        "status": {
            "model_output_type": status.get("model_output_type"),
            "classes": list(status.get("classes") or []),
            "architecture": status.get("architecture"),
            "preprocessing_summary": status.get("preprocessing_summary") or {},
        },
        "inspection": inspection,
    }


def _class_bias_check(predictions: list[str], labels: list[str]) -> dict[str, Any]:
    summary = distribution_summary(predictions, labels)
    if not predictions:
        return {"dominant_label": "", "dominant_fraction": 0.0, "prediction_distribution": summary}
    dominant_label = max(labels, key=lambda label: int(summary[label]["count"]))
    dominant_fraction = float(summary[dominant_label]["fraction"])
    return {
        "dominant_label": dominant_label,
        "dominant_fraction": dominant_fraction,
        "possible_bias": dominant_fraction >= 0.60,
        "prediction_distribution": summary,
    }


def _write_examples_csv(path: Path, examples: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image_path",
        "ground_truth_raw_label",
        "ground_truth_academic_label",
        "legacy_archived_ground_truth_academic_label",
        "raw_8class_predicted_label",
        "mapped_from_8class_state",
        "direct_4class_state",
        "raw_8class_confidence",
        "mapped_from_8class_confidence",
        "direct_4class_confidence",
        "raw_8class_probabilities",
        "mapped_4class_scores_from_8class",
        "direct_4class_probabilities",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for example in examples:
            row = dict(example)
            for key in ("raw_8class_probabilities", "mapped_4class_scores_from_8class", "direct_4class_probabilities"):
                row[key] = json.dumps(row.get(key) or {}, sort_keys=True)
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _first_column_index(header: list[str], candidates: tuple[str, ...]) -> int | None:
    for candidate in candidates:
        if candidate in header:
            return header.index(candidate)
    return None


def _academic_label_from_path(root: Path, image_path: Path) -> str:
    try:
        parts = [part.strip().lower() for part in image_path.relative_to(root).parts]
    except Exception:
        parts = [part.strip().lower() for part in image_path.parts]
    for part in parts:
        if part in ACADEMIC_LABELS:
            return part
    return ""


def _float_probabilities(probabilities: dict[str, Any]) -> dict[str, float]:
    return {str(label): float(value) for label, value in probabilities.items()}


def _rounded_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    return {label: round(float(value), 6) for label, value in sorted(probabilities.items())}


if __name__ == "__main__":
    raise SystemExit(main())
