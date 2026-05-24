from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emotion_aware_assistant.emotion.raw_emotion_pipeline import EmotionMapper, RawEmotionInferencer
from scripts.audit_emotion_model_alignment import canonical_raw_label, map_raw_label_to_academic


def image_file_sha256(path: str | Path) -> str:
    image_path = Path(path).expanduser()
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


def preprocessing_summary(status: dict[str, Any]) -> dict[str, Any]:
    summary = status.get("preprocessing_summary") if isinstance(status.get("preprocessing_summary"), dict) else {}
    return {
        "input_size": int(summary.get("input_size") or status.get("input_size") or 224),
        "mean": _float_list(summary.get("mean") or status.get("mean"), [0.485, 0.456, 0.406]),
        "std": _float_list(summary.get("std") or status.get("std"), [0.229, 0.224, 0.225]),
        "color_space": str(summary.get("color_space") or "RGB"),
        "tensor_layout": str(summary.get("tensor_layout") or "NCHW"),
        "resize_method": str(summary.get("resize_method") or "PIL.Image.resize default"),
    }


def compare_models_on_crop(
    image_path: str | Path,
    raw_checkpoint: str | Path,
    academic_checkpoint: str | Path,
    device: str = "cpu",
    raw_label: str | None = None,
) -> dict[str, Any]:
    from PIL import Image  # type: ignore

    crop_path = Path(image_path).expanduser()
    image = Image.open(crop_path).convert("RGB")
    raw_prediction = _predict_checkpoint(raw_checkpoint, image, device=device)
    academic_prediction = _predict_checkpoint(academic_checkpoint, image, device=device)
    mapper = EmotionMapper()
    mapped_state, mapped_scores = mapper.map_probs_to_state(raw_prediction["probabilities"])
    direct_probs = academic_prediction["probabilities"]
    direct_state = max(direct_probs, key=direct_probs.get) if direct_probs else ""
    result = {
        "crop_path": str(crop_path),
        "crop_hash": image_file_sha256(crop_path),
        "raw_checkpoint_path": str(Path(raw_checkpoint).expanduser()),
        "raw_checkpoint_classes": list(raw_prediction.get("classes") or []),
        "raw_preprocessing_summary": raw_prediction["preprocessing_summary"],
        "raw_8class_probabilities": raw_prediction["probabilities"],
        "raw_8class_state": max(raw_prediction["probabilities"], key=raw_prediction["probabilities"].get),
        "mapped_from_8class_state": mapped_state,
        "mapped_4class_scores_from_8class": {
            state: round(float(mapped_scores.get(state, 0.0)), 6)
            for state in ("frustration", "confusion", "boredom", "engagement")
        },
        "academic_checkpoint_path": str(Path(academic_checkpoint).expanduser()),
        "direct_4class_checkpoint_classes": list(academic_prediction.get("classes") or []),
        "academic_preprocessing_summary": academic_prediction["preprocessing_summary"],
        "direct_4class_probabilities": direct_probs,
        "direct_4class_state": direct_state,
        "direct_4class_confidence": float(direct_probs.get(direct_state, 0.0)) if direct_state else 0.0,
    }
    if raw_label:
        result = add_ground_truth_fields(result, raw_label)
    return result


def add_ground_truth_fields(result: dict[str, Any], raw_label: str) -> dict[str, Any]:
    enriched = dict(result)
    canonical = canonical_raw_label(raw_label)
    mapped_ground_truth = map_raw_label_to_academic(canonical)
    direct_state = str(enriched.get("direct_4class_state") or "")
    mapped_state = str(enriched.get("mapped_from_8class_state") or "")
    enriched.update(
        {
            "ground_truth_raw_label": canonical,
            "ground_truth_mapped_label": mapped_ground_truth,
            "direct_4class_matches_ground_truth": direct_state == mapped_ground_truth,
            "raw8_mapped_matches_ground_truth": mapped_state == mapped_ground_truth,
            "direct_4class_agrees_with_raw8_mapped": bool(direct_state and mapped_state and direct_state == mapped_state),
        }
    )
    return enriched


def main() -> int:
    parser = argparse.ArgumentParser(description="Run raw 8-class and direct 4-class emotion models on the same saved crop.")
    parser.add_argument("--image", required=True, help="Path to a saved face crop image.")
    parser.add_argument("--raw-checkpoint", required=True, help="Path to the raw 8-class checkpoint.")
    parser.add_argument("--academic-checkpoint", required=True, help="Path to the direct 4-class academic-state checkpoint.")
    parser.add_argument("--raw-label", help="Optional ground-truth raw 8-class label for this image.")
    parser.add_argument("--device", default="cpu", help="Torch device to use. Defaults to cpu.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    args = parser.parse_args()
    try:
        result = compare_models_on_crop(
            image_path=args.image,
            raw_checkpoint=args.raw_checkpoint,
            academic_checkpoint=args.academic_checkpoint,
            device=args.device,
            raw_label=args.raw_label,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(f"crop path: {result['crop_path']}")
    print(f"crop hash: {result['crop_hash']}")
    print(f"raw checkpoint path: {result['raw_checkpoint_path']}")
    print(f"raw checkpoint classes: {result['raw_checkpoint_classes']}")
    print(f"raw preprocessing: {json.dumps(result['raw_preprocessing_summary'], sort_keys=True)}")
    print(f"raw 8-class probabilities: {json.dumps(result['raw_8class_probabilities'], indent=2, sort_keys=True)}")
    print(f"mapped 4-class scores from 8-class: {json.dumps(result['mapped_4class_scores_from_8class'], indent=2, sort_keys=True)}")
    print(f"mapped state from 8-class: {result['mapped_from_8class_state']}")
    if result.get("ground_truth_raw_label"):
        print(f"ground-truth raw label: {result['ground_truth_raw_label']}")
        print(f"ground-truth mapped label: {result['ground_truth_mapped_label']}")
        print(f"raw8 mapped matches ground truth: {result['raw8_mapped_matches_ground_truth']}")
    print(f"academic checkpoint path: {result['academic_checkpoint_path']}")
    print(f"direct 4-class checkpoint classes: {result['direct_4class_checkpoint_classes']}")
    print(f"academic preprocessing: {json.dumps(result['academic_preprocessing_summary'], sort_keys=True)}")
    print(f"direct 4-class probabilities: {json.dumps(result['direct_4class_probabilities'], indent=2, sort_keys=True)}")
    print(f"direct 4-class state: {result['direct_4class_state']}")
    print(f"direct 4-class confidence: {result['direct_4class_confidence']:.6f}")
    if result.get("ground_truth_raw_label"):
        print(f"direct 4-class matches ground truth: {result['direct_4class_matches_ground_truth']}")
        print(f"direct 4-class agrees with raw8 mapped prediction: {result['direct_4class_agrees_with_raw8_mapped']}")
    return 0


def _predict_checkpoint(checkpoint_path: str | Path, image: Any, device: str) -> dict[str, Any]:
    inferencer = RawEmotionInferencer(checkpoint_path=checkpoint_path, device=device)
    prediction = inferencer.predict_probabilities(image)
    if prediction.get("error"):
        raise RuntimeError(str(prediction["error"]))
    probabilities = prediction.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        raise RuntimeError(f"No probabilities returned for checkpoint: {checkpoint_path}")
    return {
        "probabilities": {str(label): float(value) for label, value in probabilities.items()},
        "classes": list(prediction.get("classes") or []),
        "preprocessing_summary": preprocessing_summary(prediction),
    }


def _float_list(value: Any, fallback: list[float]) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return list(fallback)
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return list(fallback)


if __name__ == "__main__":
    raise SystemExit(main())
