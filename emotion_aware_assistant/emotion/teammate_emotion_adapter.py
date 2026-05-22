from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .labels import ALLOWED_EMOTIONS, EMOTION_TO_STATE


ACADEMIC_STATE_CLASSES = ["boredom", "confusion", "engagement", "frustration"]
RAW_EMOTION_CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DEFAULT_ARCHITECTURE = "convnext_tiny.fb_in22k_ft_in1k"
DEFAULT_METADATA: dict[str, Any] = {
    "model_output_type": "academic_state",
    "architecture": DEFAULT_ARCHITECTURE,
    "framework": "timm",
    "num_classes": 4,
    "classes": ACADEMIC_STATE_CLASSES,
    "class_to_idx": {
        "boredom": 0,
        "confusion": 1,
        "engagement": 2,
        "frustration": 3,
    },
    "input_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "checkpoint_key": "model_state_dict",
    "academic_label_order_source": "hardcoded",
}


class TeammateEmotionAdapter:
    """Inference-only adapter for teammate-provided emotion or academic-state checkpoints.

    The current CS731 checkpoint is Mode B: a 4-class academic-state model. Mode A
    raw facial-emotion checkpoints are kept schema-compatible for later replacement.
    """

    def __init__(
        self,
        model_dir: str | Path = "models/emotion_model",
        checkpoint_name: str = "best_model.pt",
        metadata_name: str = "metadata.json",
    ):
        self.model_dir = Path(model_dir)
        self.checkpoint_name = checkpoint_name
        self.metadata_name = metadata_name
        self.metadata: dict[str, Any] | None = None
        self.model = None
        self.device = "cpu"
        self._loaded = False
        self.load_error: str | None = None
        self.load_warnings: list[str] = []
        self.missing_keys: list[str] = []
        self.unexpected_keys: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / self.metadata_name

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / self.checkpoint_name

    @property
    def resolved_metadata(self) -> dict[str, Any]:
        if self.metadata is None:
            self.metadata = self._load_metadata()
        return self.metadata

    @property
    def model_output_type(self) -> str:
        return self._infer_model_output_type(self.resolved_metadata)

    @property
    def classes(self) -> list[str]:
        metadata = self.resolved_metadata
        classes = metadata.get("classes") or metadata.get("label_order") or []
        if isinstance(classes, list) and classes:
            if self.model_output_type == "raw_emotion":
                return [_canonical_raw_label(item) for item in classes]
            return [str(item) for item in classes]
        return ACADEMIC_STATE_CLASSES if self.model_output_type == "academic_state" else RAW_EMOTION_CLASSES

    def status(self) -> dict[str, Any]:
        metadata = self.resolved_metadata
        checkpoint_path = self._find_weight_path()
        if checkpoint_path is None and not self.load_error:
            self.load_error = "models/emotion_model/best_model.pt is missing."
        return {
            "model_loaded": self._loaded,
            "model_output_type": self.model_output_type,
            "architecture": str(metadata.get("architecture") or DEFAULT_ARCHITECTURE),
            "classes": self.classes,
            "checkpoint_path": self._safe_checkpoint_label(checkpoint_path or self.checkpoint_path),
            "raw_emotion_available": self.model_output_type == "raw_emotion",
            "academic_label_order_source": str(metadata.get("academic_label_order_source") or "unavailable"),
            "input_size": int(metadata.get("input_size") or 224),
            "mean": self._float_list(metadata.get("mean"), [0.485, 0.456, 0.406]),
            "std": self._float_list(metadata.get("std"), [0.229, 0.224, 0.225]),
            "preprocessing_summary": self._preprocessing_summary(metadata),
            "loading_error": self.load_error,
            "loading_warnings": list(self.load_warnings),
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "device": self.device,
        }

    def load(self) -> dict[str, Any]:
        if self._loaded:
            return self.status()
        metadata = self.resolved_metadata
        checkpoint_path = self._find_weight_path()
        if checkpoint_path is None:
            self.load_error = "models/emotion_model/best_model.pt is missing."
            return self.status()
        try:
            import timm  # type: ignore
            import torch  # type: ignore
        except Exception as exc:
            self.load_error = f"PyTorch/timm dependencies are unavailable: {exc}"
            return self.status()

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            metadata = self._metadata_with_checkpoint(metadata, checkpoint)
            self.metadata = metadata
            architecture = str(metadata.get("architecture") or DEFAULT_ARCHITECTURE)
            num_classes = int(metadata.get("num_classes") or len(self.classes) or 4)
            self.model = timm.create_model(architecture, pretrained=False, num_classes=num_classes)
            state_dict = self._extract_state_dict(checkpoint, metadata)
            state_dict = self._clean_state_dict_keys(state_dict)
            try:
                incompatible = self.model.load_state_dict(state_dict, strict=True)
            except Exception as exc:
                self.load_warnings.append(f"Strict state_dict load failed; retried with strict=False: {exc}")
                incompatible = self.model.load_state_dict(state_dict, strict=False)
            self.missing_keys = [str(key) for key in getattr(incompatible, "missing_keys", [])]
            self.unexpected_keys = [str(key) for key in getattr(incompatible, "unexpected_keys", [])]
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            self.load_error = None
        except Exception as exc:
            self.model = None
            self._loaded = False
            self.load_error = f"Could not load teammate checkpoint: {exc}"
        return self.status()

    def predict(self, image: Any, input_color: str = "rgb") -> dict[str, Any]:
        if not self._loaded:
            self.load()
        if not self._loaded or self.model is None:
            return {
                **self.status(),
                "error": self.load_error or "Emotion model is not loaded.",
            }
        try:
            import numpy as np  # type: ignore
            import torch  # type: ignore
            from PIL import Image  # type: ignore

            pil_image = self._to_pil_image(image, Image, np, input_color=input_color)
            size = int(self.resolved_metadata.get("input_size") or 224)
            mean = self._float_list(self.resolved_metadata.get("mean"), [0.485, 0.456, 0.406])
            std = self._float_list(self.resolved_metadata.get("std"), [0.229, 0.224, 0.225])
            resized = pil_image.resize((size, size))
            tensor = torch.tensor(np.asarray(resized), dtype=torch.float32).permute(2, 0, 1) / 255.0
            for channel in range(3):
                tensor[channel] = (tensor[channel] - float(mean[channel])) / float(std[channel])
            tensor = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(tensor), dim=1)[0].detach().cpu().numpy()
            probabilities = {label: float(probs[index]) for index, label in enumerate(self.classes)}
            if self.model_output_type == "raw_emotion":
                payload = self.raw_emotion_prediction_payload(
                    probabilities=probabilities,
                    architecture=str(self.resolved_metadata.get("architecture") or DEFAULT_ARCHITECTURE),
                    classes=self.classes,
                    device=self.device,
                )
                payload["preprocessing_summary"] = self._preprocessing_summary(self.resolved_metadata)
                return payload
            payload = self.academic_prediction_payload(
                probabilities=probabilities,
                architecture=str(self.resolved_metadata.get("architecture") or DEFAULT_ARCHITECTURE),
                classes=self.classes,
                device=self.device,
            )
            payload["academic_label_order_source"] = str(self.resolved_metadata.get("academic_label_order_source") or "unavailable")
            payload["preprocessing_summary"] = self._preprocessing_summary(self.resolved_metadata)
            return payload
        except Exception as exc:
            return {
                **self.status(),
                "error": f"Emotion model prediction failed: {exc}",
            }

    @staticmethod
    def academic_prediction_payload(
        probabilities: dict[str, float],
        architecture: str,
        classes: list[str],
        device: str,
    ) -> dict[str, Any]:
        distribution = _normalized_distribution(probabilities, ACADEMIC_STATE_CLASSES)
        academic_state = max(distribution, key=distribution.get)
        confidence = float(distribution[academic_state])
        return {
            "model_loaded": True,
            "model_output_type": "academic_state",
            "raw_emotion_available": False,
            "raw_emotion": None,
            "academic_state": academic_state,
            "confidence": confidence,
            "state_distribution": distribution,
            "architecture": architecture,
            "classes": classes,
            "device": device,
            "note": "This checkpoint predicts learning-centered academic states directly.",
        }

    @staticmethod
    def raw_emotion_prediction_payload(
        probabilities: dict[str, float],
        architecture: str,
        classes: list[str],
        device: str,
    ) -> dict[str, Any]:
        raw_distribution = _normalized_distribution(probabilities, RAW_EMOTION_CLASSES, raw=True)
        raw_emotion = max(raw_distribution, key=raw_distribution.get)
        state_distribution = {state: 0.0 for state in ACADEMIC_STATE_CLASSES}
        for emotion, probability in raw_distribution.items():
            state = EMOTION_TO_STATE.get(emotion)
            if state in state_distribution:
                state_distribution[state] += probability
        state_distribution = _normalized_distribution(state_distribution, ACADEMIC_STATE_CLASSES)
        academic_state = max(state_distribution, key=state_distribution.get)
        return {
            "model_loaded": True,
            "model_output_type": "raw_emotion",
            "raw_emotion_available": True,
            "raw_emotion": raw_emotion,
            "confidence": float(raw_distribution[raw_emotion]),
            "raw_distribution": raw_distribution,
            "academic_state": academic_state,
            "state_distribution": state_distribution,
            "architecture": architecture,
            "classes": [_canonical_raw_label(label) for label in classes],
            "device": device,
        }

    def _load_metadata(self) -> dict[str, Any]:
        metadata = dict(DEFAULT_METADATA)
        try:
            if self.metadata_path.exists():
                loaded = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    metadata.update(loaded)
                    if loaded.get("classes") or loaded.get("label_order") or loaded.get("class_to_idx"):
                        metadata["academic_label_order_source"] = "checkpoint_metadata"
        except Exception as exc:
            self.load_error = f"metadata.json could not be read: {exc}"
        metadata["model_output_type"] = self._infer_model_output_type(metadata)
        if metadata["model_output_type"] == "academic_state":
            metadata["classes"] = list(metadata.get("classes") or ACADEMIC_STATE_CLASSES)
            metadata["num_classes"] = int(metadata.get("num_classes") or 4)
            metadata["raw_emotion_available"] = False
            metadata["academic_label_order_source"] = str(metadata.get("academic_label_order_source") or "hardcoded")
        else:
            metadata["classes"] = [_canonical_raw_label(label) for label in list(metadata.get("classes") or metadata.get("label_order") or RAW_EMOTION_CLASSES)]
            metadata["num_classes"] = int(metadata.get("num_classes") or len(metadata["classes"]))
            metadata["raw_emotion_available"] = True
        return metadata

    @staticmethod
    def _metadata_with_checkpoint(metadata: dict[str, Any], checkpoint: Any) -> dict[str, Any]:
        merged = dict(metadata)
        if not isinstance(checkpoint, dict):
            return merged
        architecture = str(checkpoint.get("arch") or checkpoint.get("architecture") or "").strip()
        if architecture:
            merged["architecture"] = architecture
        if checkpoint.get("num_classes") is not None:
            try:
                merged["num_classes"] = int(checkpoint.get("num_classes"))
            except Exception:
                pass
        classes, source = _classes_from_checkpoint_metadata(checkpoint)
        if classes:
            merged["classes"] = classes
            merged["class_to_idx"] = {
                label: index
                for index, label in enumerate(classes)
            }
            merged["model_output_type"] = TeammateEmotionAdapter._infer_model_output_type(merged)
            if merged["model_output_type"] == "raw_emotion":
                merged["classes"] = [_canonical_raw_label(label) for label in classes]
            else:
                merged["academic_label_order_source"] = source
        elif int(merged.get("num_classes") or 0) == 4 and not merged.get("classes"):
            merged["classes"] = list(ACADEMIC_STATE_CLASSES)
            merged["academic_label_order_source"] = "guessed"
        return merged

    def _find_weight_path(self) -> Path | None:
        candidates = [self.model_dir / name for name in ("best_model.pt", "best_model.pth", "best_model.ckpt")]
        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    @staticmethod
    def _infer_model_output_type(metadata: dict[str, Any]) -> str:
        explicit = str(metadata.get("model_output_type") or "").strip().lower()
        if explicit in {"academic_state", "academic_state_model"}:
            return "academic_state"
        if explicit in {"raw_emotion", "raw_facial_emotion"}:
            return "raw_emotion"
        classes = [str(item).strip().lower() for item in metadata.get("classes") or metadata.get("label_order") or []]
        if len(classes) == len(ACADEMIC_STATE_CLASSES) and set(classes) == set(ACADEMIC_STATE_CLASSES):
            return "academic_state"
        canonical_raw = [_canonical_raw_label(item) for item in classes]
        if len(canonical_raw) == len(RAW_EMOTION_CLASSES) and set(canonical_raw) == set(RAW_EMOTION_CLASSES):
            return "raw_emotion"
        allowed_canonical = [_canonical_raw_label(item) for item in ALLOWED_EMOTIONS]
        if len(allowed_canonical) == len(canonical_raw) and set(canonical_raw) == set(allowed_canonical):
            return "raw_emotion"
        return "academic_state"

    @staticmethod
    def _extract_state_dict(checkpoint: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(checkpoint, dict):
            return checkpoint
        checkpoint_key = str(metadata.get("checkpoint_key") or "")
        keys = [checkpoint_key, "model_state_dict", "state_dict", "model", "net"]
        for key in keys:
            if key and isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        return checkpoint

    @staticmethod
    def _clean_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
        cleaned = {}
        for key, value in state_dict.items():
            key_text = str(key)
            changed = True
            while changed:
                changed = False
                for prefix in ("module.", "model.", "net."):
                    if key_text.startswith(prefix):
                        key_text = key_text.removeprefix(prefix)
                        changed = True
            cleaned[key_text] = value
        return cleaned

    @staticmethod
    def _to_pil_image(image: Any, Image: Any, np: Any, input_color: str = "rgb") -> Any:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("Expected an RGB or BGR image array.")
        arr = arr[:, :, :3]
        if input_color.lower() == "bgr":
            arr = arr[:, :, ::-1]
        arr = arr.astype("uint8")
        return Image.fromarray(arr).convert("RGB")

    @staticmethod
    def _float_list(value: Any, fallback: list[float]) -> list[float]:
        if not isinstance(value, list) or len(value) < 3:
            return fallback
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except Exception:
            return fallback

    @classmethod
    def _preprocessing_summary(cls, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_size": int(metadata.get("input_size") or 224),
            "mean": cls._float_list(metadata.get("mean"), [0.485, 0.456, 0.406]),
            "std": cls._float_list(metadata.get("std"), [0.229, 0.224, 0.225]),
            "color_space": "RGB",
            "tensor_layout": "NCHW",
            "resize_method": "PIL.Image.resize default",
        }

    @staticmethod
    def _safe_checkpoint_label(path: Path) -> str:
        parts = path.parts
        if "models" in parts:
            return str(Path(*parts[parts.index("models") :]))
        return str(Path("models/emotion_model") / path.name)


def _normalized_distribution(probabilities: dict[str, float], labels: list[str], raw: bool = False) -> dict[str, float]:
    source: dict[str, float] = {}
    for key, value in probabilities.items():
        label = _canonical_raw_label(key) if raw else str(key).strip().lower()
        try:
            source[label] = source.get(label, 0.0) + max(0.0, float(value))
        except Exception:
            source.setdefault(label, 0.0)
    values: dict[str, float] = {}
    for label in labels:
        key = _canonical_raw_label(label) if raw else str(label).strip().lower()
        try:
            values[key] = max(0.0, float(source.get(key, 0.0)))
        except Exception:
            values[key] = 0.0
    total = sum(values.values())
    if total <= 0:
        return {label: round(1.0 / len(labels), 6) for label in labels}
    return {_canonical_raw_label(label) if raw else str(label).strip().lower(): round(values[_canonical_raw_label(label) if raw else str(label).strip().lower()] / total, 6) for label in labels}


def _canonical_raw_label(label: Any) -> str:
    text = str(label or "").strip().lower()
    aliases = {"angry": "anger", "happiness": "happy", "sadness": "sad"}
    return aliases.get(text, text)


def _classes_from_checkpoint_metadata(checkpoint: dict[str, Any]) -> tuple[list[str], str]:
    class_to_idx = checkpoint.get("class_to_idx")
    if isinstance(class_to_idx, dict) and class_to_idx:
        try:
            return [
                str(label).strip().lower()
                for label, _ in sorted(class_to_idx.items(), key=lambda item: int(item[1]))
            ], "checkpoint_metadata"
        except Exception:
            return [str(label).strip().lower() for label in class_to_idx], "checkpoint_metadata"
    classes = checkpoint.get("classes") or checkpoint.get("label_order")
    if isinstance(classes, list) and classes:
        return [str(item).strip().lower() for item in classes], "checkpoint_metadata"
    return [], "unavailable"
