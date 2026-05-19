from __future__ import annotations

import os
from collections import Counter, deque
from pathlib import Path
from typing import Any

from emotion_aware_assistant.core.config import PROJECT_ROOT


RAW_EMOTION_CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
ACADEMIC_STATE_CLASSES = ["boredom", "confusion", "engagement", "frustration"]
MAPPED_STATE_CLASSES = ["frustration", "confusion", "boredom", "engagement"]
DEFAULT_RAW_ARCHITECTURE = "convnextv2_pico.fcmae_ft_in1k"
DEFAULT_RAW_CHECKPOINT_CANDIDATE = "models/emotion_model/raw_8class_best.pt"
DEFAULT_ACADEMIC_CHECKPOINT_CANDIDATE = "models/emotion_model/best_model.pt"
DEFAULT_CHECKPOINT_CANDIDATES = [
    DEFAULT_RAW_CHECKPOINT_CANDIDATE,
    DEFAULT_ACADEMIC_CHECKPOINT_CANDIDATE,
    "models/emotion_model/convnext_tiny_best.pt",
    "convnext_tiny/best.pt",
    "/home/rli/下载/best",
    "/home/rli/下载/convnext_tiny/best.pt",
]


def inspect_checkpoint_metadata(checkpoint: Any) -> dict[str, Any]:
    classes = _classes_from_checkpoint(checkpoint)
    model_output_type = _detect_model_output_type(classes, checkpoint)
    if model_output_type == "raw_emotion":
        classes = [_canonical_raw_label(label) for label in classes]
    architecture = ""
    num_classes = len(classes)
    if isinstance(checkpoint, dict):
        architecture = str(checkpoint.get("arch") or checkpoint.get("architecture") or "").strip()
        try:
            num_classes = int(checkpoint.get("num_classes") or num_classes)
        except Exception:
            num_classes = len(classes)
    if not architecture:
        architecture = DEFAULT_RAW_ARCHITECTURE
    return {
        "model_output_type": model_output_type,
        "raw_detection_available": model_output_type == "raw_emotion",
        "architecture": architecture,
        "classes": classes,
        "num_classes": num_classes,
    }


def load_checkpoint_for_inspection(path: str | Path) -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"PyTorch is required to inspect emotion checkpoints: {exc}") from exc
    checkpoint_path = Path(path).expanduser()
    return _safe_torch_load(torch, checkpoint_path)


def inspect_checkpoint_file(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    checkpoint = load_checkpoint_for_inspection(checkpoint_path)
    info = inspect_checkpoint_metadata(checkpoint)
    state_dict = RawEmotionInferencer._extract_state_dict(checkpoint)
    state_dict_present = isinstance(state_dict, dict) and (
        bool(state_dict)
        or (isinstance(checkpoint, dict) and any(isinstance(checkpoint.get(key), dict) for key in ("model_state_dict", "state_dict", "model", "net")))
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "arch": info.get("architecture") or "",
        "architecture": info.get("architecture") or "",
        "num_classes": info.get("num_classes"),
        "classes": list(info.get("classes") or []),
        "detected_model_mode": info.get("model_output_type") or "unknown",
        "model_output_type": info.get("model_output_type") or "unknown",
        "raw_detection_available": bool(info.get("raw_detection_available")),
        "model_state_dict_present": state_dict_present,
        "sample_keys": [str(key) for key in list(state_dict.keys())[:12]] if isinstance(state_dict, dict) else [],
        "checkpoint_keys": [str(key) for key in checkpoint.keys()] if isinstance(checkpoint, dict) else [],
    }


def select_emotion_checkpoint(
    checkpoint_path: str | Path | None = None,
    mode: str | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser() if project_root else PROJECT_ROOT
    requested_mode = _normalize_model_mode(mode or os.environ.get("EMOTION_MODEL_MODE") or "auto")
    warnings: list[str] = []

    candidates = _checkpoint_selection_candidates(checkpoint_path, requested_mode, root)
    explicit_sources = {"constructor", "RAW_EMOTION_CHECKPOINT_PATH", "EMOTION_CHECKPOINT_PATH"}
    explicit_candidates = [candidate for candidate in candidates if candidate.get("source") in explicit_sources]
    if explicit_candidates and not any(_resolve_candidate_path(candidate["path"], root).exists() for candidate in explicit_candidates):
        missing_path = _resolve_candidate_path(explicit_candidates[0]["path"], root)
        return {
            "checkpoint_path": str(missing_path),
            "arch": "",
            "architecture": "",
            "num_classes": 0,
            "classes": [],
            "detected_model_mode": "unknown",
            "model_output_type": "unknown",
            "raw_detection_available": False,
            "model_state_dict_present": False,
            "sample_keys": [],
            "checkpoint_keys": [],
            "source": explicit_candidates[0]["source"],
            "requested_model_mode": requested_mode,
            "warnings": warnings,
            "loading_error": f"Emotion checkpoint is missing: {missing_path}",
        }
    first_existing_unknown: dict[str, Any] | None = None
    for candidate in candidates:
        path = _resolve_candidate_path(candidate["path"], root)
        if not path.exists():
            continue
        try:
            info = inspect_checkpoint_file(path)
        except Exception as exc:
            info = {
                "checkpoint_path": str(path),
                "arch": "",
                "architecture": "",
                "num_classes": 0,
                "classes": [],
                "detected_model_mode": "unknown",
                "model_output_type": "unknown",
                "raw_detection_available": False,
                "model_state_dict_present": False,
                "sample_keys": [],
                "checkpoint_keys": [],
                "loading_error": str(exc),
            }
        model_output_type = str(info.get("model_output_type") or info.get("detected_model_mode") or "unknown")
        result = {
            **info,
            "checkpoint_path": str(path),
            "source": candidate["source"],
            "requested_model_mode": requested_mode,
            "warnings": list(warnings),
        }
        if requested_mode == "raw_emotion":
            if model_output_type != "raw_emotion":
                result["warnings"] = [*result["warnings"], "Selected checkpoint does not expose the 8-class raw-emotion labels."]
            return result
        if requested_mode == "academic_state":
            if model_output_type != "academic_state":
                result["warnings"] = [*result["warnings"], "Selected checkpoint does not expose the 4-class academic-state labels."]
            return result
        if candidate.get("requires_raw") and model_output_type != "raw_emotion":
            warnings.append(f"Skipped {path}: not an 8-class raw-emotion checkpoint.")
            if first_existing_unknown is None:
                first_existing_unknown = result
            continue
        if model_output_type in {"raw_emotion", "academic_state"}:
            return result
        if first_existing_unknown is None:
            first_existing_unknown = result
    if first_existing_unknown is not None:
        first_existing_unknown["warnings"] = [*first_existing_unknown.get("warnings", []), *warnings]
        return first_existing_unknown
    return {
        "checkpoint_path": "",
        "arch": "",
        "architecture": "",
        "num_classes": 0,
        "classes": [],
        "detected_model_mode": "unknown",
        "model_output_type": "unknown",
        "raw_detection_available": False,
        "model_state_dict_present": False,
        "sample_keys": [],
        "checkpoint_keys": [],
        "source": "",
        "requested_model_mode": requested_mode,
        "warnings": warnings,
        "loading_error": "Emotion checkpoint is missing. Configure RAW_EMOTION_CHECKPOINT_PATH or EMOTION_CHECKPOINT_PATH.",
    }


class EmotionMapper:
    def map_probs_to_scores(self, probs: dict[str, float]) -> dict[str, float]:
        normalized = {_canonical_raw_label(key): _safe_float(value) for key, value in probs.items()}
        return {
            "frustration": normalized.get("sad", 0.0) + normalized.get("anger", 0.0) + normalized.get("disgust", 0.0),
            "confusion": normalized.get("fear", 0.0) + normalized.get("surprise", 0.0),
            "boredom": normalized.get("contempt", 0.0),
            "engagement": normalized.get("happy", 0.0) + normalized.get("neutral", 0.0),
        }

    def map_probs_to_state(self, probs: dict[str, float]) -> tuple[str, dict[str, float]]:
        scores = self.map_probs_to_scores(probs)
        return max(scores, key=scores.get), scores

    def map_label_to_state(self, label: str) -> str:
        label = _canonical_raw_label(label)
        if label in {"sad", "anger", "disgust"}:
            return "frustration"
        if label in {"fear", "surprise"}:
            return "confusion"
        if label == "contempt":
            return "boredom"
        if label in {"happy", "neutral"}:
            return "engagement"
        return "engagement"

    def mapping_rule_for_state(self, state: str) -> str:
        rules = {
            "frustration": "sad + anger + disgust -> frustration",
            "confusion": "fear + surprise -> confusion",
            "boredom": "contempt -> boredom",
            "engagement": "happy + neutral -> engagement",
        }
        return rules.get(str(state).strip().lower(), "unknown raw-emotion mapping")

    def state_to_response_strategy(self, state: str) -> str:
        state = str(state).strip().lower()
        if state == "frustration":
            return "Use a calm and supportive tone. Break the explanation into smaller steps. Reduce jargon and provide reassurance."
        if state == "confusion":
            return "Clarify the key concept first. Use a simple analogy if helpful. Explain why the result or method may seem unexpected."
        if state == "boredom":
            return "Make the response concise and engaging. Highlight why the concept matters. Ask a guiding question or suggest a next step."
        if state == "engagement":
            return "Give a clear and efficient explanation. Focus on the main idea and help the user move forward."
        return "Give a clear, structured academic explanation."


class EmotionBuffer:
    def __init__(self, maxlen: int = 10):
        self._items: deque[str] = deque(maxlen=max(1, int(maxlen or 10)))

    @property
    def buffer_size(self) -> int:
        return self._items.maxlen or 0

    def push(self, state: str) -> str:
        normalized = str(state or "engagement").strip().lower() or "engagement"
        self._items.append(normalized)
        return self.get_stable_state()

    def get_stable_state(self) -> str:
        if not self._items:
            return "engagement"
        return Counter(self._items).most_common(1)[0][0]

    def values(self) -> list[str]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()


class RawEmotionInferencer:
    def __init__(self, checkpoint_path: str | Path | None = None, device: str | None = None):
        self.requested_checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None
        self.device_name = device
        self.device = "cpu"
        self.model = None
        self.classes: list[str] = []
        self.architecture = DEFAULT_RAW_ARCHITECTURE
        self.model_output_type = "unknown"
        self.checkpoint_path: Path | None = None
        self.load_error: str | None = None
        self.load_warnings: list[str] = []
        self._loaded = False

    def status(self) -> dict[str, Any]:
        selection = self.checkpoint_selection()
        path_text = str(selection.get("checkpoint_path") or "").strip()
        path = self.checkpoint_path or (Path(path_text) if path_text else None)
        model_output_type = self.model_output_type if self.model_output_type != "unknown" or self._loaded else str(selection.get("model_output_type") or "unknown")
        classes = self.classes if self.classes or self._loaded else list(selection.get("classes") or [])
        architecture = self.architecture if self.architecture != DEFAULT_RAW_ARCHITECTURE or self._loaded else str(selection.get("architecture") or selection.get("arch") or self.architecture)
        return {
            "model_loaded": self._loaded,
            "model_output_type": model_output_type,
            "raw_detection_available": model_output_type == "raw_emotion" and not bool(selection.get("loading_error")),
            "checkpoint_path": _safe_checkpoint_label(path) if path else "",
            "architecture": architecture,
            "classes": list(classes),
            "loading_error": self.load_error or selection.get("loading_error"),
            "loading_warnings": list(dict.fromkeys([*self.load_warnings, *(selection.get("warnings") or [])])),
            "device": self.device,
        }

    def load(self) -> dict[str, Any]:
        if self._loaded:
            return self.status()
        path = self._find_checkpoint_path()
        if not path:
            self.load_error = "Emotion checkpoint is missing. Configure RAW_EMOTION_CHECKPOINT_PATH or EMOTION_CHECKPOINT_PATH."
            self.model_output_type = "unknown"
            return self.status()
        self.checkpoint_path = path
        try:
            import timm  # type: ignore
            import torch  # type: ignore
        except Exception as exc:
            self.load_error = f"PyTorch/timm dependencies are unavailable: {exc}"
            self.model_output_type = "unknown"
            return self.status()

        try:
            self.device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = self._safe_load_checkpoint(torch, path)
            info = inspect_checkpoint_metadata(checkpoint)
            self.model_output_type = info["model_output_type"]
            self.classes = list(info["classes"])
            self.architecture = str(info["architecture"] or DEFAULT_RAW_ARCHITECTURE)
            if self.model_output_type not in {"raw_emotion", "academic_state"}:
                self.load_error = "Checkpoint classes do not match raw-emotion or academic-state labels."
                return self.status()
            model = timm.create_model(self.architecture, pretrained=False, num_classes=len(self.classes))
            state_dict = self._extract_state_dict(checkpoint)
            state_dict = self._clean_state_dict_keys(state_dict)
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception as exc:
                self.load_warnings.append(f"Strict state_dict load failed; retried with strict=False: {exc}")
                model.load_state_dict(state_dict, strict=False)
            self.model = model.to(self.device)
            self.model.eval()
            self._loaded = True
            self.load_error = None
        except Exception as exc:
            self.model = None
            self._loaded = False
            self.load_error = f"Could not load emotion checkpoint: {exc}"
        return self.status()

    def predict_probabilities(self, image: Any) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        if not self._loaded or self.model is None:
            return {**self.status(), "error": self.load_error or "Emotion checkpoint is not loaded."}
        try:
            import numpy as np  # type: ignore
            import torch  # type: ignore
            from PIL import Image  # type: ignore

            pil_image = self._to_pil_image(image, Image, np)
            resized = pil_image.resize((224, 224))
            arr = np.asarray(resized).astype("float32") / 255.0
            tensor = torch.tensor(arr).permute(2, 0, 1)
            tensor = (tensor - 0.5) / 0.5
            tensor = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(tensor), dim=1)[0].detach().cpu().numpy()
            labels = [_canonical_raw_label(label) for label in self.classes] if self.model_output_type == "raw_emotion" else list(self.classes)
            return {
                **self.status(),
                "probabilities": {label: float(probs[index]) for index, label in enumerate(labels)},
            }
        except Exception as exc:
            return {**self.status(), "error": f"Emotion model prediction failed: {exc}"}

    def _find_checkpoint_path(self) -> Path | None:
        selection = self.checkpoint_selection()
        path = str(selection.get("checkpoint_path") or "").strip()
        if not path:
            return None
        checkpoint_path = Path(path)
        return checkpoint_path if checkpoint_path.exists() else None

    def _checkpoint_candidates(self) -> list[Path]:
        requested_mode = _normalize_model_mode(os.environ.get("EMOTION_MODEL_MODE") or "auto")
        values = [entry["path"] for entry in _checkpoint_selection_candidates(self.requested_checkpoint_path, requested_mode, PROJECT_ROOT)]
        candidates: list[Path] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            candidates.append(Path(text).expanduser())
        return candidates

    def checkpoint_selection(self) -> dict[str, Any]:
        return select_emotion_checkpoint(checkpoint_path=self.requested_checkpoint_path, project_root=PROJECT_ROOT)

    def has_explicit_checkpoint(self) -> bool:
        if self.requested_checkpoint_path:
            return True
        return bool(
            os.environ.get("RAW_EMOTION_CHECKPOINT_PATH", "").strip()
            or os.environ.get("EMOTION_CHECKPOINT_PATH", "").strip()
            or _normalize_model_mode(os.environ.get("EMOTION_MODEL_MODE") or "auto") != "auto"
        )

    @staticmethod
    def _safe_load_checkpoint(torch: Any, path: Path) -> Any:
        return _safe_torch_load(torch, path)

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
        if not isinstance(checkpoint, dict):
            return checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        return checkpoint

    @staticmethod
    def _clean_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
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
    def _to_pil_image(image: Any, Image: Any, np: Any) -> Any:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("Expected an RGB image or array.")
        return Image.fromarray(arr[:, :, :3].astype("uint8")).convert("RGB")


class CombinedEmotionPipeline:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        buffer_size: int = 10,
        inferencer: RawEmotionInferencer | None = None,
        mapper: EmotionMapper | None = None,
        buffer: EmotionBuffer | None = None,
    ):
        self.mapper = mapper or EmotionMapper()
        self.buffer = buffer or EmotionBuffer(maxlen=buffer_size)
        self.inferencer = inferencer or RawEmotionInferencer(checkpoint_path=checkpoint_path)

    def status(self, fallback_status: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.inferencer.has_explicit_checkpoint() and not self.inferencer.status().get("model_loaded"):
            self.inferencer.load()
        status = self.inferencer.status()
        fallback_mode = str((fallback_status or {}).get("model_output_type") or "unknown")
        should_use_fallback_status = bool(
            fallback_status
            and (
                status.get("model_output_type") == "unknown"
                or (
                    not self.inferencer.has_explicit_checkpoint()
                    and status.get("model_output_type") == fallback_mode
                    and bool(fallback_status.get("model_loaded"))
                )
            )
        )
        if should_use_fallback_status and fallback_status:
            return {
                "model_loaded": bool(fallback_status.get("model_loaded")),
                "model_output_type": fallback_mode,
                "raw_detection_available": bool(fallback_status.get("raw_emotion_available")),
                "checkpoint_path": str(fallback_status.get("checkpoint_path") or ""),
                "architecture": str(fallback_status.get("architecture") or ""),
                "classes": list(fallback_status.get("classes") or []),
                "loading_error": fallback_status.get("loading_error"),
                "mapper_available": True,
                "buffer_size": self.buffer.buffer_size,
            }
        return {
            **status,
            "mapper_available": True,
            "buffer_size": self.buffer.buffer_size,
        }

    def predict(
        self,
        image: Any,
        fallback_prediction: dict[str, Any] | None = None,
        fallback_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selection = self.inferencer.checkpoint_selection()
        selected_mode = str(selection.get("model_output_type") or "unknown")
        if (
            fallback_prediction
            and str(fallback_prediction.get("model_output_type") or "") == "academic_state"
            and not self.inferencer.has_explicit_checkpoint()
            and selected_mode != "raw_emotion"
        ):
            return self._academic_state_payload(
                probabilities=fallback_prediction.get("state_distribution") or fallback_prediction.get("distribution") or {},
                status=fallback_status or {},
                confidence=fallback_prediction.get("confidence"),
                state=fallback_prediction.get("academic_state"),
                mapping_rule="bypassed: checkpoint directly predicts academic states",
            )
        prediction = self.inferencer.predict_probabilities(image)
        if not prediction.get("error") and prediction.get("model_loaded"):
            if prediction.get("model_output_type") == "raw_emotion":
                return self._raw_emotion_payload(prediction)
            if prediction.get("model_output_type") == "academic_state":
                return self._academic_state_payload(
                    probabilities=prediction.get("probabilities") or {},
                    status=prediction,
                    mapping_rule="bypassed: checkpoint directly predicts academic states",
                )
        if fallback_prediction and str(fallback_prediction.get("model_output_type") or "") == "academic_state":
            return self._academic_state_payload(
                probabilities=fallback_prediction.get("state_distribution") or fallback_prediction.get("distribution") or {},
                status=fallback_status or {},
                confidence=fallback_prediction.get("confidence"),
                state=fallback_prediction.get("academic_state"),
                mapping_rule="bypassed: checkpoint directly predicts academic states",
                source_error=prediction.get("error"),
            )
        if fallback_prediction and str(fallback_prediction.get("model_output_type") or "") == "raw_emotion":
            probabilities = fallback_prediction.get("raw_distribution") or fallback_prediction.get("probabilities") or {}
            return self._raw_emotion_payload({**(fallback_status or {}), "probabilities": probabilities})
        return self._unknown_payload(prediction.get("error") or "Emotion pipeline could not produce a prediction.")

    def _raw_emotion_payload(self, prediction: dict[str, Any]) -> dict[str, Any]:
        raw_probabilities = _normalized_distribution(prediction.get("probabilities") or {}, RAW_EMOTION_CLASSES, raw=True)
        raw_label = max(raw_probabilities, key=raw_probabilities.get)
        raw_confidence = float(raw_probabilities[raw_label])
        mapped_state, scores = self.mapper.map_probs_to_state(raw_probabilities)
        stable_state = self.buffer.push(mapped_state)
        return {
            "model_output_type": "raw_emotion",
            "checkpoint_path": str(prediction.get("checkpoint_path") or ""),
            "architecture": str(prediction.get("architecture") or ""),
            "classes": list(prediction.get("classes") or RAW_EMOTION_CLASSES),
            "raw_detection_available": True,
            "raw_detection": {
                "label": raw_label,
                "confidence": raw_confidence,
                "probabilities": raw_probabilities,
            },
            "mapped_academic_state": {
                "state": mapped_state,
                "scores": {state: round(float(scores.get(state, 0.0)), 6) for state in MAPPED_STATE_CLASSES},
                "mapping_rule": self.mapper.mapping_rule_for_state(mapped_state),
            },
            "smoothed_state": {
                "state": stable_state,
                "buffer": self.buffer.values(),
                "buffer_size": self.buffer.buffer_size,
            },
            "response_strategy": self.mapper.state_to_response_strategy(stable_state),
        }

    def _academic_state_payload(
        self,
        probabilities: dict[str, Any],
        status: dict[str, Any],
        confidence: Any = None,
        state: Any = None,
        mapping_rule: str = "bypassed: checkpoint directly predicts academic states",
        source_error: str | None = None,
    ) -> dict[str, Any]:
        distribution = _normalized_distribution(probabilities, ACADEMIC_STATE_CLASSES)
        academic_state = str(state or max(distribution, key=distribution.get))
        if academic_state not in distribution:
            academic_state = max(distribution, key=distribution.get)
        academic_confidence = _safe_float(confidence, distribution[academic_state])
        stable_state = self.buffer.push(academic_state)
        payload = {
            "model_output_type": "academic_state",
            "checkpoint_path": str(status.get("checkpoint_path") or ""),
            "architecture": str(status.get("architecture") or ""),
            "classes": list(status.get("classes") or ACADEMIC_STATE_CLASSES),
            "raw_detection_available": False,
            "raw_detection": None,
            "academic_state": {
                "state": academic_state,
                "confidence": academic_confidence,
                "distribution": distribution,
            },
            "mapped_academic_state": {
                "state": academic_state,
                "scores": distribution,
                "mapping_rule": mapping_rule,
            },
            "smoothed_state": {
                "state": stable_state,
                "buffer": self.buffer.values(),
                "buffer_size": self.buffer.buffer_size,
            },
            "response_strategy": self.mapper.state_to_response_strategy(stable_state),
        }
        if source_error:
            payload["pipeline_warning"] = source_error
        return payload

    def _unknown_payload(self, error: str) -> dict[str, Any]:
        return {
            "model_output_type": "unknown",
            "raw_detection_available": False,
            "raw_detection": None,
            "mapped_academic_state": None,
            "smoothed_state": {
                "state": self.buffer.get_stable_state(),
                "buffer": self.buffer.values(),
                "buffer_size": self.buffer.buffer_size,
            },
            "response_strategy": self.mapper.state_to_response_strategy(self.buffer.get_stable_state()),
            "error": error,
        }


def _classes_from_checkpoint(checkpoint: Any) -> list[str]:
    if not isinstance(checkpoint, dict):
        return []
    classes = checkpoint.get("classes") or checkpoint.get("label_order")
    if isinstance(classes, list) and classes:
        return [str(item).strip().lower() for item in classes]
    class_to_idx = checkpoint.get("class_to_idx")
    if isinstance(class_to_idx, dict) and class_to_idx:
        try:
            return [str(label).strip().lower() for label, _ in sorted(class_to_idx.items(), key=lambda item: int(item[1]))]
        except Exception:
            return [str(label).strip().lower() for label in class_to_idx]
    return []


def _checkpoint_selection_candidates(checkpoint_path: str | Path | None, requested_mode: str, root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if checkpoint_path:
        candidates.append({"path": checkpoint_path, "source": "constructor", "requires_raw": requested_mode == "raw_emotion"})
        return candidates

    raw_env = os.environ.get("RAW_EMOTION_CHECKPOINT_PATH", "").strip()
    emotion_env = os.environ.get("EMOTION_CHECKPOINT_PATH", "").strip()
    if requested_mode == "raw_emotion":
        if raw_env:
            candidates.append({"path": raw_env, "source": "RAW_EMOTION_CHECKPOINT_PATH", "requires_raw": True})
        if emotion_env:
            candidates.append({"path": emotion_env, "source": "EMOTION_CHECKPOINT_PATH", "requires_raw": True})
        candidates.append({"path": DEFAULT_RAW_CHECKPOINT_CANDIDATE, "source": "default_raw_checkpoint", "requires_raw": True})
        return _dedupe_candidate_entries(candidates)
    if requested_mode == "academic_state":
        if emotion_env:
            candidates.append({"path": emotion_env, "source": "EMOTION_CHECKPOINT_PATH", "requires_raw": False})
        candidates.append({"path": DEFAULT_ACADEMIC_CHECKPOINT_CANDIDATE, "source": "default_academic_checkpoint", "requires_raw": False})
        return _dedupe_candidate_entries(candidates)

    if raw_env:
        candidates.append({"path": raw_env, "source": "RAW_EMOTION_CHECKPOINT_PATH", "requires_raw": True})
    if emotion_env:
        candidates.append({"path": emotion_env, "source": "EMOTION_CHECKPOINT_PATH", "requires_raw": False})
    candidates.append({"path": DEFAULT_RAW_CHECKPOINT_CANDIDATE, "source": "default_raw_checkpoint", "requires_raw": True})
    candidates.append({"path": DEFAULT_ACADEMIC_CHECKPOINT_CANDIDATE, "source": "default_academic_checkpoint", "requires_raw": False})
    for value in DEFAULT_CHECKPOINT_CANDIDATES:
        candidates.append({"path": value, "source": "default_candidate", "requires_raw": value == DEFAULT_RAW_CHECKPOINT_CANDIDATE})
    return _dedupe_candidate_entries(candidates)


def _dedupe_candidate_entries(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate.get("path") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(candidate)
    return deduped


def _resolve_candidate_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _normalize_model_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower()
    aliases = {
        "": "auto",
        "raw": "raw_emotion",
        "raw_emotion_model": "raw_emotion",
        "raw_facial_emotion": "raw_emotion",
        "academic": "academic_state",
        "academic_state_model": "academic_state",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in {"auto", "raw_emotion", "academic_state"} else "auto"


def _safe_torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _detect_model_output_type(classes: list[str], checkpoint: Any = None) -> str:
    explicit = ""
    if isinstance(checkpoint, dict):
        explicit = str(checkpoint.get("model_output_type") or "").strip().lower()
    if explicit in {"raw_emotion", "raw_emotion_model", "raw_facial_emotion"}:
        return "raw_emotion"
    if explicit in {"academic_state", "academic_state_model"}:
        return "academic_state"
    canonical_raw = [_canonical_raw_label(label) for label in classes]
    if len(canonical_raw) == len(RAW_EMOTION_CLASSES) and set(canonical_raw) == set(RAW_EMOTION_CLASSES):
        return "raw_emotion"
    academic = [str(label).strip().lower() for label in classes]
    if len(academic) == len(ACADEMIC_STATE_CLASSES) and set(academic) == set(ACADEMIC_STATE_CLASSES):
        return "academic_state"
    return "unknown"


def _normalized_distribution(probabilities: dict[str, Any], labels: list[str], raw: bool = False) -> dict[str, float]:
    values: dict[str, float] = {}
    source = {_canonical_raw_label(key) if raw else str(key).strip().lower(): value for key, value in probabilities.items()}
    for label in labels:
        key = _canonical_raw_label(label) if raw else label
        values[key] = max(0.0, _safe_float(source.get(key), 0.0))
    total = sum(values.values())
    if total <= 0:
        return {label: round(1.0 / len(labels), 6) for label in labels}
    return {label: round(values[_canonical_raw_label(label) if raw else label] / total, 6) for label in labels}


def _canonical_raw_label(label: Any) -> str:
    text = str(label or "").strip().lower()
    aliases = {"angry": "anger", "happiness": "happy", "sadness": "sad"}
    return aliases.get(text, text)


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _safe_checkpoint_label(path: Path | None) -> str:
    if not path:
        return ""
    try:
        resolved = path.resolve()
        root = PROJECT_ROOT.resolve()
        if resolved == root or root in resolved.parents:
            return str(resolved.relative_to(root))
    except Exception:
        pass
    return str(path)
