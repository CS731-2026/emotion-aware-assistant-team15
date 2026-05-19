from __future__ import annotations

import threading
import time
from typing import Callable

from .affective_trend_tracker import AffectiveTrendTracker
from .emotion_buffer import EmotionBuffer
from .face_detector import create_face_detector
from .state_mapper import map_prediction_to_learning_state


class CameraWorker:
    """Threaded webcam pipeline. It is optional and never calls the LLM."""

    def __init__(
        self,
        recognizer,
        buffer: EmotionBuffer,
        tracker: AffectiveTrendTracker,
        status_callback: Callable[[object, object], None] | None = None,
        frame_callback: Callable[[object], None] | None = None,
        camera_index: int = 0,
        target_fps: float = 12.0,
    ):
        self.recognizer = recognizer
        self.buffer = buffer
        self.tracker = tracker
        self.status_callback = status_callback
        self.frame_callback = frame_callback
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.detector = create_face_detector()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.last_error: str | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            import cv2  # type: ignore
        except Exception:
            self.last_error = "OpenCV is not installed; camera mode is unavailable."
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.last_error = "No webcam could be opened; use manual emotion mode."
            return
        delay = 1.0 / max(1.0, self.target_fps)
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.last_error = "Webcam frame capture failed."
                    break
                if self.frame_callback:
                    self.frame_callback(frame)
                boxes = self.detector.detect(frame) if self.detector.is_available else []
                if boxes:
                    box = max(boxes, key=lambda item: item.w * item.h * item.confidence)
                    crop = frame[box.y : box.y + box.h, box.x : box.x + box.w]
                    prediction = self.recognizer.predict(crop)
                    prediction = type(prediction)(
                        emotion=prediction.emotion,
                        confidence=prediction.confidence,
                        probabilities=prediction.probabilities,
                        timestamp=prediction.timestamp,
                        face_bbox=(box.x, box.y, box.w, box.h),
                        source=prediction.source,
                    )
                    smoothed = self.buffer.add(prediction)
                    state = map_prediction_to_learning_state(prediction, smoothed)
                    snapshot = self.tracker.update(state)
                    if self.status_callback:
                        self.status_callback(smoothed, snapshot)
                time.sleep(delay)
        finally:
            cap.release()
