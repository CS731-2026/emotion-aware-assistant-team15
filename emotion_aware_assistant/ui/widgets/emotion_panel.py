from __future__ import annotations

from PyQt5.QtWidgets import QComboBox, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget  # type: ignore

from emotion_aware_assistant.emotion.labels import ALLOWED_EMOTIONS, LEARNING_STATES


class EmotionPanel(QWidget):  # pragma: no cover - requires PyQt5
    def __init__(self):
        super().__init__()
        self.start_button = QPushButton("Start camera")
        self.stop_button = QPushButton("Stop camera")
        self.mode = QComboBox()
        self.mode.addItems(["auto", "manual", "dummy", "teammate model"])
        self.override = QComboBox()
        self.override.addItems(["auto"] + ALLOWED_EMOTIONS + [state for state in LEARNING_STATES if state != "uncertain"])
        self.raw = QLabel("-")
        self.smoothed = QLabel("-")
        self.state = QLabel("-")
        self.trend = QLabel("-")
        self.confidence = QLabel("-")
        self.duration = QLabel("-")
        self.strategy = QLabel("-")
        form = QFormLayout()
        form.addRow("Mode", self.mode)
        form.addRow("Override", self.override)
        form.addRow("Raw", self.raw)
        form.addRow("Smoothed", self.smoothed)
        form.addRow("Learning state", self.state)
        form.addRow("Trend", self.trend)
        form.addRow("Confidence", self.confidence)
        form.addRow("Duration", self.duration)
        form.addRow("Strategy", self.strategy)
        layout = QVBoxLayout(self)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addLayout(form)

    def update_snapshot(self, snapshot) -> None:
        self.raw.setText(snapshot.raw_emotion)
        self.smoothed.setText(snapshot.smoothed_emotion)
        self.state.setText(snapshot.state)
        self.trend.setText(snapshot.trend)
        self.confidence.setText(f"{snapshot.confidence:.2f}")
        self.duration.setText(f"{snapshot.duration_sec:.1f}s")
        self.strategy.setText(snapshot.strategy)
