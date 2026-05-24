from __future__ import annotations

from PyQt5.QtWidgets import QFormLayout, QLabel, QWidget  # type: ignore


class ControlPanel(QWidget):  # pragma: no cover - requires PyQt5
    def __init__(self):
        super().__init__()
        self.openrouter = QLabel("-")
        self.emotion_model = QLabel("-")
        self.face_detector = QLabel("-")
        self.speech = QLabel("-")
        self.log_path = QLabel("-")
        self.log_path.setWordWrap(True)
        layout = QFormLayout(self)
        layout.addRow("OpenRouter", self.openrouter)
        layout.addRow("Emotion model", self.emotion_model)
        layout.addRow("Face detector", self.face_detector)
        layout.addRow("Speech", self.speech)
        layout.addRow("Log file", self.log_path)

    def set_status(self, session) -> None:
        status = session.status(mode="gui")
        self.openrouter.setText("available" if status.llm_available else "dummy/fallback")
        self.emotion_model.setText("manual/dummy fallback")
        self.face_detector.setText("optional")
        self.speech.setText("optional")
        self.log_path.setText(str(session.logger.path))
