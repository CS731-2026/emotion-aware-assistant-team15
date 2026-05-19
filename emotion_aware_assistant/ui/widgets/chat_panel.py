from __future__ import annotations

from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout, QWidget  # type: ignore


class ChatPanel(QWidget):  # pragma: no cover - requires PyQt5
    def __init__(self, model_aliases: list[str]):
        super().__init__()
        self.history = QTextEdit()
        self.history.setReadOnly(True)
        self.input = QTextEdit()
        self.input.setMaximumHeight(90)
        self.ask_button = QPushButton("Ask")
        self.model_selector = QComboBox()
        self.model_selector.addItems(model_aliases or ["dummy"])
        self.followup_buttons = [
            QPushButton("Define terms"),
            QPushButton("Example"),
            QPushButton("Takeaway"),
        ]
        top = QHBoxLayout()
        top.addWidget(self.model_selector)
        top.addWidget(self.ask_button)
        followups = QHBoxLayout()
        for button in self.followup_buttons:
            followups.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.history, 1)
        layout.addWidget(self.input)
        layout.addLayout(top)
        layout.addLayout(followups)

    def append_user(self, text: str) -> None:
        self.history.append(f"<b>User:</b> {text}")

    def append_assistant(self, text: str) -> None:
        self.history.append(f"<b>Assistant:</b><br>{text.replace(chr(10), '<br>')}")
