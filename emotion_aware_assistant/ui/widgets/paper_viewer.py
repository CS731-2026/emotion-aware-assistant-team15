from __future__ import annotations

from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QPlainTextEdit, QVBoxLayout, QWidget  # type: ignore


class PaperViewer(QWidget):  # pragma: no cover - requires PyQt5
    def __init__(self):
        super().__init__()
        self.open_pdf_button = QPushButton("Open PDF")
        self.open_txt_button = QPushButton("Open TXT")
        self.prev_button = QPushButton("Prev")
        self.next_button = QPushButton("Next")
        self.explain_button = QPushButton("Explain selected")
        self.summarize_button = QPushButton("Summarize page")
        self.page_label = QLabel("Page -")
        self.text = QPlainTextEdit()
        self.text.setPlaceholderText("Open a PDF or TXT paper.")
        self.selection_preview = QLabel("Selected passage preview")
        self.selection_preview.setWordWrap(True)

        top = QHBoxLayout()
        for widget in [self.open_pdf_button, self.open_txt_button, self.prev_button, self.next_button, self.page_label]:
            top.addWidget(widget)
        actions = QHBoxLayout()
        actions.addWidget(self.explain_button)
        actions.addWidget(self.summarize_button)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.text, 1)
        layout.addWidget(self.selection_preview)
        layout.addLayout(actions)

    def set_page_text(self, text: str, page_number: int, page_count: int) -> None:
        self.text.setPlainText(text)
        self.page_label.setText(f"Page {page_number}/{page_count}")

    def selected_text(self) -> str:
        return self.text.textCursor().selectedText().replace("\u2029", "\n")
