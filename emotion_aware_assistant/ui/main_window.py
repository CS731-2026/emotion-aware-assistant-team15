from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QMessageBox, QSplitter, QVBoxLayout, QWidget  # type: ignore

from emotion_aware_assistant.app import AssistantSession
from emotion_aware_assistant.llm.model_registry import configured_models
from emotion_aware_assistant.ui.widgets.chat_panel import ChatPanel
from emotion_aware_assistant.ui.widgets.control_panel import ControlPanel
from emotion_aware_assistant.ui.widgets.emotion_panel import EmotionPanel
from emotion_aware_assistant.ui.widgets.paper_viewer import PaperViewer


class MainWindow(QWidget):  # pragma: no cover - requires PyQt5
    def __init__(self, config):
        super().__init__()
        self.setWindowTitle("Emotion-Aware Academic Assistant")
        self.resize(1180, 760)
        self.config = config
        self.session = AssistantSession(config)
        self.paper = PaperViewer()
        self.chat = ChatPanel(list(configured_models(config).keys()))
        self.emotion = EmotionPanel()
        self.control = ControlPanel()
        self.control.set_status(self.session)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.emotion)
        right_layout.addWidget(self.control)
        right_layout.addWidget(self.chat, 1)

        splitter = QSplitter()
        splitter.addWidget(self.paper)
        splitter.addWidget(right)
        splitter.setSizes([700, 480])
        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

        self.paper.open_txt_button.clicked.connect(lambda: self._open_file("TXT files (*.txt)"))
        self.paper.open_pdf_button.clicked.connect(lambda: self._open_file("PDF files (*.pdf)"))
        self.paper.prev_button.clicked.connect(self._prev_page)
        self.paper.next_button.clicked.connect(self._next_page)
        self.paper.explain_button.clicked.connect(self._explain_selected)
        self.paper.summarize_button.clicked.connect(self._summarize_page)
        self.chat.ask_button.clicked.connect(self._ask)
        self.emotion.override.currentTextChanged.connect(self._set_override)

    def _open_file(self, filter_text: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open paper", str(Path.cwd()), filter_text)
        if not path:
            return
        try:
            document = self.session.load_document(path)
            page = document.page(1)
            self.paper.set_page_text(page.text, 1, document.page_count)
        except Exception as exc:
            QMessageBox.warning(self, "Open failed", str(exc))

    def _prev_page(self) -> None:
        self._goto_page(max(1, self.session.current_page_number - 1))

    def _next_page(self) -> None:
        if self.session.document:
            self._goto_page(min(self.session.document.page_count, self.session.current_page_number + 1))

    def _goto_page(self, page_number: int) -> None:
        if not self.session.document:
            return
        text = self.session.set_page(page_number)
        self.paper.set_page_text(text, page_number, self.session.document.page_count)

    def _set_override(self, value: str) -> None:
        try:
            if value == "auto":
                return
            snapshot = self.session.set_override(value)
            self.emotion.update_snapshot(snapshot)
        except Exception as exc:
            QMessageBox.warning(self, "Emotion override failed", str(exc))

    def _selected_context(self):
        selected = self.paper.selected_text()
        if selected:
            self.session.set_manual_context(selected, title=self.session.document.title if self.session.document else "Selected text")
        return self.session.current_paper_context()

    def _explain_selected(self) -> None:
        self._ask_with_question("Can you explain this selected passage?")

    def _summarize_page(self) -> None:
        self._ask_with_question("Can you summarize the current page?")

    def _ask(self) -> None:
        question = self.chat.input.toPlainText().strip()
        self._ask_with_question(question)

    def _ask_with_question(self, question: str) -> None:
        if not question:
            return
        self.chat.append_user(question)
        response = self.session.ask(question, paper_context=self._selected_context(), model_alias=self.chat.model_selector.currentText())
        self.chat.append_assistant(response.text)
        self.chat.input.clear()
