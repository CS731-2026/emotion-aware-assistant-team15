from __future__ import annotations

try:
    from PyQt5.QtCore import QThread, pyqtSignal  # type: ignore
except Exception:  # pragma: no cover - only used when PyQt5 is installed
    QThread = object  # type: ignore

    def pyqtSignal(*args, **kwargs):  # type: ignore
        return None


class LLMWorker(QThread):  # type: ignore[misc]
    completed = pyqtSignal(object)

    def __init__(self, session, question: str, context=None, model_alias: str | None = None):
        super().__init__()
        self.session = session
        self.question = question
        self.context = context
        self.model_alias = model_alias

    def run(self):  # pragma: no cover - requires PyQt5 event loop
        response = self.session.ask(self.question, paper_context=self.context, model_alias=self.model_alias)
        self.completed.emit(response)
