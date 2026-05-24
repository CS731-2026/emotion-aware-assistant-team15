from __future__ import annotations


class DummySpeechRecognizer:
    @property
    def is_available(self) -> bool:
        return False

    @property
    def status_message(self) -> str:
        return "Speech input is disabled because no speech backend is configured."

    def transcribe_once(self) -> str:
        raise RuntimeError(self.status_message)
