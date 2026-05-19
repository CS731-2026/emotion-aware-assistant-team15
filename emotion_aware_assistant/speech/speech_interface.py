from __future__ import annotations

from typing import Protocol


class SpeechRecognizer(Protocol):
    def transcribe_once(self) -> str:
        ...

    @property
    def is_available(self) -> bool:
        ...

    @property
    def status_message(self) -> str:
        ...
