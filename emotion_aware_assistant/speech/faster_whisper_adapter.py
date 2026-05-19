from __future__ import annotations


class FasterWhisperAdapter:
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None
        self._error: str | None = None
        try:
            from faster_whisper import WhisperModel  # type: ignore

            self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        except Exception as exc:
            self._error = str(exc)

    @property
    def is_available(self) -> bool:
        return self.model is not None

    @property
    def status_message(self) -> str:
        if self.is_available:
            return "Speech input is available through faster-whisper."
        return f"faster-whisper is unavailable: {self._error or 'not installed'}"

    def transcribe_once(self) -> str:
        raise RuntimeError(
            "Microphone recording is intentionally not started in the adapter scaffold. "
            "Connect an audio capture utility here for the final demo."
        )
