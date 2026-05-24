from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> Path:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / "assistant.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    return log_file
