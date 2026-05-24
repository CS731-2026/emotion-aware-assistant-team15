from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        print(f"numpy unavailable: {exc}")
        return 1
    from emotion_aware_assistant.emotion.teammate_emotion_adapter import TeammateEmotionAdapter

    adapter = TeammateEmotionAdapter()
    status = adapter.load()
    print("model_status:")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    dummy = np.full((224, 224, 3), 128, dtype="uint8")
    prediction = adapter.predict(dummy)
    print("prediction:")
    print(f"model_output_type: {prediction.get('model_output_type')}")
    print(f"academic_state: {prediction.get('academic_state')}")
    print(f"confidence: {prediction.get('confidence')}")
    print(f"state_distribution: {json.dumps(prediction.get('state_distribution'), ensure_ascii=False)}")
    print(f"raw_emotion_available: {prediction.get('raw_emotion_available')}")
    if prediction.get("error"):
        print(f"error: {prediction.get('error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
