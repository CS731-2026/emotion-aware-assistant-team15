# Emotion Model Placeholder

Install local emotion checkpoints here. Do not commit weight files.

```text
best_model.pt
metadata.json
raw_8class_best.pt
```

The 4-class academic-state checkpoint path remains supported as a diagnostic/baseline/fallback model:

```bash
python scripts/install_emotion_checkpoint.py --source /home/rli/下载/best
```

For the teammate raw 8-class facial emotion checkpoint:

```bash
cp /home/rli/下载/convnext_tiny_8_emotion/convnext_tiny.fb_in22k_ft_in1k_best.pt models/emotion_model/raw_8class_best.pt
python scripts/configure_emotion_checkpoint.py --checkpoint models/emotion_model/raw_8class_best.pt --mode raw_emotion
```

Raw 8-class labels are `anger, contempt, disgust, fear, happy, neutral, sad, surprise`. The final active chain uses this raw checkpoint as low-level evidence, maps probabilities into learning-centered academic scores, then combines reaction windows and dialogue/strategy history into a process-aware learning signal package. In `raw_emotion` mode the direct 4-class diagnostic must not influence strategy selection. `metadata.json` is safe to track; `*.pt`, `*.pth`, and `*.ckpt` are ignored recursively.
