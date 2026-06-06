# Emotion Model Placeholder

Install local emotion checkpoints here. Do not commit weight files.

```text
best_model.pt
raw_8class_best.pt
```

The complete source code is available in the GitHub Classroom repository. Because the trained emotion checkpoints are large binary files and are not stored directly in the normal Git repository, the checkpoint files used by the final demo are provided separately through REANNZ FileSender:

https://filesender.reannz.co.nz/?s=download&token=d1fcd0a7-d7e4-4c8b-87c5-7c1380827c74

The archive contains the final 8-class raw-emotion runtime checkpoint, `raw_8class_best.pt`, and the supported 4-state academic-state checkpoint, `best_model.pt`. Extract the archive at the repository root or place the checkpoint files in this directory.

The 4-class academic-state checkpoint path remains supported as a baseline/fallback/comparison model:

```bash
python scripts/install_emotion_checkpoint.py --source /home/rli/downloads/best
```

For the teammate raw 8-class facial emotion checkpoint:

```bash
cp /home/rli/downloads/convnext_tiny_8_emotion/convnext_tiny.fb_in22k_ft_in1k_best.pt models/emotion_model/raw_8class_best.pt
python scripts/configure_emotion_checkpoint.py --checkpoint models/emotion_model/raw_8class_best.pt --mode raw_emotion
```

Raw 8-class labels are `anger, contempt, disgust, fear, happy, neutral, sad, surprise`. The final active chain uses this raw checkpoint as low-level evidence, maps probabilities into learning-centered academic scores, then combines reaction windows and dialogue/strategy history into a process-aware learning signal package. In `raw_emotion` mode the direct 4-class comparison model must not influence strategy selection. `metadata.json` is safe to track; `*.pt`, `*.pth`, and `*.ckpt` are ignored recursively.
