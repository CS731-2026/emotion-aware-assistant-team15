# Emotion Model Placeholder

Install the teammate academic-state checkpoint here:

```text
best_model.pt
metadata.json
```

The current checkpoint is a 4-class academic-state model, not an 8-class raw facial emotion model. Use:

```bash
python scripts/install_emotion_checkpoint.py --source /home/rli/下载/best
```

Do not commit weight files. `metadata.json` is safe to track; `*.pt`, `*.pth`, and `*.ckpt` are ignored.
