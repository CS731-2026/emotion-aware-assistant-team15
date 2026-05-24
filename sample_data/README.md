# Sample Data

`sample_paper.txt` is generated or verified by:

```bash
python scripts/create_sample_data.py
```

It is a generic academic-style sample for smoke checks and demos.

## Curated Demo Fixtures

- `demo_papers/` contains small synthetic PDFs for `/pdf-chat` upload/open testing.
- `demo_runtime/` contains sanitized prompt-library and prompt-snapshot examples for `/llm-compare`.
- `demo_settings/` contains a local settings template with blank credentials only.

`runtime_uploads/` and `logs/` are generated locally and intentionally ignored by Git. Configure real provider credentials through `/settings`, and do not commit local runtime files, uploaded papers, generated embeddings, logs, or saved API keys.
