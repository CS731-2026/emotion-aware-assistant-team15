# Evaluation Scaffold

This folder documents the evaluation workflow used by the local LLM comparison tool.

The final LLM comparison used an LLM-as-a-judge evaluation with human supervision. ChatGPT scored each response according to fixed rubrics for answer quality, strategy quality, safety, and latency. Human supervision was used to review the judge outputs and check that the ratings were reasonable and consistent with the response content.

Real comparison logs should remain outside Git unless they have been reviewed, sanitized, and intentionally copied into the repository.
