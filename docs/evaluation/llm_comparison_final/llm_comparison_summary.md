# Final LLM Comparison Summary

This folder contains the final LLM comparison evidence used for the CS731 Team 15 Emotion-Aware Academic Assistant report.

## Files

- `2026-05-25-result.xlsx`: final LLM scoring and latency table.
- `test-result-final-02.zip`: supporting test outputs and raw comparison evidence.
- `llm_comparison_summary.md`: cleaned summary of the final comparison results.

## Evaluation Method

The final LLM comparison used an LLM-as-a-judge evaluation with human supervision. ChatGPT scored each response according to a fixed rubric, and the scores were manually reviewed to check that the ratings were reasonable and consistent with the response content.

Three LLMs were compared:

- Claude Opus 4.7 Fast
- Gemini 3.1 Pro Preview
- DeepSeek V4 Pro

The models were tested across three prompt types:

- Baseline explanation
- Strategy-conditioned response
- Strategy planner

For baseline and strategy-conditioned responses, the scoring criteria were:

1. Clarity
2. Pedagogical quality
3. Emotional alignment
4. Structure
5. Cognitive load

For strategy-planner outputs, the scoring criteria were:

1. Strategy appropriateness
2. Learning-state use
3. Output format correctness
4. Rationale quality
5. Safety / non-overreaction

Each criterion was scored from 1 to 5, giving a maximum score of 25 per response.

## Overall Model Comparison

| Model | Average Score | Average Latency | Overall Rank |
| --- | ---: | ---: | ---: |
| Claude Opus 4.7 Fast | 23.7 | 7.4s | 1 |
| Gemini 3.1 Pro Preview | 21.3 | 19.6s | 2 |
| DeepSeek V4 Pro | 19.7 | 38.1s | 3 |

## Prompt-Type Comparison

| Prompt Type | Average Score | Average Latency | Interpretation |
| --- | ---: | ---: | --- |
| Baseline explanation | 22.3 | 20.5s | Highest average quality |
| Strategy-conditioned response | 21.7 | 12.8s | Best quality-speed balance |
| Strategy planner | 20.7 | 31.9s | Slowest and less stable for frequent real-time use |

## Key Findings

Claude achieved the best overall performance, with the highest average score and the lowest average latency. It was selected as the final demo LLM.

The strategy-conditioned prompt provided the best quality-speed trade-off among the three prompt types, making it suitable for interactive tutoring responses.

The strategy planner prompt had the highest latency, so it should be used when a new teaching strategy needs to be selected rather than for every response.

## Sanitization Note

The ZIP file contains JSON comparison outputs. Before committing future raw logs, review them for API keys, private conversations, user identifiers, local paths, or unrelated screenshots.
