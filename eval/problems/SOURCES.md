# Benchmark sources

Built by `eval/build_benchmark.py`; every reference solution is executed
against its tests and dropped if it fails.

| File | Source | License |
|---|---|---|
| `handwritten.jsonl` | written for this repo | same as repo |
| `humaneval.jsonl` | [openai/human-eval](https://github.com/openai/human-eval) — Chen et al. 2021, *Evaluating Large Language Models Trained on Code* | MIT |
| `mbpp.jsonl` | [google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp) (sanitized split) — Austin et al. 2021, *Program Synthesis with Large Language Models*. Converted from a text prompt to a signature + docstring prompt. | CC BY 4.0 |
| `fim.jsonl` | derived: one line of each reference solution above blanked out | as its source |
| `completion.jsonl` | written for this repo | same as repo |

bigcode decontaminated starcoderdata (the training corpus) against HumanEval
and MBPP.
