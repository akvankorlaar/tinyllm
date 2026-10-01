# tinypython

**Experiment 1 — SmolLM2-135M Python specialization.**

One question: *can ~30M carefully selected Python tokens make a 135M
general-purpose model substantially better at Python?*

This repo is deliberately boring. No tokenizer changes, no architecture
changes, no new loss, no RL, no distillation. Just clean data + continued
causal-LM pretraining + a small execution-based benchmark.

## Scientific design

Same 30M-token budget for each trained variant, so any difference is the data,
not the compute:

| Model | Training |
|---|---|
| **A** | original SmolLM2-135M (no training) |
| **B** | base + 30M **general** tokens (control) |
| **C** | base + 30M **Python** tokens (treatment) |

All three run against the same held-out Python benchmark.

- `C > B > A` → Python specialization helps. Keep going.
- `C ≈ B` → data recipe isn't good enough.
- `C < A` → also informative (catastrophic forgetting / bad data).

## Benchmark

- **A. Completion** (`eval/completion.py`) — continue a prefix; surface signal.
- **B. Fill-in-the-middle** (`eval/fim.py`) — infill a hole, then execute.
- **C. Execute** (`eval/execute.py`) — solve a problem, run against hidden
  tests. **This is the one that matters.** Pretty nonsense that doesn't run fails.
- Plus held-out **perplexity** on Python the model never trained on.

## Layout

```
data/     download.py  clean.py  deduplicate.py  common.py
model/    train.py
eval/     completion.py  fim.py  execute.py  run.py  compare.py  common_eval.py
          problems/{problems,completion,fim}.jsonl
configs/  experiment_001.yaml
scripts/  run_pipeline.sh
```

Everything is driven by `configs/experiment_001.yaml`.

## Setup (uv)

```bash
uv sync
```

`torch` resolves to a CPU wheel by default. On a CUDA box, install the matching
wheel: `uv pip install torch --index-url https://download.pytorch.org/whl/cu121`.

### Data source

Default is `data.source: starcoderdata` — inline content, **no AWS**, already
cleaned/deduped/filtered. Setup:

1. Accept the gate (instant) at
   <https://huggingface.co/datasets/bigcode/starcoderdata>.
2. `uv run hf auth login` (paste a read token).

Other sources (set `data.source`):

- `the-stack-v2-dedup` — Stack v2's scale, but metadata-only on HF; content
  comes from the Software Heritage S3 bucket, so it also needs AWS creds
  (`uv run aws configure`, any free account). Gate:
  <https://huggingface.co/datasets/bigcode/the-stack-v2-dedup>.
- `codeparrot` — no gate, but script-based; may not load under datasets 5.x.

## Run

Full pipeline (data + eval need no GPU; training does):

```bash
bash scripts/run_pipeline.sh
```

Or step by step:

```bash
uv run python data/download.py    configs/experiment_001.yaml --corpus python
uv run python data/download.py    configs/experiment_001.yaml --corpus general
uv run python data/clean.py       configs/experiment_001.yaml --corpus python
uv run python data/deduplicate.py configs/experiment_001.yaml --corpus python
# ... general too, then:
uv run python model/train.py      configs/experiment_001.yaml --variant python
uv run python eval/run.py         configs/experiment_001.yaml --model runs/experiment_001_python --name C_python
uv run python eval/compare.py     configs/experiment_001.yaml
```

## Compute

30M tokens × 135M params × ~6 FLOPs/token ≈ **2.4 × 10¹⁶ FLOPs**. One modern GPU
is plenty; prioritize fast iteration over utilization. There is no GPU on the
dev box — build and test the pipeline here, train on a rented/remote GPU.
