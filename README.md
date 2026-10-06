# tinypython

**The best Python LM we can get at ~20M parameters, trained from scratch on a laptop.**

A `MistralForCausalLM` (18.9M params) with its own 16k byte-level BPE
tokenizer, pretrained on ~400M tokens of cleaned Python from
`bigcode/starcoderdata`. No pretrained weights; every component is ours.

Why from scratch: there is no good open-source code model at this size, and
reusing a general tokenizer (e.g. SmolLM2's 49k vocab) would spend most of a
20M budget on embeddings. A Python-only 16k vocab costs 6.3M params and
leaves the rest for the transformer.

## Model

| | |
|---|---|
| architecture | `MistralForCausalLM` (RoPE, RMSNorm, SwiGLU, GQA), random init |
| size | 8 layers, hidden 384, FFN 1024, 6 heads / 2 KV heads, tied embeddings = **18.9M params** |
| tokenizer | byte-level BPE, 16,384 vocab, fit on the training corpus |
| context | 512 tokens |
| data | ~400M tokens (~21 tokens/param, ≈ Chinchilla-optimal) |
| compute | 6 × 18.9M × 400M ≈ 4.5 × 10¹⁶ FLOPs ≈ **3.6 h on an M4 Max (MPS)** |

All of it is set in `configs/tiny_20m.yaml`.

## Benchmark

- **Held-out bits/byte** — the main metric. Files held out before any
  training. Unlike perplexity, bits/byte is tokenizer-independent, so you can
  compare against any HF model (e.g. `HuggingFaceTB/SmolLM2-135M`).
- **A. Completion** (`eval/completion.py`) — continue a prefix; surface signal.
- **B. Fill-in-the-middle** (`eval/fim.py`) — prefix-continuation fallback, then execute.
- **C. Execute** (`eval/execute.py`) — solve a problem, run against hidden
  tests. At 20M params expect few passes; the problem set is tiny (8).

## Layout

```
data/     download.py  clean.py  deduplicate.py  tokenizer.py  common.py
model/    train.py
eval/     completion.py  fim.py  execute.py  run.py  compare.py  common_eval.py
          problems/{problems,completion,fim}.jsonl
configs/  tiny_20m.yaml  smoke.yaml
scripts/  run_pipeline.sh  smoke.sh
```

## Setup

```bash
uv sync
```

Training uses MPS (Apple GPU) or CUDA when available, CPU otherwise.

Data: accept the (instant) gate at
<https://huggingface.co/datasets/bigcode/starcoderdata>, then
`uv run hf auth login` with a read token.

## Run

Offline smoke test (stdlib as corpus, ~1 minute, no HF login):

```bash
bash scripts/smoke.sh
```

Full pipeline:

```bash
bash scripts/run_pipeline.sh
```

Or step by step:

```bash
uv run python data/download.py    configs/tiny_20m.yaml   # starcoderdata -> raw + held-out
uv run python data/clean.py       configs/tiny_20m.yaml   # drop generated/minified/unparseable
uv run python data/deduplicate.py configs/tiny_20m.yaml   # exact (+ optional MinHash) dedup
uv run python data/tokenizer.py   configs/tiny_20m.yaml   # fit BPE, encode to uint16 .bin
uv run python model/train.py      configs/tiny_20m.yaml   # -> runs/tiny_20m
uv run python eval/run.py         configs/tiny_20m.yaml --model runs/tiny_20m --name tiny_20m
uv run python eval/compare.py     configs/tiny_20m.yaml
```
