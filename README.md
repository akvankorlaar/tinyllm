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

599 executable problems (`eval/problems/`, sources and licenses in
[SOURCES.md](eval/problems/SOURCES.md)): HumanEval (164), MBPP sanitized
(427, rewritten as signature + docstring prompts) and 8 handwritten.

| Metric | What it measures |
|---|---|
| **held-out bits/byte** | how well the model predicts Python it never trained on. Files come from held-out *repositories*, so no repo is in both train and eval. |
| **solution bits/byte** | how well the model predicts each problem's reference solution given its prompt. Near-zero pass rates make this the most useful task metric at 20M. |
| **pass@1** per suite | completion cut at the end of the function body, executed against hidden tests in a resource-limited subprocess |
| **fim pass@1** | one line of a reference solution blanked; the model writes it from the prefix (our tokenizer has no FIM tokens); executed. ~1% passes by luck. |
| completion match | surface check on 4 prefixes |

Bits/byte does not depend on the tokenizer, so compare against any HF model
(e.g. `HuggingFaceTB/SmolLM2-135M`) with it; perplexity only between models
that share a tokenizer. A full eval takes ~2 minutes; `--limit N` runs N items
per suite. Rebuild the problem files with `eval/build_benchmark.py` (fetches
from GitHub).

## Layout

```
data/     download.py  clean.py  deduplicate.py  tokenizer.py  common.py
model/    train.py
eval/     run.py  compare.py  execute.py  fim.py  completion.py  common_eval.py
          build_benchmark.py
          problems/{handwritten,humaneval,mbpp,fim,completion}.jsonl
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

## Training on a remote GPU

The pipeline is the same on a Linux CUDA box; the PyPI `torch` wheel there
already includes CUDA.

```bash
git clone https://github.com/akvankorlaar/tinyllm.git && cd tinyllm
uv sync
uv run hf auth login                  # starcoderdata gate
bash scripts/run_pipeline.sh          # download -> ... -> train -> eval
```

- Set `train.torch_compile: true` for extra speed on CUDA.
- On a big GPU, raise `train.per_device_batch_size` and lower `grad_accum` to
  keep the same tokens per step (batch x accum x seq_len, now 131k).
- Checkpoints are written every `save_steps`; after an interruption, continue
  with `uv run python model/train.py configs/tiny_20m.yaml --resume`.
- Bring back `runs/tiny_20m/` (the final model + tokenizer, ~80 MB) and
  `eval/results/`; checkpoints in `runs/tiny_20m/checkpoint-*` can stay.
