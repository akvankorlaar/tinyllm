#!/usr/bin/env bash
# Full pipeline, end to end. Run from the repo root.
#   bash scripts/run_pipeline.sh
# Assumes `uv sync` has been run and the starcoderdata gate is accepted
# (uv run hf auth login).
set -euo pipefail
CFG=configs/tiny_20m.yaml

echo "==> 1. download Python corpus (+ held-out eval split)"
uv run python data/download.py "$CFG"

echo "==> 2. clean"
uv run python data/clean.py "$CFG"

echo "==> 3. deduplicate"
uv run python data/deduplicate.py "$CFG"

echo "==> 4. fit tokenizer + encode corpus"
uv run python data/tokenizer.py "$CFG"

echo "==> 5. train"
uv run python model/train.py "$CFG"

echo "==> 6. evaluate"
uv run python eval/run.py "$CFG" --model runs/tiny_20m --name tiny_20m

echo "==> 7. compare"
uv run python eval/compare.py "$CFG"
