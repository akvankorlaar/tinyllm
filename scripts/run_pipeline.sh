#!/usr/bin/env bash
# Full pipeline, end to end. Run from the repo root.
#   bash scripts/run_pipeline.sh [config]   (default configs/tiny_10m.yaml)
# Assumes `uv sync` has been run and the starcoderdata gate is accepted
# (uv run hf auth login).
set -euo pipefail
CFG=${1:-configs/tiny_10m.yaml}
NAME=$(basename "$CFG" .yaml)

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
uv run python eval/run.py "$CFG" --model "runs/$NAME" --name "$NAME"

echo "==> 7. compare"
uv run python eval/compare.py "$CFG"
