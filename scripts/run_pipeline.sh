#!/usr/bin/env bash
# Experiment 1, end to end. Run from the repo root.
#   bash scripts/run_pipeline.sh
# Assumes `uv sync` has been run and (for the-stack-v2-dedup) HF gate accepted
# + AWS creds configured. Training needs a GPU; the data/eval steps do not.
set -euo pipefail
CFG=configs/experiment_001.yaml

echo "==> 1. download Python corpus (+ held-out eval split)"
uv run python data/download.py "$CFG" --corpus python

echo "==> 2. download general corpus (control for model B)"
uv run python data/download.py "$CFG" --corpus general

echo "==> 3. clean"
uv run python data/clean.py "$CFG" --corpus python
uv run python data/clean.py "$CFG" --corpus general

echo "==> 4. deduplicate"
uv run python data/deduplicate.py "$CFG" --corpus python
uv run python data/deduplicate.py "$CFG" --corpus general

echo "==> 5. train C (python) and B (general)   [needs GPU]"
uv run python model/train.py "$CFG" --variant python
uv run python model/train.py "$CFG" --variant general

echo "==> 6. evaluate A (base), B (general), C (python)"
uv run python eval/run.py "$CFG" --model HuggingFaceTB/SmolLM2-135M --name A_base
uv run python eval/run.py "$CFG" --model runs/experiment_001_general   --name B_general
uv run python eval/run.py "$CFG" --model runs/experiment_001_python    --name C_python

echo "==> 7. compare"
uv run python eval/compare.py "$CFG"
