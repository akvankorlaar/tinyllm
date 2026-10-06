#!/usr/bin/env bash
# Offline end-to-end smoke test: seeds data/smoke/raw with this Python's own
# stdlib source (no Hugging Face login needed), then runs every pipeline step
# with configs/smoke.yaml. Takes a few minutes. Run from the repo root.
set -euo pipefail
CFG=configs/smoke.yaml
rm -rf data/smoke

echo "==> seed raw corpus from the Python stdlib"
uv run python - <<'PY'
import json, pathlib, sysconfig
src = pathlib.Path(sysconfig.get_paths()["stdlib"])
out = pathlib.Path("data/smoke/raw/python.jsonl")
held = pathlib.Path("data/smoke/eval_heldout/python_heldout.jsonl")
out.parent.mkdir(parents=True); held.parent.mkdir(parents=True)
files = sorted(p for p in src.rglob("*.py") if "site-packages" not in p.parts)
with out.open("w") as fo, held.open("w") as fh:
    for i, p in enumerate(files):
        rec = {"content": p.read_text(errors="ignore"), "path": str(p.relative_to(src)),
               "repo": "stdlib", "source": "stdlib"}
        (fh if i % 20 == 0 else fo).write(json.dumps(rec) + "\n")
print(f"seeded {len(files)} files")
PY

uv run python data/clean.py       "$CFG"
uv run python data/deduplicate.py "$CFG"
uv run python data/tokenizer.py   "$CFG"
uv run python model/train.py      "$CFG"
uv run python eval/run.py         "$CFG" --model runs/smoke --name smoke
uv run python eval/compare.py     "$CFG"
