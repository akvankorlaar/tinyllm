"""Download raw Python source from bigcode/starcoderdata.

starcoderdata is gated (instant accept) with inline content, already
cleaned/deduped/filtered upstream. Setup:
  1. accept the gate at https://huggingface.co/datasets/bigcode/starcoderdata
  2. uv run hf auth login

Usage
-----
  python data/download.py configs/tiny_20m.yaml

The stream is split BY REPOSITORY, so held-out files never share a repo with
training files (files in one repo are near-duplicates of each other in style,
names and boilerplate):
  data.heldout_dir : files from repos whose name hashes into
                     data.heldout_repo_fraction; up to eval_heldout_files of
                     them, at most heldout_files_per_repo per repo
  data.raw_dir     : files from all other repos, fed to clean.py
Files from held-out repos beyond those caps are dropped, never trained on.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm

from common import load_config, resolve, write_jsonl

# A char/token ratio good enough to decide when to STOP downloading.
# Real token counts are measured later by data/tokenize.py.
CHARS_PER_TOKEN = 3.6


def approx_tokens(n_chars: int) -> float:
    return n_chars / CHARS_PER_TOKEN


def is_heldout_repo(repo: str, fraction: float) -> bool:
    """Stable, order-independent repo split (same answer on every run)."""
    h = int.from_bytes(hashlib.sha256(repo.encode("utf-8")).digest()[:8], "big")
    return h / 2**64 < fraction


def iter_starcoderdata(language: str):
    # starcoderdata is organized into per-language subdirs; content is inline.
    ds = load_dataset(
        "bigcode/starcoderdata",
        data_dir=language.lower(),
        split="train",
        streaming=True,
    )
    for row in ds:
        content = row.get("content", "")
        if not content:
            continue
        yield {
            "content": content,
            "path": row.get("max_stars_repo_path", ""),
            "repo": row.get("max_stars_repo_name", ""),
            "source": "starcoderdata",
        }


def download(cfg: dict) -> None:
    d = cfg["data"]
    if d["source"] != "starcoderdata":
        sys.exit(f"Unknown source: {d['source']}")
    split(cfg, iter_starcoderdata(d["language"]))


def split(cfg: dict, records) -> None:
    """Write `records` to the held-out and raw files, split by repo."""
    d = cfg["data"]
    target = d["train_tokens"] * d["overshoot"]
    heldout_n = d["eval_heldout_files"]
    per_repo = d["heldout_files_per_repo"]
    fraction = d["heldout_repo_fraction"]

    raw_path = resolve(d["raw_dir"]) / "python.jsonl"
    heldout_path = resolve(d["heldout_dir"]) / "python_heldout.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    heldout_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(raw_path, [])  # truncate: re-running must not append duplicates

    got_tokens = 0.0
    held = 0
    kept = 0
    held_per_repo: Counter = Counter()
    raw_buf, held_buf = [], []
    pbar = tqdm(desc="python files", unit="file", mininterval=2.0)
    for rec in records:
        if not rec["content"].strip():
            continue
        if is_heldout_repo(rec["repo"], fraction):
            if held < heldout_n and held_per_repo[rec["repo"]] < per_repo:
                held_buf.append(rec)
                held += 1
                held_per_repo[rec["repo"]] += 1
        else:
            raw_buf.append(rec)
            kept += 1
            got_tokens += approx_tokens(len(rec["content"]))
        pbar.update(1)
        pbar.set_postfix(held=held, raw=kept, approx_tok=f"{got_tokens/1e6:.1f}M")
        if len(raw_buf) >= 2000:
            write_jsonl(raw_path, raw_buf, append=True)
            raw_buf.clear()
        if got_tokens >= target:
            break
    pbar.close()

    if held_buf:
        write_jsonl(heldout_path, held_buf)
    if raw_buf:
        write_jsonl(raw_path, raw_buf, append=True)

    print(f"held-out files : {held} from {len(held_per_repo)} repos  -> {heldout_path}")
    print(f"raw files      : {kept}  -> {raw_path}")
    print(f"approx tokens  : {got_tokens/1e6:.1f}M (target {target/1e6:.1f}M, pre-clean)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    args = ap.parse_args()
    download(load_config(args.config))


if __name__ == "__main__":
    main()
