"""Download raw Python source from bigcode/starcoderdata.

starcoderdata is gated (instant accept) with inline content, already
cleaned/deduped/filtered upstream. Setup:
  1. accept the gate at https://huggingface.co/datasets/bigcode/starcoderdata
  2. uv run hf auth login

Usage
-----
  python data/download.py configs/tiny_20m.yaml

The stream is split into:
  data.heldout_dir : first `eval_heldout_files` files, held out before training
  data.raw_dir     : the rest, fed to clean.py
"""
from __future__ import annotations

import argparse
import sys

from datasets import load_dataset
from tqdm import tqdm

from common import load_config, resolve, write_jsonl

# A char/token ratio good enough to decide when to STOP downloading.
# Real token counts are measured later by data/tokenize.py.
CHARS_PER_TOKEN = 3.6


def approx_tokens(n_chars: int) -> float:
    return n_chars / CHARS_PER_TOKEN


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
    target = d["train_tokens"] * d["overshoot"]
    heldout_n = d["eval_heldout_files"]

    raw_path = resolve(d["raw_dir"]) / "python.jsonl"
    heldout_path = resolve(d["heldout_dir"]) / "python_heldout.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    heldout_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(raw_path, [])  # truncate: re-running must not append duplicates

    got_tokens = 0.0
    held = 0
    kept = 0
    raw_buf, held_buf = [], []
    pbar = tqdm(desc="python files", unit="file", mininterval=2.0)
    for rec in iter_starcoderdata(d["language"]):
        if not rec["content"].strip():
            continue
        if held < heldout_n:
            held_buf.append(rec)
            held += 1
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

    print(f"held-out files : {held}  -> {heldout_path}")
    print(f"raw files      : {kept}  -> {raw_path}")
    print(f"approx tokens  : {got_tokens/1e6:.1f}M (target {target/1e6:.1f}M, pre-clean)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    args = ap.parse_args()
    download(load_config(args.config))


if __name__ == "__main__":
    main()
