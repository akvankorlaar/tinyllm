"""Deduplicate cleaned files: exact (sha256) then near-dup (MinHash LSH).

Exact: hash normalized content (whitespace-collapsed). Drops identical files.
Near : 5-gram MinHash + LSH at dedup.near_threshold Jaccard. Drops files that
       are near-copies of one already kept. Needs `datasketch`; if missing,
       near-dedup is skipped with a warning.

Usage
-----
  python data/deduplicate.py configs/experiment_001.yaml --corpus python
"""
from __future__ import annotations

import argparse
import hashlib
import re

from tqdm import tqdm

from common import load_config, read_jsonl, resolve, write_jsonl

_WS = re.compile(r"\s+")


def norm(content: str) -> str:
    return _WS.sub(" ", content).strip()


def shingles(content: str, k: int = 5) -> set[str]:
    toks = norm(content).split(" ")
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def dedup(cfg: dict, corpus: str) -> None:
    d = cfg["data"]
    dd = cfg["dedup"]
    name = "python.jsonl" if corpus == "python" else "general.jsonl"
    src = resolve(d["clean_dir"]) / name
    out = resolve(d["dedup_dir"]) / name
    if not src.exists():
        raise SystemExit(f"missing input {src}; run clean.py --corpus {corpus} first")

    # --- exact ---
    seen: set[str] = set()
    staged = []
    n_in = 0
    for rec in tqdm(read_jsonl(src), desc="exact", unit="file"):
        n_in += 1
        h = hashlib.sha256(norm(rec["content"]).encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        staged.append(rec)
    n_exact = len(staged)

    # --- near ---
    kept = staged
    if dd.get("near"):
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            print("datasketch not installed; skipping near-dedup "
                  "(uv add datasketch to enable)")
        else:
            lsh = MinHashLSH(threshold=dd["near_threshold"], num_perm=dd["num_perm"])
            kept = []
            for i, rec in enumerate(tqdm(staged, desc="near", unit="file")):
                m = MinHash(num_perm=dd["num_perm"])
                for sh in shingles(rec["content"]):
                    m.update(sh.encode("utf-8"))
                if lsh.query(m):
                    continue
                lsh.insert(str(i), m)
                kept.append(rec)

    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, kept)
    print(f"\ndedup {corpus}: in {n_in} -> exact {n_exact} -> near {len(kept)}")
    print(f"  -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--corpus", choices=["python", "general"], default="python")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dedup(cfg, args.corpus)


if __name__ == "__main__":
    main()
