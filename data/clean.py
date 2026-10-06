"""Clean raw Python: drop generated/vendored/minified/broken files.

  - must parse as Python 3 (ast.parse) if clean.require_parse
  - reject generated/vendored markers, huge/tiny files, minified lines,
    low alphanumeric ratio, heavy non-ASCII

Usage
-----
  python data/clean.py configs/tiny_20m.yaml
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter

from tqdm import tqdm

from common import load_config, read_jsonl, resolve, write_jsonl


def alpha_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(c.isalnum() for c in s) / len(s)


def non_ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(ord(c) > 127 for c in s) / len(s)


def reason_reject(content: str, cfg: dict) -> str | None:
    c = cfg["clean"]
    lines = content.splitlines()
    n = len(lines)
    if n < c["min_lines"]:
        return "too_short"
    if n > c["max_lines"]:
        return "too_long"

    low = content.lower()
    for marker in c["drop_markers"]:
        if marker in low:
            return "generated_marker"

    max_len = max((len(l) for l in lines), default=0)
    if max_len > c["max_line_length"]:
        return "long_line"           # likely minified / data blob
    avg_len = len(content) / max(n, 1)
    if avg_len > c["max_avg_line_length"]:
        return "high_avg_line"       # likely minified
    if alpha_ratio(content) < c["min_alpha_ratio"]:
        return "low_alpha"
    if non_ascii_ratio(content) > c["max_non_ascii_ratio"]:
        return "non_ascii"

    if c["require_parse"]:
        try:
            ast.parse(content)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            return "parse_error"
    return None


def clean_corpus(cfg: dict) -> None:
    d = cfg["data"]
    src = resolve(d["raw_dir"]) / "python.jsonl"
    out = resolve(d["clean_dir"]) / "python.jsonl"
    if not src.exists():
        raise SystemExit(f"missing input {src}; run download.py first")

    stats: Counter = Counter()
    buf = []
    kept = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    # truncate output
    write_jsonl(out, [])
    for rec in tqdm(read_jsonl(src), desc="clean", unit="file"):
        content = rec.get("content", "")
        r = reason_reject(content, cfg)
        if r is not None:
            stats[r] += 1
            continue
        buf.append(rec)
        kept += 1
        stats["kept"] += 1
        if len(buf) >= 2000:
            write_jsonl(out, buf, append=True)
            buf.clear()
    if buf:
        write_jsonl(out, buf, append=True)

    total = sum(stats.values())
    print(f"\ncleaned: kept {kept}/{total} -> {out}")
    for k, v in stats.most_common():
        print(f"  {k:18s} {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    args = ap.parse_args()
    clean_corpus(load_config(args.config))


if __name__ == "__main__":
    main()
