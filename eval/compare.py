"""Compare model result JSONs and print a markdown table.

  python eval/compare.py configs/tiny_20m.yaml
  python eval/compare.py configs/tiny_20m.yaml tiny_20m smollm2_135m

With no names given, compares every *.json in eval.results_dir.
Compare held-out quality across tokenizers with bits/byte (lower is better);
perplexity is only comparable between models sharing a tokenizer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
from common import load_config, resolve  # noqa: E402


def _fmt(x, pct=False, digits=2):
    if x is None:
        return "-"
    return f"{x*100:.1f}%" if pct else f"{x:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("names", nargs="*", help="result labels; default = all")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rdir = resolve(cfg["eval"]["results_dir"])
    if args.names:
        files = [rdir / f"{n}.json" for n in args.names]
    else:
        files = sorted(rdir.glob("*.json"))
    rows = [json.loads(p.read_text()) for p in files if p.exists()]
    if not rows:
        raise SystemExit(f"no result JSONs in {rdir}; run eval/run.py first")

    suites = list(cfg["eval"]["suites"])
    cols = (["heldout bits/byte", "solution bits/byte"] + [f"{s} pass@1" for s in suites]
            + ["fim pass@1", "completion match", "heldout ppl"])
    print("| model | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for r in rows:
        ex = r.get("execute") or {}
        pp = r.get("perplexity") or {}
        name = r["name"] + (f" (limit {r['limit']})" if r.get("limit") else "")
        vals = ([_fmt(pp.get("bits_per_byte"), digits=3),
                 _fmt(r.get("solution_bits_per_byte"), digits=3)]
                + [_fmt((ex.get(s) or {}).get("score"), pct=True) for s in suites]
                + [_fmt((r.get("fim") or {}).get("score"), pct=True),
                   _fmt((r.get("completion") or {}).get("match_rate"), pct=True),
                   _fmt(pp.get("perplexity"))])
        print(f"| {name} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
