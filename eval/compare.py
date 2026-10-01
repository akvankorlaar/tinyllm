"""Compare model result JSONs (A / B / C) and print a markdown table.

  python eval/compare.py configs/experiment_001.yaml
  python eval/compare.py configs/experiment_001.yaml A_base B_general C_python

With no names given, compares every *.json in eval.results_dir.
Verdict: C > B > A means Python specialization helped beyond the token budget.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
from common import load_config, resolve  # noqa: E402


def _fmt(x, pct=False):
    if x is None:
        return "-"
    return f"{x*100:.1f}%" if pct else f"{x:.2f}"


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

    header = "| model | execute pass@1 | completion match | fim pass@1 | heldout ppl |"
    sep =    "|---|---|---|---|---|"
    print(header)
    print(sep)
    for r in rows:
        c = r.get("C_execute") or {}
        a = r.get("A_completion") or {}
        b = r.get("B_fim") or {}
        pp = r.get("perplexity") or {}
        print(f"| {r['name']} "
              f"| {_fmt(c.get('score'), pct=True)} "
              f"| {_fmt(a.get('match_rate'), pct=True)} "
              f"| {_fmt(b.get('score'), pct=True)} "
              f"| {_fmt(pp.get('perplexity'))} |")


if __name__ == "__main__":
    main()
