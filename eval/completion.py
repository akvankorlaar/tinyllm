"""Benchmark A: next-token / line completion.

Give a prefix (e.g. the start of fibonacci); the model continues. We check the
generated text contains the expected continuation substrings (whitespace-
normalized) and that `prefix + generation` parses as valid Python.

This is a surface signal (does it continue plausibly), not correctness.
Correctness is benchmark C (execute.py).
"""
from __future__ import annotations

import ast
import re

from common_eval import generate
from tqdm import tqdm

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def run_completions(tok, model, cfg, items: list[dict]) -> dict:
    e = cfg["eval"]
    results = []
    n_pass = 0
    n_parse = 0
    for it in tqdm(items, desc="completion", unit="item"):
        gen = generate(tok, model, it["prompt"], e["max_new_tokens"], e["temperature"])
        gnorm = _norm(gen)
        hit = all(_norm(x) in gnorm for x in it["expect"])
        try:
            ast.parse(it["prompt"] + gen)
            parses = True
        except (SyntaxError, ValueError):
            parses = False
        n_pass += int(hit)
        n_parse += int(parses)
        results.append({"id": it["id"], "matched": hit, "parses": parses})
    n = len(items)
    return {
        "metric": "substring_match",
        "n": n,
        "matched": n_pass,
        "match_rate": n_pass / max(n, 1),
        "parse_rate": n_parse / max(n, 1),
        "per_item": results,
    }
