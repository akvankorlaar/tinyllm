"""Select function-rich, documented Python files for a quality data mix.

Raw GitHub Python is dominated by imports, configs, framework boilerplate
and data literals. The benchmarks ask for a function body from a signature
and docstring, so this keeps files that look like that:

  - path is not boilerplate (migrations, settings, setup.py, __init__.py, ...)
  - has a function, and >= quality.min_func_frac of its lines are inside one
  - < quality.max_literal_frac of its lines are big constant literals
  - >= quality.min_control_per_100 if/for/while/comprehensions per 100 lines
  - >= quality.min_docstring_funcs functions with a docstring

Writes two shuffled token streams of quality.tokens each, encoded with the
existing tokenizer (no refit); point data.train_bin at one to train on it:
  quality.out_dir/quality.bin : files that pass the filter
  quality.out_dir/control.bin : files drawn at random, unfiltered (baseline)

Usage
-----
  python data/quality.py configs/tiny_10m_anneal_quality.yaml
"""
from __future__ import annotations

import argparse
import ast
import random
import re
import warnings
from collections import Counter
from multiprocessing import Pool

from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from common import load_config, read_jsonl, resolve, write_jsonl
from tokenizer import encode

BOILERPLATE_PATH = re.compile(
    r"(^|/)(migrations/|settings[^/]*\.py$|setup\.py$|__init__\.py$|conf\.py$"
    r"|urls\.py$|admin\.py$|apps\.py$|wsgi\.py$|asgi\.py$|manage\.py$)"
)
FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)
CONTROL = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.comprehension)


def features(src: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # invalid escapes in old code
        tree = ast.parse(src)
    lines = src.splitlines()
    nonblank = max(sum(1 for l in lines if l.strip()), 1)
    funcs = [n for n in ast.walk(tree) if isinstance(n, FUNC)]
    in_func: set[int] = set()
    for f in funcs:
        in_func.update(range(f.lineno, f.end_lineno + 1))
    func_lines = sum(1 for i, l in enumerate(lines, 1) if i in in_func and l.strip())
    literal_lines = 0
    for n in ast.walk(tree):
        span = n.end_lineno - n.lineno + 1 if hasattr(n, "end_lineno") and n.end_lineno else 0
        if isinstance(n, (ast.List, ast.Tuple, ast.Set)) and span >= 4:
            if all(isinstance(e, ast.Constant) for e in n.elts):
                literal_lines += span
        elif isinstance(n, ast.Dict) and span >= 4:
            if all(isinstance(v, ast.Constant) for v in n.values):
                literal_lines += span
        elif isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)) and span >= 20:
            literal_lines += span
    return {
        "funcs": len(funcs),
        "doc_funcs": sum(1 for f in funcs if ast.get_docstring(f)),
        "func_frac": func_lines / nonblank,
        "literal_frac": min(1.0, literal_lines / nonblank),
        "control_per_100": 100 * sum(isinstance(n, CONTROL) for n in ast.walk(tree)) / nonblank,
    }


def reason_reject(rec: dict, q: dict) -> str | None:
    if BOILERPLATE_PATH.search(rec.get("path", "")):
        return "boilerplate_path"
    try:
        f = features(rec["content"])
    except (SyntaxError, ValueError, RecursionError):
        return "parse_error"
    if f["funcs"] == 0 or f["func_frac"] < q["min_func_frac"]:
        return "few_functions"
    if f["literal_frac"] >= q["max_literal_frac"]:
        return "data_literals"
    if f["control_per_100"] < q["min_control_per_100"]:
        return "little_logic"
    if f["doc_funcs"] < q["min_docstring_funcs"]:
        return "no_docstrings"
    return None


def _judge(args):
    rec, q = args
    return reason_reject(rec, q)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    q = cfg["quality"]

    src = resolve(q["source"])
    recs = list(read_jsonl(src))
    with Pool() as pool:
        reasons = list(tqdm(pool.imap(_judge, ((r, q) for r in recs), chunksize=256),
                            total=len(recs), desc="score", unit="file"))
    stats = Counter(r or "kept" for r in reasons)
    print(f"quality: kept {stats['kept']}/{len(recs)} from {src}")
    for k, v in stats.most_common():
        print(f"  {k:18s} {v}")

    rng = random.Random(cfg["seed"])
    kept = [r for r, why in zip(recs, reasons) if why is None]
    control = recs[:]
    rng.shuffle(kept)
    rng.shuffle(control)

    out = resolve(q["out_dir"])
    tok = PreTrainedTokenizerFast.from_pretrained(str(resolve(cfg["tokenizer"]["dir"])))
    for name, rows in (("quality", kept), ("control", control)):
        jsonl = out / f"{name}.jsonl"
        write_jsonl(jsonl, rows)
        n = encode(tok, jsonl, out / f"{name}.bin", q["tokens"])
        print(f"{name}: {n/1e6:.1f}M tokens (budget {q['tokens']/1e6:.1f}M) -> {out / name}.bin")
        if n < q["tokens"]:
            print("  WARNING: ran out of files before the budget")


if __name__ == "__main__":
    main()
