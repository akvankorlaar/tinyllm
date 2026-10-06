"""Build the benchmark files in eval/problems/ from public sources.

  python eval/build_benchmark.py

Writes (all committed; rerun only to rebuild):
  humaneval.jsonl : OpenAI HumanEval, 164 problems (MIT)
  mbpp.jsonl      : Google MBPP sanitized, 427 problems (CC-BY-4.0), converted
                    from a text prompt to a signature + docstring prompt
  fim.jsonl       : single-line infilling: one line of each reference solution
                    (handwritten + humaneval + mbpp) blanked out

Every problem has the same fields: id, prompt, solution, entry_point, tests.
Each reference `prompt + solution + tests` is executed; problems whose
reference fails are dropped. Both sources are on GitHub (no Hugging Face), and
bigcode decontaminated starcoderdata against both.
"""
from __future__ import annotations

import ast
import gzip
import json
import random
import re
import sys
import textwrap
import urllib.request
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from common import read_jsonl, resolve, write_jsonl  # noqa: E402
from common_eval import run_program  # noqa: E402

HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
MBPP_URL = ("https://raw.githubusercontent.com/google-research/google-research/"
            "master/mbpp/sanitized-mbpp.json")
PROBLEMS = REPO_ROOT / "eval" / "problems"
CACHE = resolve("data/benchmark_src")


def fetch(url: str) -> bytes:
    dest = CACHE / url.rsplit("/", 1)[-1]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
    return dest.read_bytes()


def humaneval() -> list[dict]:
    rows = [json.loads(l) for l in gzip.decompress(fetch(HUMANEVAL_URL)).decode().splitlines() if l]
    return [{
        "id": r["task_id"].replace("HumanEval/", "humaneval/"),
        "prompt": r["prompt"],
        "solution": r["canonical_solution"],
        "entry_point": r["entry_point"],
        "tests": f"{r['test']}\ncheck({r['entry_point']})\n",
    } for r in rows]


def mbpp_problem(r: dict) -> dict | None:
    """Rewrite an MBPP row as signature + docstring prompt and a 4-space body."""
    try:
        with warnings.catch_warnings():  # MBPP has non-raw regex strings
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(r["code"])
    except SyntaxError:
        return None
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    called = set(re.findall(r"\b(\w+)\s*\(", r["test_list"][0]))
    target = next((n for n in reversed(defs) if n.name in called), None)
    if target is None:
        return None
    # Everything else at module level (imports, helper functions) is context.
    preamble = [ast.unparse(n) for n in tree.body if n is not target]
    preamble += [imp for imp in r["test_imports"] if imp not in preamble]
    body = target.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]  # drop an existing docstring; ours replaces it
    if not body:
        return None
    doc = "\n    ".join(textwrap.wrap(r["prompt"].strip(), 72))
    prompt = "\n".join(preamble) + ("\n\n\n" if preamble else "")
    prompt += (f"def {target.name}({ast.unparse(target.args)}):\n"
               f'    """{doc}\n    {r["test_list"][0]}\n    """\n')
    solution = textwrap.indent("\n".join(ast.unparse(s) for s in body), "    ") + "\n"
    tests = "\n".join(r["test_imports"] + r["test_list"]) + "\n"
    return {"id": f"mbpp/{r['task_id']}", "prompt": prompt, "solution": solution,
            "entry_point": target.name, "tests": tests}


def mbpp() -> list[dict]:
    rows = json.loads(fetch(MBPP_URL))
    out = [p for p in map(mbpp_problem, rows) if p is not None]
    print(f"mbpp: converted {len(out)}/{len(rows)}")
    return out


def verified(problems: list[dict]) -> list[dict]:
    ok = []
    for p in problems:
        passed, detail = run_program(f"{p['prompt']}{p['solution']}\n\n{p['tests']}", 10)
        if passed:
            ok.append(p)
        else:
            print(f"  drop {p['id']}: reference fails ({detail})")
    return ok


def fim_items(problems: list[dict], rng: random.Random) -> list[dict]:
    """Blank one non-trivial line of each reference solution."""
    items = []
    for p in problems:
        lines = p["solution"].splitlines(keepends=True)
        candidates = [i for i, l in enumerate(lines) if len(l.strip()) >= 8]
        if not candidates:
            continue
        i = rng.choice(candidates)
        items.append({
            "id": f"fim/{p['id']}",
            "prefix": p["prompt"] + "".join(lines[:i]),
            "middle": lines[i],
            "suffix": "".join(lines[i + 1:]),
            "tests": p["tests"],
        })
    return items


def main() -> None:
    hand = list(read_jsonl(PROBLEMS / "handwritten.jsonl"))
    he, mb = verified(humaneval()), verified(mbpp())
    write_jsonl(PROBLEMS / "humaneval.jsonl", he)
    write_jsonl(PROBLEMS / "mbpp.jsonl", mb)
    fim = fim_items(verified(hand) + he + mb, random.Random(0))
    write_jsonl(PROBLEMS / "fim.jsonl", fim)
    print(f"humaneval {len(he)}  mbpp {len(mb)}  fim {len(fim)}  -> {PROBLEMS}")


if __name__ == "__main__":
    main()
