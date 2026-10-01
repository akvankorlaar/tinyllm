"""Benchmark C (the important one): solve real problems, execute against tests.

Each problem gives a function signature + docstring as the prompt. The model
completes the body; we run `prompt + completion` plus hidden asserts in a
sandboxed subprocess. Syntactically pretty code that does not run = fail.

Reports pass@1 (or pass@k via eval.num_samples).
"""
from __future__ import annotations

from common_eval import extract_code, generate, run_program
from tqdm import tqdm


def _candidate_program(prompt: str, completion: str, tests: str) -> str:
    body = extract_code(completion) if "```" in completion else completion
    return f"{prompt}{body}\n\n{tests}\n"


def run_problems(tok, model, cfg, problems: list[dict]) -> dict:
    e = cfg["eval"]
    k = e.get("num_samples", 1)
    results = []
    n_pass = 0
    for p in tqdm(problems, desc="execute", unit="prob"):
        passed = False
        detail = "no_sample"
        for _ in range(k):
            comp = generate(tok, model, p["prompt"],
                            e["max_new_tokens"], e["temperature"])
            prog = _candidate_program(p["prompt"], comp, p["tests"])
            ok, detail = run_program(prog, e["timeout_sec"])
            if ok:
                passed = True
                break
        n_pass += int(passed)
        results.append({"id": p["id"], "passed": passed, "detail": detail})
    return {
        "metric": "pass@1" if k == 1 else f"pass@{k}",
        "n": len(problems),
        "passed": n_pass,
        "score": n_pass / max(len(problems), 1),
        "per_problem": results,
    }
