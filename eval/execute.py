"""Benchmark C (the important one): solve real problems, execute against tests.

Each problem gives a function signature + docstring as the prompt. The model
completes the body (cut at the first non-indented line); we run
`prompt + completion` plus hidden asserts in a sandboxed subprocess.
Syntactically pretty code that does not run = fail. Reports pass@1 (or pass@k
via eval.num_samples).

Also reports solution bits/byte: how well the model predicts the reference
solution given the prompt. Pass rates sit near zero for tiny models; this
stays informative and is comparable across tokenizers.
"""
from __future__ import annotations

from common_eval import continuation_bits, end_of_body, generate, run_program
from tqdm import tqdm


def _body_done(text: str) -> bool:
    return end_of_body(text) < len(text)


def run_problems(tok, model, cfg, problems: list[dict], desc: str = "execute") -> dict:
    e = cfg["eval"]
    k = e.get("num_samples", 1)
    results = []
    n_pass = 0
    bits = n_bytes = 0
    for p in tqdm(problems, desc=desc, unit="prob"):
        passed = False
        detail = "no_sample"
        for _ in range(k):
            comp = generate(tok, model, p["prompt"], e["max_new_tokens"],
                            e["temperature"], stop=_body_done)
            body = comp[: end_of_body(comp)]
            ok, detail = run_program(f"{p['prompt']}{body}\n\n{p['tests']}\n", e["timeout_sec"])
            if ok:
                passed = True
                break
        b, nb = continuation_bits(tok, model, p["prompt"], p["solution"])
        bits += b
        n_bytes += nb
        n_pass += int(passed)
        results.append({"id": p["id"], "passed": passed, "detail": detail,
                        "solution_bpb": b / nb})
    return {
        "metric": "pass@1" if k == 1 else f"pass@{k}",
        "n": len(problems),
        "passed": n_pass,
        "score": n_pass / max(len(problems), 1),
        "solution_bits_per_byte": bits / max(n_bytes, 1),
        "solution_bits": bits,
        "solution_bytes": n_bytes,
        "per_problem": results,
    }
