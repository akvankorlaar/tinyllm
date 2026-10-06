"""Benchmark B: single-line fill-in-the-middle.

Each item blanks one line of a reference solution (eval/build_benchmark.py):
prefix, the hidden middle line, suffix, tests. The model writes ONE line; we
assemble `prefix + line + suffix` and execute it against the tests.

If the tokenizer exposes FIM sentinels (<fim_prefix>/<fim_suffix>/<fim_middle>)
we prompt with the suffix too. Our tokenizer has none (training is plain
causal LM), so the model sees only the prefix: a weaker task by design.
"""
from __future__ import annotations

from common_eval import generate, run_program
from tqdm import tqdm

FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<fim_prefix>", "<fim_suffix>", "<fim_middle>"


def _has_fim(tok) -> bool:
    vocab = tok.get_vocab()
    return all(t in vocab for t in (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE))


def _first_line(text: str) -> str:
    return text.split("\n", 1)[0] + "\n"


def run_fim(tok, model, cfg, items: list[dict]) -> dict:
    e = cfg["eval"]
    use_fim = _has_fim(tok)
    mode = "native_fim" if use_fim else "prefix_only"
    results = []
    n_pass = 0
    for it in tqdm(items, desc=f"fim[{mode}]", unit="item"):
        if use_fim:
            prompt = f"{FIM_PREFIX}{it['prefix']}{FIM_SUFFIX}{it['suffix']}{FIM_MIDDLE}"
        else:
            prompt = it["prefix"]
        gen = generate(tok, model, prompt, e["max_new_tokens"], e["temperature"],
                       stop=lambda t: "\n" in t)
        line = _first_line(gen)
        ok, detail = run_program(f"{it['prefix']}{line}{it['suffix']}\n\n{it['tests']}\n",
                                 e["timeout_sec"])
        n_pass += int(ok)
        results.append({"id": it["id"], "passed": ok, "exact": line.strip() == it["middle"].strip(),
                        "detail": detail})
    n = len(items)
    return {
        "metric": "fim_pass@1",
        "mode": mode,
        "n": n,
        "passed": n_pass,
        "score": n_pass / max(n, 1),
        "exact_match": sum(r["exact"] for r in results) / max(n, 1),
        "per_item": results,
    }
