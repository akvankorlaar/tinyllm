"""Benchmark B: fill-in-the-middle.

If the tokenizer exposes FIM sentinels (<fim_prefix>/<fim_suffix>/<fim_middle>),
we do real infilling. Our tokenizer has no FIM sentinels (training is plain
causal LM), so we fall back to prefix-continuation: generate from the prefix,
then assemble `prefix + generation + suffix` and execute. The fallback is
weaker by design.

Each item: prefix, suffix, hidden tests. Assembled program is run sandboxed.
"""
from __future__ import annotations

from common_eval import extract_code, generate, run_program
from tqdm import tqdm

FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<fim_prefix>", "<fim_suffix>", "<fim_middle>"


def _has_fim(tok) -> bool:
    vocab = tok.get_vocab()
    return all(t in vocab for t in (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE))


def run_fim(tok, model, cfg, items: list[dict]) -> dict:
    e = cfg["eval"]
    use_fim = _has_fim(tok)
    mode = "native_fim" if use_fim else "prefix_fallback"
    results = []
    n_pass = 0
    for it in tqdm(items, desc=f"fim[{mode}]", unit="item"):
        if use_fim:
            prompt = f"{FIM_PREFIX}{it['prefix']}{FIM_SUFFIX}{it['suffix']}{FIM_MIDDLE}"
            middle = generate(tok, model, prompt, e["max_new_tokens"], e["temperature"])
            program = f"{it['prefix']}{extract_code(middle)}{it['suffix']}\n\n{it['tests']}\n"
        else:
            middle = generate(tok, model, it["prefix"], e["max_new_tokens"], e["temperature"])
            program = f"{it['prefix']}{middle}\n{it['suffix']}\n\n{it['tests']}\n"
        ok, detail = run_program(program, e["timeout_sec"])
        n_pass += int(ok)
        results.append({"id": it["id"], "passed": ok, "detail": detail})
    n = len(items)
    return {
        "metric": "fim_pass@1",
        "mode": mode,
        "n": n,
        "passed": n_pass,
        "score": n_pass / max(n, 1),
        "per_item": results,
    }
