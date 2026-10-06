"""Run the full benchmark on ONE model, write a results JSON.

  python eval/run.py configs/tiny_20m.yaml --model runs/tiny_20m --name tiny_20m
  python eval/run.py configs/tiny_20m.yaml \
      --model HuggingFaceTB/SmolLM2-135M --name smollm2_135m   # reference point

Benchmarks:
  - held-out perplexity and bits/byte on Python never trained on
  - execute: pass@1 on each suite in eval.suites (handwritten, humaneval,
    mbpp), plus bits/byte of each reference solution given its prompt
  - fim: single-line infilling, executed
  - completion: continue a prefix; surface signal
Perplexity depends on the tokenizer; bits/byte does not, so use bits/byte to
compare models with different tokenizers. --limit N runs N items per suite.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

import torch  # noqa: E402

from common import load_config, read_jsonl, resolve  # noqa: E402
from common_eval import device, load  # noqa: E402
from completion import run_completions  # noqa: E402
from execute import run_problems  # noqa: E402
from fim import run_fim  # noqa: E402


def load_items(path: Path) -> list[dict]:
    return list(read_jsonl(path)) if path.exists() else []


@torch.no_grad()
def heldout_perplexity(tok, model, cfg) -> dict:
    d = cfg["data"]
    hp = resolve(d["heldout_dir"]) / "python_heldout.jsonl"
    if not hp.exists():
        return {"available": False}
    seq_len = cfg["model"]["seq_len"]
    max_tokens = cfg["eval"]["perplexity_tokens"]
    eos = tok.eos_token_id
    buf: list[int] = []
    total_nll = 0.0
    total_tok = 0
    seen_bytes = seen_tok = 0  # for this tokenizer's bytes/token on held-out text
    for rec in read_jsonl(hp):
        ids = tok(rec["content"], add_special_tokens=False)["input_ids"] + [eos]
        seen_bytes += len(rec["content"].encode("utf-8"))
        seen_tok += len(ids)
        buf.extend(ids)
        while len(buf) >= seq_len and total_tok < max_tokens:
            block = torch.tensor([buf[:seq_len]], device=device())
            buf = buf[seq_len:]
            out = model(block, labels=block)
            total_nll += out.loss.item() * (seq_len - 1)
            total_tok += seq_len - 1
        if total_tok >= max_tokens:
            break
    if total_tok == 0:
        return {"available": False}
    nll = total_nll / total_tok
    return {"available": True, "tokens": total_tok, "perplexity": math.exp(nll),
            "bits_per_byte": nll / math.log(2) * seen_tok / seen_bytes}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--model", required=True, help="HF id or local run dir")
    ap.add_argument("--name", required=True, help="label, e.g. tiny_20m")
    ap.add_argument("--limit", type=int, default=None, help="max items per suite (quick runs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    e = cfg["eval"]
    tok, model = load(args.model)

    def items(path: str) -> list[dict]:
        return load_items(resolve(path))[: args.limit]

    execute = {}
    for suite, path in e["suites"].items():
        problems = items(path)
        if problems:
            execute[suite] = run_problems(tok, model, cfg, problems, desc=suite)
    fims = items(e["fim_file"])
    completions = items(e["completion_file"])

    summary = {
        "name": args.name,
        "model": args.model,
        "device": device(),
        "limit": args.limit,
        "perplexity": heldout_perplexity(tok, model, cfg),
        "solution_bits_per_byte": (
            sum(r["solution_bits"] for r in execute.values())
            / max(sum(r["solution_bytes"] for r in execute.values()), 1)
        ) if execute else None,
        "execute": execute,
        "fim": run_fim(tok, model, cfg, fims) if fims else None,
        "completion": run_completions(tok, model, cfg, completions) if completions else None,
    }

    out_dir = resolve(e["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.name}.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n=== {args.name} ({args.model}) ===")
    pp = summary["perplexity"]
    if pp.get("available"):
        print(f"held-out perplexity : {pp['perplexity']:.2f}  "
              f"bits/byte {pp['bits_per_byte']:.3f}  ({pp['tokens']} tok)")
    if summary["solution_bits_per_byte"] is not None:
        print(f"solution bits/byte  : {summary['solution_bits_per_byte']:.3f}")
    for suite, c in execute.items():
        print(f"execute {suite:11s} {c['metric']}: {c['passed']}/{c['n']} = {c['score']:.3f}")
    if summary["fim"]:
        b = summary["fim"]
        print(f"fim[{b['mode']}] : {b['passed']}/{b['n']} = {b['score']:.3f}  "
              f"exact {b['exact_match']:.3f}")
    if summary["completion"]:
        a = summary["completion"]
        print(f"completion match_rate: {a['match_rate']:.3f}  parse_rate {a['parse_rate']:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
