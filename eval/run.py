"""Run the full benchmark on ONE model, write a results JSON.

  python eval/run.py configs/tiny_20m.yaml --model runs/tiny_20m --name tiny_20m
  python eval/run.py configs/tiny_20m.yaml \
      --model HuggingFaceTB/SmolLM2-135M --name smollm2_135m   # reference point

Benchmarks: A completion, B fim, C execute, plus held-out perplexity and
bits-per-byte. Perplexity depends on the tokenizer; bits-per-byte does not,
so use bpb to compare models with different tokenizers.
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
    ap.add_argument("--name", required=True, help="label, e.g. A_base / C_python")
    args = ap.parse_args()

    cfg = load_config(args.config)
    e = cfg["eval"]
    tok, model = load(args.model)

    problems = load_items(resolve(e["problems_file"]))
    completions = load_items(resolve(e["completion_file"]))
    fims = load_items(resolve(e["fim_file"]))

    summary = {
        "name": args.name,
        "model": args.model,
        "device": device(),
        "perplexity": heldout_perplexity(tok, model, cfg),
        "A_completion": run_completions(tok, model, cfg, completions) if completions else None,
        "B_fim": run_fim(tok, model, cfg, fims) if fims else None,
        "C_execute": run_problems(tok, model, cfg, problems) if problems else None,
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
    if summary["C_execute"]:
        c = summary["C_execute"]
        print(f"C execute  {c['metric']} : {c['passed']}/{c['n']} = {c['score']:.3f}")
    if summary["A_completion"]:
        a = summary["A_completion"]
        print(f"A complete match_rate: {a['match_rate']:.3f}  parse_rate {a['parse_rate']:.3f}")
    if summary["B_fim"]:
        b = summary["B_fim"]
        print(f"B fim[{b['mode']}] : {b['passed']}/{b['n']} = {b['score']:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
