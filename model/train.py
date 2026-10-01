"""Continued pretraining of SmolLM2-135M (causal LM) for Experiment 1.

Same script trains both trained variants, selected by --variant:
  python   -> model C (treatment), data = data/dedup/python.jsonl
  general  -> model B (control),   data = data/dedup/general.jsonl
Model A is the untouched base (no training; eval only).

Entry point:
  python model/train.py configs/experiment_001.yaml --variant python

Deliberately boring (per the Experiment 1 brief): stock tokenizer, stock
architecture, plain causal-LM loss. No FIM, no RL, no distillation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from common import load_config, read_jsonl, resolve  # noqa: E402


def pick_precision(requested: str) -> tuple[bool, bool]:
    """Return (bf16, fp16) flags honoring hardware support."""
    if not torch.cuda.is_available():
        return False, False  # CPU: fp32
    if requested == "bf16" and torch.cuda.is_bf16_supported():
        return True, False
    return False, True  # fall back to fp16 on GPU


def build_packed_dataset(path: Path, tokenizer, seq_len: int, token_budget: int) -> Dataset:
    """Tokenize docs, separate with EOS, pack into fixed seq_len blocks.

    Stops once `token_budget` tokens have been consumed.
    """
    eos = tokenizer.eos_token_id
    buf: list[int] = []
    blocks: list[list[int]] = []
    used = 0
    pbar = tqdm(desc="packing", unit="tok", total=token_budget)
    for rec in read_jsonl(path):
        ids = tokenizer(rec["content"], add_special_tokens=False)["input_ids"]
        ids.append(eos)
        buf.extend(ids)
        used += len(ids)
        pbar.update(len(ids))
        while len(buf) >= seq_len:
            blocks.append(buf[:seq_len])
            buf = buf[seq_len:]
        if used >= token_budget:
            break
    pbar.close()
    tokens_packed = len(blocks) * seq_len
    print(f"packed {len(blocks)} blocks x {seq_len} = {tokens_packed/1e6:.2f}M tokens "
          f"(budget {token_budget/1e6:.1f}M)")
    if not blocks:
        raise SystemExit(f"no blocks packed from {path}; is it empty?")
    return Dataset.from_dict({"input_ids": blocks})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--variant", choices=["python", "general"], default=None,
                    help="overrides train.variant in the config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    variant = args.variant or cfg["train"]["variant"]
    set_seed(cfg["seed"])

    m, d, t = cfg["model"], cfg["data"], cfg["train"]
    data_file = resolve(d["dedup_dir"]) / (
        "python.jsonl" if variant == "python" else "general.jsonl"
    )
    if not data_file.exists():
        raise SystemExit(f"missing {data_file}; run the data pipeline for "
                         f"--corpus {variant} first")

    out_dir = resolve(t["runs_dir"]) / f"{cfg['experiment']}_{variant}"
    print(f"variant={variant}  base={m['base']}  out={out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(m["base"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16, fp16 = pick_precision(t["precision"])
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(m["base"], dtype=dtype)
    if t.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_ds = build_packed_dataset(
        data_file, tokenizer, m["seq_len"], d["train_tokens"]
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # transformers 5.x dropped warmup_ratio; derive warmup_steps from it.
    import math as _math
    eff_batch = t["per_device_batch_size"] * t["grad_accum"]
    steps_per_epoch = _math.ceil(len(train_ds) / eff_batch)
    total_steps = max(1, steps_per_epoch * t["epochs"])
    warmup_steps = int(t["warmup_ratio"] * total_steps)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=float(t["lr"]),
        weight_decay=t["weight_decay"],
        warmup_steps=warmup_steps,
        lr_scheduler_type=t["lr_scheduler"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        bf16=bf16,
        fp16=fp16,
        report_to=[],
        seed=cfg["seed"],
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
    )

    print(f"precision: bf16={bf16} fp16={fp16} (device "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'})")
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
