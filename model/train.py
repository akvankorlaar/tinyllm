"""Pretrain a tiny MistralForCausalLM from scratch on Python.

Reads the uint16 token streams written by data/tokenizer.py, cuts them into
fixed seq_len blocks, and trains a randomly initialized model with the
architecture in the config's `model:` section. Runs on MPS (Apple GPU), CUDA,
or CPU, whichever is available.

Entry point:
  python model/train.py configs/tiny_20m.yaml
  python model/train.py configs/tiny_20m.yaml --resume   # from last checkpoint

`data.train_bin` overrides the training stream (default
<tokens_dir>/train.bin). `train.init_from` starts from a trained model dir
instead of random init, with a fresh optimizer and LR schedule.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import (  # noqa: E402
    MistralConfig,
    MistralForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

from common import load_config, resolve  # noqa: E402

# Held-out tokens scored at each save, for an eval_loss curve during training.
EVAL_BLOCKS = 64


class TokenBlocks(torch.utils.data.Dataset):
    """Non-overlapping seq_len blocks over a flat uint16 token file."""

    def __init__(self, path: Path, seq_len: int, max_blocks: int | None = None):
        self.path, self.seq_len = path, seq_len
        n = (path.stat().st_size // 2) // seq_len
        self.n = min(n, max_blocks) if max_blocks else n
        self._data = None  # opened lazily so the dataset stays picklable

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        if self._data is None:
            self._data = np.memmap(self.path, dtype=np.uint16, mode="r")
        s = i * self.seq_len
        ids = torch.from_numpy(self._data[s : s + self.seq_len].astype(np.int64))
        return {"input_ids": ids, "labels": ids}


def pick_precision(requested: str) -> tuple[bool, bool]:
    """Return (bf16, fp16) flags honoring hardware support."""
    if requested != "bf16":
        return False, False
    if torch.cuda.is_available():
        # Pre-Ampere GPUs emulate bf16 (slower than fp32) and run fp16 slowly
        # without tensor cores (e.g. GTX 1650), so stay in fp32 there.
        native_bf16 = torch.cuda.get_device_capability()[0] >= 8
        return (True, False) if native_bf16 else (False, False)
    if torch.backends.mps.is_available():
        return True, False
    return False, False  # CPU: fp32


def build_model(m: dict, tok) -> MistralForCausalLM:
    config = MistralConfig(
        vocab_size=len(tok),
        hidden_size=m["hidden_size"],
        intermediate_size=m["intermediate_size"],
        num_hidden_layers=m["num_hidden_layers"],
        num_attention_heads=m["num_attention_heads"],
        num_key_value_heads=m["num_key_value_heads"],
        max_position_embeddings=m["seq_len"],
        rope_theta=m["rope_theta"],
        sliding_window=None,
        tie_word_embeddings=m["tie_word_embeddings"],
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    return MistralForCausalLM(config)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the last checkpoint in the run dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    m, d, t = cfg["model"], cfg["data"], cfg["train"]

    tokens_dir = resolve(d["tokens_dir"])
    train_bin = resolve(d.get("train_bin", tokens_dir / "train.bin"))
    heldout_bin = tokens_dir / "heldout.bin"
    if not train_bin.exists():
        raise SystemExit(f"missing {train_bin}; run data/tokenizer.py (or data/quality.py) first")

    tok = PreTrainedTokenizerFast.from_pretrained(str(resolve(cfg["tokenizer"]["dir"])))
    if t.get("init_from"):
        model = MistralForCausalLM.from_pretrained(str(resolve(t["init_from"])))
        print(f"init from {resolve(t['init_from'])}")
    else:
        model = build_model(m, tok)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: MistralForCausalLM, {n_params/1e6:.1f}M params")

    train_ds = TokenBlocks(train_bin, m["seq_len"])
    eval_ds = TokenBlocks(heldout_bin, m["seq_len"], EVAL_BLOCKS) if heldout_bin.exists() else None
    print(f"train: {len(train_ds)} blocks x {m['seq_len']} = "
          f"{len(train_ds) * m['seq_len'] / 1e6:.1f}M tokens")

    out_dir = resolve(t["runs_dir"]) / cfg["experiment"]
    bf16, fp16 = pick_precision(t["precision"])
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=float(t["lr"]),
        weight_decay=t["weight_decay"],
        adam_beta2=0.95,
        max_grad_norm=t["max_grad_norm"],
        warmup_steps=t["warmup_steps"],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": t["min_lr_ratio"]},
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=t["save_steps"],
        bf16=bf16,
        fp16=fp16,
        report_to=[],
        seed=cfg["seed"],
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),  # unsupported on MPS
        torch_compile=t.get("torch_compile", False),
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds)

    print(f"precision: bf16={bf16} fp16={fp16} (device {targs.device})")
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
