"""Fit a byte-level BPE tokenizer on the deduped corpus, then encode it.

1. Fit: byte-level BPE (vocab tokenizer.vocab_size) on the first
   tokenizer.train_chars characters of data/dedup/python.jsonl. Saved as a
   Hugging Face tokenizer to tokenizer.dir. Skipped if it already exists
   (pass --refit to redo it).
2. Encode: every deduped file + EOS, concatenated into a flat uint16 array
   (data.tokens_dir/train.bin), stopping at data.train_tokens. The held-out
   split is encoded too (heldout.bin) for perplexity.

Usage
-----
  python data/tokenizer.py configs/tiny_20m.yaml
"""
from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from common import load_config, read_jsonl, resolve

EOS = "<|endoftext|>"


def iter_texts(path: Path, max_chars: int):
    used = 0
    for rec in read_jsonl(path):
        yield rec["content"]
        used += len(rec["content"])
        if used >= max_chars:
            return


def fit(corpus: Path, out_dir: Path, vocab_size: int, max_chars: int) -> None:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(iter_texts(corpus, max_chars), trainer=trainer)
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token=EOS, bos_token=EOS, pad_token=EOS,
        unk_token=None,
    )
    hf.save_pretrained(str(out_dir))
    print(f"tokenizer: vocab {hf.vocab_size} -> {out_dir}")


def encode(tok, src: Path, out: Path, max_tokens: int | None) -> int:
    """Encode docs (+EOS) into a flat uint16 file. Returns tokens written."""
    assert len(tok) <= 65536, "uint16 storage needs vocab <= 65536"
    eos = tok.eos_token_id
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    recs = read_jsonl(src)
    pbar = tqdm(desc=f"encode {src.name}", unit="tok", unit_scale=True, total=max_tokens)
    with open(out, "wb") as f:
        while True:
            batch = [r["content"] for r in islice(recs, 1000)]
            if not batch:
                break
            ids = tok(batch, add_special_tokens=False)["input_ids"]
            arr = np.fromiter((t for doc in ids for t in (*doc, eos)), dtype=np.uint16)
            if max_tokens is not None:
                arr = arr[: max_tokens - written]
            arr.tofile(f)
            written += len(arr)
            pbar.update(len(arr))
            if max_tokens is not None and written >= max_tokens:
                break
    pbar.close()
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--refit", action="store_true", help="refit even if a tokenizer exists")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d, tc = cfg["data"], cfg["tokenizer"]

    corpus = resolve(d["dedup_dir"]) / "python.jsonl"
    if not corpus.exists():
        raise SystemExit(f"missing input {corpus}; run deduplicate.py first")
    tok_dir = resolve(tc["dir"])
    if args.refit or not (tok_dir / "tokenizer.json").exists():
        fit(corpus, tok_dir, tc["vocab_size"], tc["train_chars"])
    tok = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))

    tokens_dir = resolve(d["tokens_dir"])
    n = encode(tok, corpus, tokens_dir / "train.bin", d["train_tokens"])
    print(f"train: {n/1e6:.1f}M tokens (budget {d['train_tokens']/1e6:.1f}M)")
    if n < d["train_tokens"]:
        print("  WARNING: corpus ran out before the budget; raise data.overshoot")
    heldout = resolve(d["heldout_dir"]) / "python_heldout.jsonl"
    if heldout.exists():
        n = encode(tok, heldout, tokens_dir / "heldout.bin", None)
        print(f"heldout: {n/1e6:.2f}M tokens")


if __name__ == "__main__":
    main()
