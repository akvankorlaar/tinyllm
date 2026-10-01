"""Download raw Python (or general) text for Experiment 1.

Sources
-------
python corpus (data.source):
  starcoderdata      : bigcode/starcoderdata. Inline content, NO AWS. Gated
                       (instant accept). Already cleaned/deduped/filtered.
                       Default and recommended for Experiment 1.
  the-stack-v2-dedup : bigcode/the-stack-v2-dedup. Metadata only on HF; file
                       CONTENT is fetched from the Software Heritage S3 bucket.
                       Requires: accept the dataset gate on huggingface.co AND
                       AWS credentials (any free account) for boto3. Use when
                       you need Stack v2's scale.
  codeparrot         : codeparrot/github-code-clean. No gate, but script-based;
                       may not load under datasets 5.x.

general corpus (data.general_source), used for the control model B:
  fineweb-edu        : HuggingFaceFW/fineweb-edu, sample-10BT. No gate.

Usage
-----
  python data/download.py configs/experiment_001.yaml --corpus python
  python data/download.py configs/experiment_001.yaml --corpus general

The python corpus is split into:
  data.heldout_dir : first `eval_heldout_files` files, held out before training
  data.raw_dir     : the rest, fed to clean.py
"""
from __future__ import annotations

import argparse
import sys

from datasets import load_dataset
from tqdm import tqdm

from common import load_config, resolve, write_jsonl

# A char/token ratio good enough to decide when to STOP downloading.
# Real token counts are measured later by the tokenizer in pack/train.
CHARS_PER_TOKEN = 3.6


def approx_tokens(n_chars: int) -> float:
    return n_chars / CHARS_PER_TOKEN


# --------------------------------------------------------------------------- #
# The Stack v2 dedup: content lives in Software Heritage S3.
# --------------------------------------------------------------------------- #
def _stack_v2_content_fetcher():
    import gzip

    import boto3
    from botocore.config import Config
    from smart_open import open as s_open

    session = boto3.Session()
    creds = session.get_credentials()
    if creds is None:
        sys.exit(
            "No AWS credentials found. The Stack v2 content is served from the\n"
            "Software Heritage S3 bucket and needs AWS creds (any free account):\n"
            "  aws configure            # or export AWS_ACCESS_KEY_ID / SECRET\n"
            "Or switch to the no-gate fallback by setting data.source: codeparrot."
        )
    s3 = session.client("s3", config=Config(retries={"max_attempts": 5}))

    def fetch(blob_id: str, src_encoding: str) -> str | None:
        url = f"s3://softwareheritage/content/{blob_id}"
        try:
            with s_open(url, "rb", transport_params={"client": s3}) as fin:
                raw = gzip.decompress(fin.read())
            return raw.decode(src_encoding or "utf-8", errors="ignore")
        except Exception:
            return None

    return fetch


def iter_stack_v2(language: str):
    ds = load_dataset(
        "bigcode/the-stack-v2-dedup",
        language,
        split="train",
        streaming=True,
    )
    fetch = _stack_v2_content_fetcher()
    for row in ds:
        # Cheap metadata pre-filters: skip before paying for an S3 fetch.
        if row.get("is_vendor") or row.get("is_generated"):
            continue
        content = fetch(row["blob_id"], row.get("src_encoding", "utf-8"))
        if not content:
            continue
        yield {
            "content": content,
            "path": row.get("path", ""),
            "repo": row.get("repo_name", ""),
            "source": "the-stack-v2-dedup",
        }


def iter_starcoderdata(language: str):
    # starcoderdata is organized into per-language subdirs; content is inline.
    ds = load_dataset(
        "bigcode/starcoderdata",
        data_dir=language.lower(),
        split="train",
        streaming=True,
    )
    for row in ds:
        content = row.get("content", "")
        if not content:
            continue
        yield {
            "content": content,
            "path": row.get("max_stars_repo_path", ""),
            "repo": row.get("max_stars_repo_name", ""),
            "source": "starcoderdata",
        }


def iter_codeparrot(language: str):
    # github-code-clean uses language names like "Python".
    ds = load_dataset(
        "codeparrot/github-code-clean",
        languages=[language],
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    for row in ds:
        lic = row.get("license", "")
        if lic and lic not in {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause"}:
            continue
        yield {
            "content": row["code"],
            "path": row.get("path", ""),
            "repo": row.get("repo_name", ""),
            "source": "codeparrot",
        }


def iter_fineweb_edu():
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    for row in ds:
        yield {
            "content": row["text"],
            "path": "",
            "repo": row.get("url", ""),
            "source": "fineweb-edu",
        }


def get_python_iter(source: str, language: str):
    if source == "starcoderdata":
        return iter_starcoderdata(language)
    if source == "the-stack-v2-dedup":
        return iter_stack_v2(language)
    if source == "codeparrot":
        return iter_codeparrot(language)
    sys.exit(f"Unknown python source: {source}")


def get_general_iter(source: str):
    if source == "fineweb-edu":
        return iter_fineweb_edu()
    sys.exit(f"Unknown general source: {source}")


def download_python(cfg: dict) -> None:
    d = cfg["data"]
    target = d["train_tokens"] * d["overshoot"]
    heldout_n = d["eval_heldout_files"]

    raw_path = resolve(d["raw_dir"]) / "python.jsonl"
    heldout_path = resolve(d["heldout_dir"]) / "python_heldout.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    heldout_path.parent.mkdir(parents=True, exist_ok=True)

    it = get_python_iter(d["source"], d["language"])

    got_tokens = 0.0
    held = 0
    kept = 0
    raw_buf, held_buf = [], []
    pbar = tqdm(desc="python files", unit="file", mininterval=2.0)
    for rec in it:
        if not rec["content"].strip():
            continue
        if held < heldout_n:
            held_buf.append(rec)
            held += 1
        else:
            raw_buf.append(rec)
            kept += 1
            got_tokens += approx_tokens(len(rec["content"]))
        pbar.update(1)
        pbar.set_postfix(held=held, raw=kept, approx_tok=f"{got_tokens/1e6:.1f}M")
        if len(raw_buf) >= 2000:
            write_jsonl(raw_path, raw_buf, append=True)
            raw_buf.clear()
        if got_tokens >= target:
            break
    pbar.close()

    if held_buf:
        write_jsonl(heldout_path, held_buf)
    if raw_buf:
        write_jsonl(raw_path, raw_buf, append=True)

    print(f"held-out files : {held}  -> {heldout_path}")
    print(f"raw files      : {kept}  -> {raw_path}")
    print(f"approx tokens  : {got_tokens/1e6:.1f}M (target {target/1e6:.1f}M, pre-clean)")


def download_general(cfg: dict) -> None:
    d = cfg["data"]
    target = d["train_tokens"] * d["overshoot"]
    out = resolve(d["general_dir"]) / "general.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    it = get_general_iter(d["general_source"])
    got_tokens = 0.0
    kept = 0
    buf = []
    pbar = tqdm(desc="general docs", unit="doc", mininterval=2.0)
    for rec in it:
        if not rec["content"].strip():
            continue
        buf.append(rec)
        kept += 1
        got_tokens += approx_tokens(len(rec["content"]))
        pbar.update(1)
        pbar.set_postfix(approx_tok=f"{got_tokens/1e6:.1f}M")
        if len(buf) >= 2000:
            write_jsonl(out, buf, append=True)
            buf.clear()
        if got_tokens >= target:
            break
    pbar.close()
    if buf:
        write_jsonl(out, buf, append=True)
    print(f"general docs : {kept}  -> {out}")
    print(f"approx tokens: {got_tokens/1e6:.1f}M (target {target/1e6:.1f}M, pre-clean)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--corpus", choices=["python", "general"], default="python")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.corpus == "python":
        download_python(cfg)
    else:
        download_general(cfg)


if __name__ == "__main__":
    main()
