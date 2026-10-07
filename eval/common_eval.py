"""Shared eval helpers: load a model, generate text, run code safely."""
from __future__ import annotations

import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(model_path: str):
    """Load a tokenizer+model from an HF id or a local run dir."""
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device())
    model.eval()
    return tok, model


def context_len(model) -> int:
    return model.config.max_position_embeddings


_DEDENT = re.compile(r"\n(?=\S)")


def end_of_body(text: str) -> int:
    """Index where a function body ends: the first line not indented."""
    m = _DEDENT.search(text)
    return m.start() if m else len(text)


class _StopAt(StoppingCriteria):
    """Stop once `done(generated_text)` is true (checked every step)."""

    def __init__(self, tok, n_prompt: int, done):
        self.tok, self.n_prompt, self.done = tok, n_prompt, done

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        done = self.done(self.tok.decode(input_ids[0, self.n_prompt:], skip_special_tokens=True))
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


@torch.no_grad()
def generate(tok, model, prompt: str, max_new_tokens: int, temperature: float,
             stop=None) -> str:
    """Return only the newly generated text (prompt stripped).

    The prompt is left-truncated so prompt + generation fits the model's
    context. `stop(text) -> bool` ends generation early.

    Trailing whitespace is cut from the prompt before encoding: BPE merges a
    newline with the next line's indent, so a prompt ending in a bare "\n"
    is a token boundary the model never sees in training (it then emits a
    blank line and ends the function). The returned text still continues
    the original `prompt`.
    """
    base = prompt.rstrip()
    tail = prompt[len(base):]

    def after_prompt(text: str) -> str:
        if text.startswith(tail):
            return text[len(tail):]
        return "" if tail.startswith(text) else text

    ids = tok(base, return_tensors="pt", add_special_tokens=False)["input_ids"]
    ctx = context_len(model)
    max_new_tokens = min(max_new_tokens, ctx // 2)
    ids = ids[:, -(ctx - max_new_tokens):].to(device())
    do_sample = temperature and temperature > 0
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tok.pad_token_id,
        stopping_criteria=StoppingCriteriaList(
            [_StopAt(tok, ids.shape[1], lambda t: stop(after_prompt(t)))]) if stop else None,
    )
    return after_prompt(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))


@torch.no_grad()
def continuation_bits(tok, model, prompt: str, continuation: str) -> tuple[float, int]:
    """Bits the model needs to encode `continuation` given `prompt`.

    Returns (total bits, continuation bytes); bits/bytes is tokenizer-free.
    Prompt and continuation are tokenized separately so the boundary is exact.
    The prompt's trailing whitespace moves into the continuation (see
    `generate`), so the boundary falls where BPE would put one.
    """
    base = prompt.rstrip()
    prompt, continuation = base, prompt[len(base):] + continuation
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(continuation, add_special_tokens=False)["input_ids"]
    ctx = context_len(model)
    if len(c) > ctx - 1:
        c = c[:ctx - 1]
        continuation = tok.decode(c)
    p = p[-(ctx - len(c)):] if len(p) + len(c) > ctx else p
    ids = torch.tensor([p + c], device=device())
    logits = model(ids).logits[0, len(p) - 1 : -1].float()
    nll = torch.nn.functional.cross_entropy(logits, ids[0, len(p):], reduction="sum")
    return nll.item() / math.log(2), len(continuation.encode("utf-8"))


# --------------------------------------------------------------------------- #
# Sandboxed execution: run candidate code + asserts in a locked-down subprocess.
# --------------------------------------------------------------------------- #
# Each limit is best-effort: macOS rejects RLIMIT_AS, which must not abort the
# program (that would fail every candidate regardless of the code).
_LIMITS = """
import resource, sys
for _lim, _val in ((resource.RLIMIT_CPU, {cpu}), (resource.RLIMIT_AS, {mem})):
    try:
        resource.setrlimit(_lim, (_val, _val))
    except (ValueError, OSError):
        pass
sys.setrecursionlimit(10000)
"""


def run_program(program: str, timeout_sec: int, mem_mb: int = 512) -> tuple[bool, str]:
    """Run `program` in a fresh python subprocess. Returns (passed, detail)."""
    harness = _LIMITS.format(cpu=timeout_sec, mem=mem_mb * 1024 * 1024) + program
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(harness)
        tmp = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=timeout_sec + 2,
        )
        if proc.returncode == 0:
            return True, "ok"
        return False, (proc.stderr.strip().splitlines() or ["nonzero exit"])[-1]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001
        return False, f"runner_error: {e}"
    finally:
        try:
            Path(tmp).unlink()
        except OSError:
            pass
