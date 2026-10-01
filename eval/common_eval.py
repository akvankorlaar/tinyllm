"""Shared eval helpers: load a model, generate text, run code safely."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(model_path: str):
    """Load a tokenizer+model from an HF id or a local run dir."""
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device())
    model.eval()
    return tok, model


@torch.no_grad()
def generate(tok, model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    """Return only the newly generated text (prompt stripped)."""
    inputs = tok(prompt, return_tensors="pt").to(device())
    do_sample = temperature and temperature > 0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def extract_code(text: str) -> str:
    """Strip markdown fences if the model emitted them."""
    if "```" in text:
        parts = text.split("```")
        # take first fenced block
        block = parts[1]
        if block.startswith("python"):
            block = block[len("python"):]
        return block
    return text


# --------------------------------------------------------------------------- #
# Sandboxed execution: run candidate code + asserts in a locked-down subprocess.
# --------------------------------------------------------------------------- #
_LIMITS = """
import resource, sys
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
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
