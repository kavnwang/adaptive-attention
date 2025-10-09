import os
import json
import threading
from typing import Generator

import torch
import fla.models.transformer  # noqa: F401  # registers custom transformer classes with HF Auto*
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

APP_HOST = os.environ.get("APP_HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("APP_PORT", "7860"))

HF_MODEL_DIR = os.environ.get("HF_MODEL_DIR")  # REQUIRED
if not HF_MODEL_DIR:
    raise RuntimeError(
        "Set HF_MODEL_DIR to the path of your converted HF model directory "
        "(the one containing config.json)"
    )

def _load_raw_config(hf_dir: str) -> dict:
    cfg_path = os.path.join(hf_dir, "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _resolve_dtype(device: str) -> torch.dtype:
    """Pick a torch dtype compatible with FlashAttention (fp16/bf16 on CUDA)."""
    env = os.environ.get("TORCH_DTYPE")
    if env:
        key = env.strip().lower().replace("torch.", "")
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if key not in mapping:
            raise ValueError(
                f"Unsupported TORCH_DTYPE '{env}'. Use one of: {', '.join(sorted(mapping))}."
            )
        return mapping[key]

    if device == "cuda":
        # FlashAttention kernels expect fp16/bf16 inputs; default to fp16 unless overridden.
        return torch.float16
    return torch.float32


# --- Load model & tokenizer at import time so the first request is fast.
try:
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR, use_fast=True)
except Exception:
    # Some tokenizers don't ship a 'fast' tokenizer; fall back silently.
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR, use_fast=False)

# For bento/fla models, trust_remote_code=True so custom model classes load.
# Device policy: cuda if available, else cpu. (No accelerate dependency here.)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = _resolve_dtype(device)
if device == "cuda" and dtype == torch.float32:
    raise RuntimeError(
        "FlashAttention-backed transformer requires fp16 or bf16 weights on CUDA. "
        "Set TORCH_DTYPE=float16 or TORCH_DTYPE=bfloat16, or unset TORCH_DTYPE."
    )

model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_DIR,
    trust_remote_code=True,
    dtype=dtype,
)
model.to(device=device, dtype=dtype)
model.eval()

# Infer maximum total tokens budget based on model config.
# For JOYCE-integrated configs, prefer 2 * seq_len (where compression triggers),
# but never exceed max_position_embeddings if set.
_raw_cfg = _load_raw_config(HF_MODEL_DIR)
_cfg_mpe = _raw_cfg.get("max_position_embeddings")
_cfg_seq = _raw_cfg.get("seq_len")

def _infer_max_total_tokens() -> int:
    # Try via loaded HF config first (unknown keys typically survive as attributes)
    mpe = getattr(model.config, "max_position_embeddings", None)
    seq = getattr(model.config, "seq_len", None)

    # Fallback to raw config.json values if needed
    if mpe is None:
        mpe = _cfg_mpe
    if seq is None:
        seq = _cfg_seq

    max_total = None
    if isinstance(seq, int) and seq > 0:
        # Training recipe targets total length ~ 2 * seq_len
        candidate = 2 * int(seq)
        if isinstance(mpe, int) and mpe > 0:
            max_total = min(candidate, int(mpe))
        else:
            max_total = candidate
    elif isinstance(mpe, int) and mpe > 0:
        max_total = int(mpe)
    else:
        # Conservative default if neither is available
        max_total = 4096

    # Always clamp to a reasonable positive integer
    return max(1, int(max_total))

MODEL_MAX_TOTAL_TOKENS = _infer_max_total_tokens()

app = FastAPI(title="HF Localhost Minimal Server")


@app.get("/api/healthz")
def healthz():
    return {"status": "ok", "device": device}


def _sse_format(data: str, event: str | None = None) -> bytes:
    """Format a Server-Sent Event line."""
    lines = []
    if event:
        lines.append(f"event: {event}")
    # Ensure no bare CR/LF inside; SSE is line-oriented.
    safe = data.replace("\r", "\\r")
    for line in safe.split("\n"):
        lines.append(f"data: {line}")
    lines.append("")  # terminator
    return ("\n".join(lines) + "\n").encode("utf-8")


def generate_stream(prompt: str,
                    max_new_tokens: int = 256,
                    temperature: float = 0.8,
                    top_p: float = 0.95) -> Generator[bytes, None, None]:
    """
    SSE generator that yields chunks as the model produces them.
    Uses TextIteratorStreamer to stream tokens.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Enforce total tokens budget: input_len + max_new_tokens <= MODEL_MAX_TOTAL_TOKENS
    input_len = int(inputs.get("input_ids").shape[1])
    remaining_budget = max(0, int(MODEL_MAX_TOTAL_TOKENS) - input_len)
    if remaining_budget <= 0:
        # Prompt already exceeds (or equals) the max total tokens budget
        yield _sse_format(
            f"Prompt length {input_len} exceeds max total tokens {MODEL_MAX_TOTAL_TOKENS}"
        )
        yield _sse_format("[DONE]", event="done")
        return
    if max_new_tokens > remaining_budget:
        max_new_tokens = remaining_budget

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        timeout=None,
        # Avoid showing special tokens mid-stream.
        # Note: decode kwargs are forwarded by the streamer.
        clean_up_tokenization_spaces=True
    )

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # Initial event to signal start
    yield _sse_format("", event="start")

    for token_text in streamer:
        # Send each fragment as an SSE "data" line
        if token_text:
            yield _sse_format(token_text)

    # Final event to tell the client we're done
    yield _sse_format("[DONE]", event="done")


@app.get("/api/generate")
def api_generate(
    prompt: str = Query(..., min_length=1),
    max_new_tokens: int = Query(256, ge=1, le=MODEL_MAX_TOTAL_TOKENS),
    temperature: float = Query(0.8, ge=0.0, le=5.0),
    top_p: float = Query(0.95, ge=0.0, le=1.0),
):
    try:
        return StreamingResponse(
            generate_stream(prompt, max_new_tokens, temperature, top_p),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve the static index.html
app.mount("/", StaticFiles(directory="static", html=True), name="static")
