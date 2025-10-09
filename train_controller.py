# adaptive-attention/train_controller.py
import math, json, random, argparse, time
from enum import IntEnum
from itertools import cycle
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# --- Cause Detection for Full Forward Pass ---
class FullLossCause(IntEnum):
    OK = 0
    FP16_OVERFLOW = 1
    ZERO_COUNTED_TOKENS = 2
    PAD_MISALIGNMENT = 3
    VOCAB_MISMATCH = 4
    UNKNOWN_NONFINITE = 5

@torch.no_grad()
def diagnose_full_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    ignore_idx: int = -100,
    probe_logits: bool = True,
) -> Tuple[FullLossCause, dict]:
    """
    Returns (cause, info) where cause != OK means computing the default CE loss may be NaN/Inf.
    info contains fields for printing.
    """
    info = {}
    V = getattr(model.config, "vocab_size", None)
    B, T = input_ids.shape
    info["B"], info["T"] = int(B), int(T)

    # 1) Build pad-safe labels for the full forward (simulate HF's causal left-shift)
    labels_full = labels.clone()
    if attention_mask is not None:
        ignore_new = (attention_mask == 0)
        apply_on_src = torch.roll(ignore_new, shifts=1, dims=1)
        apply_on_src[:, 0] = False
        labels_full[apply_on_src] = ignore_idx

    counted = int((labels_full != ignore_idx).sum().item())
    info["counted_tokens"] = counted

    # 2) Vocab sanity
    mx = int(labels.max().item())
    mn = int(labels.min().item())
    info["label_min"], info["label_max"] = mn, mx
    if V is not None:
        info["V"] = int(V)
        if mx >= V or mn < ignore_idx:
            return FullLossCause.VOCAB_MISMATCH, info

    # 3) Zero denominator case
    if counted == 0:
        return FullLossCause.ZERO_COUNTED_TOKENS, info

    # 4) Optional logits probe to catch fp16/bf16 overflow *before* CE
    if probe_logits:
        out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(out, "logits"):
            logits = out.logits
            info["logits_dtype"] = str(logits.dtype).replace("torch.", "")
            if not torch.isfinite(logits).all():
                info["n_nan_logits"] = int(torch.isnan(logits).sum().item())
                info["n_inf_logits"] = int(torch.isinf(logits).sum().item())
                return FullLossCause.FP16_OVERFLOW, info

    # 5) If we get here, inputs look sane; any non-finite loss later is "unknown"
    return FullLossCause.OK, info

# --- Backbone LM (choose what you use in-flame/llmonade setup) ---
# Provide both Transformer and Mamba2; default to Transformer.
from fla.models.transformer.configuration_transformer import TransformerConfig
from fla.models.transformer.modeling_transformer import TransformerForCausalLM
from fla.models.mamba2.configuration_mamba2 import Mamba2Config
from fla.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM

# --- Controller head (new) ---
from fla.models.joyce_controller import JoyceControllerConfig, JoyceControllerForCompressionRatio
from fla.layers.joyce import JoyceCompressionBlock, JoyceUpsamplingBlock, JoyceBlockCfg

# If your Joyce compressor/upsampler lives in this repo, import it; otherwise keep a placeholder.
try:
    from your_joyce_pkg.compression import compress_block_forward, upsample_init   # <-- replace with your symbols
except Exception:
    compress_block_forward, upsample_init = None, None

# Fallback: simple input-level "compression" by masking tokens to approximate a keep ratio.
# This does NOT replicate Joyce layer-L semantics but allows collecting labels end-to-end.

class _NanSentinel:
    def __init__(self):
        self.tripped = False
        self.messages = []

    def check(self, name: str, t: torch.Tensor):
        if not torch.isfinite(t).all():
            self.tripped = True
            with torch.no_grad():
                n_nans = torch.isnan(t).sum().item()
                n_infs = torch.isinf(t).sum().item()
                self.messages.append(f"[NaNGuard] {name}: NaN={n_nans} Inf={n_infs} "
                                     f"min={t.nanmin().item() if t.numel() else 'NA'} "
                                     f"max={t.nanmax().item() if t.numel() else 'NA'} "
                                     f"dtype={t.dtype} shape={tuple(t.shape)}")

def _dump_bad_batch(dump_dir: Path, payload: dict, tag: str):
    dump_dir.mkdir(parents=True, exist_ok=True)
    out = dump_dir / f"bad_batch_{tag}.pt"
    torch.save(payload, out)
    print(f"[NaNGuard] Dumped repro payload to: {out}")

def _prefix_keep_mask(attn: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    # attn: [B, T] in {0,1}, where 1 marks valid tokens
    B, T = attn.shape
    keep = torch.zeros_like(attn)
    for b in range(B):
        # valid region is assumed to be a prefix in typical LM training
        # if not, we still keep a left-justified prefix among the valid positions
        idx_valid = torch.nonzero(attn[b] > 0, as_tuple=False).squeeze(-1)
        L = int(idx_valid.numel())
        if L == 0:
            continue
        k = max(1, int(math.ceil(L * float(keep_ratio))))
        k = min(k, L)
        kept = idx_valid[:k]
        keep[b, kept] = 1
    return keep

def _uniform_keep_mask(attn: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    # attn: [B, T] in {0,1}
    B, T = attn.shape
    keep = torch.zeros_like(attn)
    for b in range(B):
        idx_valid = torch.nonzero(attn[b] > 0, as_tuple=False).squeeze(-1)
        L = int(idx_valid.numel())
        if L == 0:
            continue
        k = max(1, int(math.ceil(L * keep_ratio)))
        if k >= L:
            keep[b, idx_valid] = 1
            continue
        # Uniformly spaced positions among valid tokens
        sel = torch.linspace(0, L - 1, steps=k, device=attn.device)
        sel = torch.round(sel).to(torch.long)
        picked = idx_valid.index_select(0, sel)
        keep[b, picked] = 1
    return keep

def _compress_block_forward_fallback(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_L: int,
    compression_ratio: float,
) -> torch.Tensor:
    sentinel = _NanSentinel()
    
    # Compose a new attention mask with approximately the requested keep ratio.
    # Use prefix mask to avoid holey attention that breaks fused kernels
    keep_mask = _prefix_keep_mask(attention_mask, float(compression_ratio)).to(attention_mask.dtype)
    attn_comp = (attention_mask & keep_mask).to(attention_mask.dtype)
    
    # Safety: ensure at least one valid token per sequence after compression
    with torch.no_grad():
        # check prefix property: ones then zeros for each row (within the valid region)
        diffs = attn_comp[:, 1:] - attn_comp[:, :-1]
        if (diffs < -1).any() or (diffs > 1).any():
            print("[warn] unexpected values in attn_comp")
        # If any 1 appears after a 0 (hole), flag
        holes = (diffs == 1).any().item()
        if holes:
            print("[warn] attn_comp is not prefix; this can break fused kernels")
    
    # Build labels so that model's internal shift results in ignored positions for masked queries.
    ignore_idx = -100
    labels_mod = labels.clone()
    # Positions to ignore after the model's internal left-shift
    ignore_new = (attn_comp == 0)
    apply_on_src = torch.roll(ignore_new, shifts=1, dims=1)
    apply_on_src[:, 0] = False  # first position's label isn't used post-shift
    labels_mod[apply_on_src] = ignore_idx

    out = model(
        input_ids=input_ids,
        attention_mask=attn_comp,
        labels=labels_mod,
        output_hidden_states=False,
        return_dict=True,
    )
    
    sentinel.check("fallback.attn_comp", attn_comp)
    
    # Some HF models don't expose logits when labels are provided. Re-run without labels to inspect.
    try:
        with torch.no_grad():
            out_no_lbl = model(input_ids=input_ids,
                               attention_mask=attn_comp,
                               output_hidden_states=False,
                               return_dict=True)
            if hasattr(out_no_lbl, "logits"):
                sentinel.check("fallback.logits", out_no_lbl.logits)
    except Exception:
        pass

    if sentinel.tripped:
        _dump_bad_batch(Path("debug_dumps"), {
            "input_ids": input_ids.detach().cpu(),
            "labels": labels.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "attn_comp": attn_comp.detach().cpu(),
        }, tag="fallback")
        for m in sentinel.messages:
            print(m)

    return out.loss.detach()

if compress_block_forward is None:
    compress_block_forward = _compress_block_forward_fallback


# --------------------------
# Optional: Wire actual Joyce AE compressor from exp/joyce_ae_160M
# --------------------------
class _BaseStackAdapter:
    """Minimal shim to run a HF-style decoder stack in slices."""
    def __init__(self, base_model: nn.Module):
        self.base = base_model
        # embeddings
        if hasattr(base_model, "get_input_embeddings"):
            self.tok_embed = base_model.get_input_embeddings()
        elif hasattr(base_model, "model") and hasattr(base_model.model, "embed_tokens"):
            self.tok_embed = base_model.model.embed_tokens
        else:
            raise AttributeError("Could not find token embedding on base model")
        # layers
        self.layers = (
            getattr(getattr(base_model, "model", base_model), "layers", None)
            or getattr(getattr(base_model, "transformer", base_model), "layers", None)
            or getattr(getattr(base_model, "model", base_model), "h", None)
            or getattr(getattr(base_model, "transformer", base_model), "h", None)
            or getattr(base_model, "layers", None)
        )
        if not isinstance(self.layers, (list, nn.ModuleList)):
            raise AttributeError("Could not find decoder layers list on base model")
        # final norm
        self.final_norm = (
            getattr(getattr(base_model, "model", base_model), "norm", None)
            or getattr(base_model, "norm", None)
            or getattr(getattr(base_model, "transformer", base_model), "ln_f", None)
            or getattr(base_model, "ln_f", None)
        )
        if self.final_norm is None:
            raise AttributeError("Could not find final norm on base model")
        # lm head
        if hasattr(base_model, "lm_head"):
            self.lm_head = base_model.lm_head
        elif hasattr(base_model, "get_output_embeddings"):
            self.lm_head = base_model.get_output_embeddings()
        else:
            raise AttributeError("Could not find lm_head on base model")

    def _call_block(self, block: nn.Module, x: torch.Tensor, position_ids: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Call a decoder block and always return the hidden_states tensor (index 0 if tuple)."""
        out = None
        try:
            out = block(x, position_ids=position_ids, attention_mask=attention_mask, use_cache=False)
        except TypeError:
            try:
                out = block(x, position_ids=position_ids, attention_mask=attention_mask)
            except TypeError:
                out = block(x)
        # Many HF-style blocks return a tuple; take the first element as hidden states
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_embed(input_ids)

    def run_layers(self, x: torch.Tensor, start: int, end: int, position_ids: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        for i in range(start, end):
            x = self._call_block(self.layers[i], x, position_ids, attention_mask)
        return x

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        h = self.final_norm(h)
        return self.lm_head(h)


@torch.no_grad()
def _compress_block_forward_joyce(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_L: int,
    compression_ratio: float,
    *,
    joyce_modules: Tuple[JoyceCompressionBlock, JoyceUpsamplingBlock],
) -> torch.Tensor:
    """
    Compute NLL using the provided Joyce compressor/upsampler inserted at layer L.
    The keep ratio is simulated by gating the compressed tokens (first k kept, rest zero).
    """
    sentinel = _NanSentinel()
    device = input_ids.device
    adapter = _BaseStackAdapter(model)
    B, T = input_ids.shape
    # Embedding and prefix layers
    h = adapter.embed(input_ids)
    pos_ids = torch.arange(0, T, device=device)[None, :].expand(B, T)
    hL = adapter.run_layers(h, 0, layer_L + 1, pos_ids, attention_mask)

    sentinel.check("joyce.hL", hL)

    compress, upsample = joyce_modules
    # ensure modules are on same device/dtype
    compress = compress.to(device=device, dtype=hL.dtype)
    upsample = upsample.to(device=device, dtype=hL.dtype)

    # Compress and gate according to keep ratio
    z = compress(hL, attn_mask=None)  # [B, C, D]
    sentinel.check("joyce.z", z)
    C = z.shape[1]
    k = max(1, int(math.ceil(C * float(compression_ratio))))
    if k < C:
        z[:, k:, :].zero_()

    # Upsample to full length and run remaining layers
    # Re-tie mix_up weight to ensure shared storage on same device/dtype after any .to(...)
    try:
        if hasattr(upsample, "mix_up") and hasattr(compress, "mix_down"):
            upsample.mix_up.weight = compress.mix_down.weight.T
    except Exception:
        pass
    y = upsample(z, x_ctx=hL, attn_mask=None)  # [B, T, D]
    sentinel.check("joyce.upsampled", y)
    y = adapter.run_layers(y, layer_L + 1, len(adapter.layers), pos_ids, attention_mask)
    logits = adapter.logits(y)
    sentinel.check("joyce.logits", logits)

    # Compute CE like HF causal LM: shift by one
    B, T, V = logits.shape
    tgt = labels[:, 1:].contiguous().clone()
    # ignore pad positions based on attention mask
    if attention_mask is not None:
        ignore_idx = -100
        tgt[attention_mask[:, 1:] == 0] = ignore_idx
        loss = nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(B * (T - 1), V),
            tgt.view(B * (T - 1)),
            reduction="mean",
            ignore_index=ignore_idx,
        )
    else:
        pred = logits[:, :-1, :].contiguous().view(B * (T - 1), V)
        loss = nn.functional.cross_entropy(pred, tgt.view(B * (T - 1)), reduction="mean")
    
    if sentinel.tripped:
        _dump_bad_batch(Path("debug_dumps"), {
            "input_ids": input_ids.detach().cpu(),
            "labels": labels.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "layer_L": layer_L,
            "compression_ratio": float(compression_ratio),
            "hL": hL.detach().cpu(),
            "z": z.detach().cpu(),
            "k": k,
        }, tag=f"joyce_r{compression_ratio:.3f}_L{layer_L}")
        for m in sentinel.messages:
            print(m)
    
    return loss.detach()


def _maybe_load_joyce_modules(
    *,
    hidden_size: int,
    num_heads: int,
    seq_len: int,
    num_compressed: int,
    latent_dim: int,
    norm_eps: float,
    mlp_ratio: float,
    dropout: float,
    use_refine_attn: bool,
    tie_mixers: bool,
    ckpt_dir: Optional[str],
    ckpt_step: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[JoyceCompressionBlock, JoyceUpsamplingBlock]:
    """Build Joyce modules and optionally load weights from DCP checkpoint at exp/.../checkpoint/step-XXXX."""
    cfg = JoyceBlockCfg(
        dim=hidden_size,
        latent_dim=latent_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        norm_eps=norm_eps,
    )
    compress = JoyceCompressionBlock(cfg=cfg, t_in=seq_len, t_out=num_compressed, depth=1, tie_up_down=tie_mixers)
    upsample = JoyceUpsamplingBlock(
        cfg=cfg,
        t_out=seq_len,
        t_in=num_compressed,
        depth=1,
        tie_up_down=compress.mix_down if tie_mixers else None,
        use_refine_attn=use_refine_attn,
    )

    if ckpt_dir:
        try:
            import tempfile, os
            from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

            step = ckpt_step
            if step is None or step < 0:
                # Support either a directory containing step-* subdirs, or a specific step-* directory
                base = ckpt_dir
                basename = os.path.basename(os.path.normpath(base))
                if basename.startswith("step-") and os.path.isdir(base):
                    step_dir = base
                else:
                    dirs = [d for d in os.listdir(base) if d.startswith("step-")]
                    steps = [int(d.split("-")[-1]) for d in dirs] if dirs else []
                    step = max(steps) if steps else None
                    step_dir = os.path.join(base, f"step-{step}") if step is not None else None
            else:
                step_dir = os.path.join(ckpt_dir, f"step-{step}")
            if step_dir and os.path.isdir(step_dir):
                with tempfile.TemporaryDirectory() as tmp:
                    out_pt = os.path.join(tmp, "checkpoint.pt")
                    dcp_to_torch_save(step_dir, out_pt)
                    state = torch.load(out_pt, map_location="cpu")
                    state = state.get("model", state)
                # Try loading weights by matching known prefixes
                comp_sd = {k.split("model.compressor.", 1)[1]: v for k, v in state.items() if k.startswith("model.compressor.")}
                up_sd = {k.split("model.upsampler.", 1)[1]: v for k, v in state.items() if k.startswith("model.upsampler.")}
                missing_comp = compress.load_state_dict(comp_sd, strict=False)
                missing_up = upsample.load_state_dict(up_sd, strict=False)
                # It's fine if some keys don't match; we'll run with partial load
        except Exception as _e:
            # Fall back to randomly initialized modules
            pass

    compress = compress.to(device=device, dtype=dtype)
    upsample = upsample.to(device=device, dtype=dtype)
    # If mixers are tied, re-tie after moving to ensure the shared tensor is on the same device
    try:
        if tie_mixers and hasattr(upsample, "mix_up") and hasattr(compress, "mix_down"):
            upsample.mix_up.weight = compress.mix_down.weight.T
    except Exception:
        pass
    return compress, upsample


# --------------------------
# Data
# --------------------------
class LMSeqDataset(Dataset):
    """Trivial dataset wrapper for tokenized text."""
    def __init__(self, tokenizer, texts: List[str], seq_len: int):
        self.tok = tokenizer
        self.texts = texts
        self.seq_len = seq_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        ids = self.tok(self.texts[idx], truncation=True, max_length=self.seq_len+1, padding="max_length", return_tensors="pt")["input_ids"].squeeze(0)
        input_ids = ids[:,]  # [seq_len+1]
        # model expects labels aligned and will shift inside forward; keep length seq_len+1.
        return {"input_ids": input_ids[:-1], "labels": input_ids[:-1].clone()}  # predict next token for all but last


def make_collate(pad_id: int):
    def _collate(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        labels    = torch.stack([b["labels"]    for b in batch], dim=0)
        attn_mask = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}
    return _collate


# --------------------------
# Utilities
# --------------------------
@torch.no_grad()
def nll_loss(model, input_ids, labels, attention_mask) -> torch.Tensor:
    """Compute token-wise NLL averaged over non-pad positions."""
    out = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, return_dict=True)
    # HF-style CausalLM returns CrossEntropy over all tokens; normalize by tokens actually counted.
    loss = out.loss
    return loss.detach()

@torch.no_grad()
def collect_shrunken_states(layer_hidden: torch.Tensor, num_compressed: int) -> torch.Tensor:
    """
    Derive the shrunken hidden-state sequence that Joyce's up-sampler init would see.
    If your real up-sampler init is available, call it here instead.
    """
    B, T, H = layer_hidden.shape
    if num_compressed >= T:
        return layer_hidden
    # Simple uniform pooling as a placeholder for the learned init:
    stride = max(1, T // num_compressed)
    idx = torch.arange(0, T, step=stride, device=layer_hidden.device)[:num_compressed]  # [num_compressed]
    picked = layer_hidden.index_select(1, idx)  # [B, num_compressed, H]
    return picked


def best_ratio_for_delta(deltas: List[float], ratios: List[float], delta_star: float) -> float:
    """
    Pick r* with Δ(r) closest to target Δ*, robust to NaN/Inf. For ties, prefer the
    smallest r (max compression). If all candidates are invalid, fall back to the
    largest r (least compression) to be conservative.
    """
    import math as _m
    # Build clean diffs and ignore NaN/Inf by mapping them to +inf
    diffs = []
    for d in deltas:
        try:
            x = float(d)
        except Exception:
            x = float("inf")
        v = abs(x - float(delta_star))
        if not _m.isfinite(v):
            v = float("inf")
        diffs.append(v)

    best_idx = None
    best_val = float("inf")
    for j, v in enumerate(diffs):
        rj = ratios[j]
        if v < best_val:
            best_val = v
            best_idx = j
        elif v == best_val and best_idx is not None and rj < ratios[best_idx]:
            # tie: prefer smaller r (more compression)
            best_idx = j

    if best_idx is None or not _m.isfinite(best_val):
        # Fallback: choose least compression to avoid destabilizing training
        return ratios[-1]
    return ratios[best_idx]


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--layer_L", type=int, default=12)
    parser.add_argument("--num_compressed_states", type=int, default=64)
    parser.add_argument("--delta_star", type=float, default=0.5)
    parser.add_argument("--ratio_grid", type=str, default="0.05,0.1,0.2,0.4,0.8")
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, choices=["transformer", "mamba2"], default="transformer")
    parser.add_argument(
        "--backbone_config",
        type=str,
        default="llmonade/configs/transformer/t340M.json",
        help="Path to backbone JSON config (used for transformer).",
    )
    parser.add_argument("--joyce_ckpt_dir", type=str, default="exp/joyce_ae_160M/checkpoint", help="Path to Joyce AE DCP checkpoint dir (contains step-*/ directories).")
    parser.add_argument("--joyce_ckpt_step", type=int, default=-1, help="Checkpoint step from joyce_ckpt_dir to load (default: latest).")
    parser.add_argument("--joyce_latent_dim", type=int, default=256, help="Latent dim used by Joyce AE.")
    parser.add_argument("--joyce_use_refine_attn", action="store_true", help="Enable refine attention in upsampler (if trained that way).")
    parser.add_argument("--joyce_no_tie_mixers", action="store_true", help="Disable tying mix up/down weights.")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Compute dtype for backbone (flash-attn requires bf16/fp16).",
    )
    parser.add_argument("--save_dir", type=str, default="controller_ckpt")
    parser.add_argument("--debug_nans", action="store_true", help="Enable NaN debugging mode with anomaly detection")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ratio_grid = [float(x) for x in args.ratio_grid.split(",")]

    # Debug mode setup
    if args.debug_nans:
        torch.autograd.set_detect_anomaly(True)  # for controller backprop path
        # Prefer numerically safer matmul; also disable TF32 if you want bitwise repro
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        # Optional: make runs deterministic for repro
        torch.manual_seed(1234)
        random.seed(1234)
        torch.cuda.manual_seed_all(1234)

    # ---- backbone language model (any ForCausalLM in fla will work) ----
    if args.backbone == "transformer":
        # Prefer loading from a JSON to match your llmonade configs
        cfg_dict = None
        cfg_path = Path(args.backbone_config)
        if cfg_path.exists():
            try:
                cfg_dict = json.loads(cfg_path.read_text())
            except Exception:
                cfg_dict = None
        if cfg_dict is None:
            # Minimal fallback if the JSON isn't available
            cfg_dict = {
                "vocab_size": 50304,
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_heads": 32,
                "max_position_embeddings": 8192,
                "fuse_cross_entropy": False,
            }
        # Ensure we can request hidden states from the backbone
        cfg_dict["output_hidden_states"] = True
        # Avoid fused CE for stability with mixed dtypes during supervision collection
        cfg_dict["fuse_cross_entropy"] = False
        backbone_cfg = TransformerConfig(**cfg_dict)
        backbone = TransformerForCausalLM(backbone_cfg).eval()
    else:
        # Mamba2 option retained for parity
        backbone_cfg = Mamba2Config(
            vocab_size=65536,
            hidden_size=1024,
            num_hidden_layers=24,
            num_heads=16,
            state_size=64,
            conv_kernel=4,
            output_hidden_states=True,  # important
            fuse_cross_entropy=False,
        )
        backbone = Mamba2ForCausalLM(backbone_cfg).eval()  # eval for label collection

    # Select dtype for flash-attn compatibility
    def _pick_dtype() -> torch.dtype:
        if args.dtype == "bf16":
            return torch.bfloat16
        if args.dtype == "fp16":
            return torch.float16
        if args.dtype == "fp32":
            return torch.float32
        # auto
        if device.type == "cuda":
            bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            return torch.bfloat16 if bf16_ok else torch.float16
        return torch.float32

    dtype = _pick_dtype()
    # Move backbone to device (and dtype if CUDA)
    if device.type == "cuda":
        backbone = backbone.to(device=device, dtype=dtype)
    else:
        backbone = backbone.to(device)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", use_fast=True)  # or your tokenizer
    if tokenizer.pad_token_id is None and getattr(tokenizer, "eos_token", None) is not None:
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass
    # Ensure vocab sizes align when using a randomly initialized backbone
    if getattr(tokenizer, "vocab_size", None) and tokenizer.vocab_size != backbone.config.vocab_size:
        try:
            backbone.resize_token_embeddings(tokenizer.vocab_size)
        except Exception:
            pass

    # ---- toy text corpus; replace with your loader ----
    texts = ["This is a short synthetic example for controller training. " * 100] * 500
    dataset = LMSeqDataset(tokenizer, texts, seq_len=args.seq_len)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(pad_id),
        drop_last=True,
    )

    # ---- controller model ----
    ctrl_cfg = JoyceControllerConfig(
        hidden_size=backbone_cfg.hidden_size,
        controller_hidden_size=512,
        num_delta_freqs=32,
        delta_embed_scale=1.0,
        pooler="mean_ln",
        dropout=0.0,
        ratio_min=0.0,
        ratio_max=1.0,
        num_compressed_states=args.num_compressed_states,
    )
    controller = JoyceControllerForCompressionRatio(ctrl_cfg).to(device).train()
    opt = torch.optim.AdamW(controller.parameters(), lr=args.lr)

    # Optionally set up real Joyce compressor/upsampler from checkpoint
    joyce_modules: Optional[Tuple[JoyceCompressionBlock, JoyceUpsamplingBlock]] = None
    try:
        tie_mix = not args.joyce_no_tie_mixers
        joyce_modules = _maybe_load_joyce_modules(
            hidden_size=backbone_cfg.hidden_size,
            num_heads=getattr(backbone_cfg, "num_heads", getattr(backbone_cfg, "num_attention_heads", 16)),
            seq_len=args.seq_len,
            num_compressed=args.num_compressed_states,
            latent_dim=args.joyce_latent_dim,
            norm_eps=getattr(backbone_cfg, "norm_eps", 1e-6),
            mlp_ratio=4.0,
            dropout=0.0,
            use_refine_attn=args.joyce_use_refine_attn,
            tie_mixers=tie_mix,
            ckpt_dir=args.joyce_ckpt_dir,
            ckpt_step=args.joyce_ckpt_step,
            device=device,
            dtype=dtype if device.type == "cuda" else torch.float32,
        )
        # If we have joyce modules, override the compressor forward used in Δ(r) sweep
        def _compress_forward_real(model, input_ids, labels, attention_mask, layer_L, compression_ratio):
            return _compress_block_forward_joyce(
                model=model,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                layer_L=layer_L,
                compression_ratio=compression_ratio,
                joyce_modules=joyce_modules,
            )
        global compress_block_forward
        compress_block_forward = _compress_forward_real
    except Exception:
        # keep fallback
        pass

    # -------------- Phase 1: Collect supervision --------------
    # For each batch:
    #   1) Forward backbone to get Loss(full) and layer L hidden states
    #   2) For each r in ratio_grid: run your *compressed* forward to get Loss(comp[r]) and Δ[r]
    #   3) Pick r* s.t. Δ[r*] ~ Δ* (closest)  -> label
    #   4) Produce shrunken_states that Joyce's up-sampler init would see (from the same layer-L states)
    #   5) Train controller on (shrunken_states, Δ*) -> r*
    #
    # Note: If you already have an offline table of (Δ, r*) for your prompts, skip collection.
    step = 0
    skips_bad_sweep = 0
    skips_nonfinite_full = 0
    data_iter = cycle(loader)  # infinite iterator over batches

    def _t(): return time.perf_counter()

    while step < args.train_steps:
        batch = next(data_iter)

        # Move
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # === Diagnose full forward causes ===
        cause, cinfo = diagnose_full_forward(
            model=backbone,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            ignore_idx=-100,
            probe_logits=True,  # set False if you want zero overhead here
        )

        if cause != FullLossCause.OK:
            # One-line, explicit reason:
            print(f"[cause] full_forward={cause.name} "
                  f"T={cinfo.get('T')} counted={cinfo.get('counted_tokens')} "
                  f"label_min={cinfo.get('label_min')} label_max={cinfo.get('label_max')} "
                  f"V={cinfo.get('V')} logits_dtype={cinfo.get('logits_dtype')} "
                  f"nNaN={cinfo.get('n_nan_logits',0)} nInf={cinfo.get('n_inf_logits',0)}")
            # If you want to *skip* this batch, keep your existing skip+counter here:
            skips_nonfinite_full += 1
            continue

        # Build pad-safe labels for the actual full loss compute (same logic as in the detector)
        ignore_idx = -100
        labels_full = labels.clone()
        if attention_mask is not None:
            ignore_new = (attention_mask == 0)
            apply_on_src = torch.roll(ignore_new, shifts=1, dims=1)
            apply_on_src[:, 0] = False
            labels_full[apply_on_src] = ignore_idx

        # Now the actual full forward
        t0 = _t()
        out_full = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_full,
            output_hidden_states=True,
            return_dict=True,
        )
        t1 = _t()
        loss_full = out_full.loss.detach()
        if not torch.isfinite(loss_full):
            # Extremely rare: logits were finite but CE still blew up (classify as UNKNOWN)
            print(f"[cause] full_forward={FullLossCause.UNKNOWN_NONFINITE.name} "
                  f"T={cinfo.get('T')} counted={cinfo.get('counted_tokens')}")
            skips_nonfinite_full += 1
            continue
        if (step % 10) == 0:
            print(f"[timer] full_forward={t1-t0:.3f}s  step={step}")
        layer_hid = out_full.hidden_states[args.layer_L]  # [B, T, H]
        # Guard against rare NaNs/Infs in fused kernels; sanitize before using as features
        if not torch.isfinite(layer_hid).all():
            layer_hid = torch.nan_to_num(layer_hid, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) Δ for each ratio r
        if compress_block_forward is None:
            # If your compressor isn't wired yet, skip training on this step (no noisy labels).
            continue

        t2 = _t()
        deltas = []
        bad_entries = 0
        for r in ratio_grid:
            t_r0 = _t()
            loss_comp_r = compress_block_forward(
                model=backbone,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                layer_L=args.layer_L,
                compression_ratio=r,
            )
            t_r1 = _t()
            if (step % 10) == 0:
                print(f"[timer] r={r:.3f} comp_forward={t_r1-t_r0:.3f}s")
            
            if not torch.isfinite(loss_comp_r):
                # Mark this ratio as unusable; record +inf delta and keep going
                deltas.append(float("inf"))
                bad_entries += 1
                continue
            d = (loss_comp_r - loss_full).item()
            if not math.isfinite(d):
                deltas.append(float("inf"))
                bad_entries += 1
            else:
                deltas.append(d)

        t3 = _t()
        if (step % 10) == 0:
            print(f"[timer] delta_sweep={t3-t2:.3f}s (bad_entries={bad_entries})")

        # If *all* ratios are bad, skip this batch but COUNT it
        if all([not math.isfinite(x) for x in deltas]):
            skips_bad_sweep += 1
            if (step % 10) == 0:
                print(f"[skip] all Δ(r) non-finite this batch (total skips: {skips_bad_sweep})")
            continue

        # 3) Choose r* for target Δ*
        r_star = best_ratio_for_delta(deltas, ratio_grid, args.delta_star)

        # 4) Build shrunken states
        with torch.no_grad():
            shrunken_states = collect_shrunken_states(layer_hid, args.num_compressed_states)  # [B, t, H]
            # Sanitize and cast features for controller
            shrunken_states = torch.nan_to_num(shrunken_states, nan=0.0, posinf=0.0, neginf=0.0)
            shrunken_states = torch.clamp(shrunken_states, min=-1e3, max=1e3)
            shrunken_states = shrunken_states.to(controller.dtype)

        # 5) Train controller: (shrunken_states, Δ*) -> r*
        delta_star_batch = torch.full((input_ids.size(0),), float(args.delta_star), device=device)
        target_ratio = torch.full((input_ids.size(0),), float(r_star), device=device)

        out_ctrl = controller(
            shrunken_states=shrunken_states,
            delta=delta_star_batch,
            target_ratio=target_ratio,
            return_dict=True,
        )
        loss = out_ctrl.loss
        # Safety: skip update on non-finite loss/predictions
        if not torch.isfinite(loss):
            # Optional: print a one-line notice each 50 steps to avoid spam
            if step % 50 == 0:
                print("[warn] non-finite controller loss; skipping step")
            step += 1
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"[{step}] loss={loss.item():.4f}  r*={r_star:.3f}  mean_pred={out_ctrl.ratio.mean().item():.3f} "
                  f"(skips: full={skips_nonfinite_full}, sweep={skips_bad_sweep})")
        step += 1

    # Save controller
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    controller.save_pretrained(out_dir.as_posix())
    print(f"Saved controller to: {out_dir}")

if __name__ == "__main__":
    main()
