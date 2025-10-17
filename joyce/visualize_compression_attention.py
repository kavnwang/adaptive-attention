#!/usr/bin/env python3
"""
Visualize compressor attention over vision patches for each image.

For each image entry in a CLEVR-style annotations JSON, this script:
  1) Runs the vision encoder to get patch features
  2) Projects to the text model hidden size if needed
  3) Builds the compressor queries ("newly generated" compression tokens)
  4) Computes attention weights from compression tokens (queries) to input patches (keys)
  5) Saves per-token attention heatmaps overlaid on the image (and raw .npy arrays)

Notes
- This uses the same AdaptiveAttention compressor as training. The compressor can be
  untrained (random) or loaded from a checkpoint trained with train_visual_compressor.py.
- The attention computation mirrors the math in forward_compress, but uses a
  plain PyTorch softmax to obtain weights per head, then averages across heads.
- Defaults to sampling a subset of compression tokens for visualization to keep
  the number of output images reasonable. You can set --max_tokens_per_image -1 to
  visualize all.

Example
  python visualize_compression_attention.py \
    --text_model_path exp/transformer_160M \
    --annotations_json data/clevr-mini/clevr_summaries.json \
    --images_root data/clevr-mini \
    --out_dir attn_maps \
    --max_images 10 \
    --max_tokens_per_image 16

Optionally load a trained visual compressor checkpoint (from train_visual_compressor.py):
  python visualize_compression_attention.py \
    --text_model_path exp/transformer_160M \
    --load_text_checkpoint exp/visual_compressor_training/checkpoint_step_XXXX.pt \
    --annotations_json data/clevr-mini/clevr_summaries.json \
    --images_root data/clevr-mini
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from fla.layers.adaptive import AdaptiveAttention
from fla.models.transformer import TransformerForCausalLM, TransformerConfig


def add_compressor_to_layer(
    model: nn.Module,
    layer_idx: int,
    compress_deterministic: bool = True,
    compress_num_tokens: int = 32,
) -> None:
    layer = model.model.layers[layer_idx]
    config = model.config

    compressor = AdaptiveAttention(
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        qkv_bias=getattr(config, 'qkv_bias', False),
        qk_norm=getattr(config, 'qk_norm', False),
        window_size=getattr(config, 'window_size', None),
        rope_theta=getattr(config, 'rope_theta', 10000.0),
        max_position_embeddings=config.max_position_embeddings,
        layer_idx=layer_idx,
        compress_deterministic=compress_deterministic,
        compress_num_tokens=compress_num_tokens,
        intermediate_size=config.intermediate_size,
    )

    layer.compressor = compressor


def build_text_model(text_model_path: str, layer_idx: int, compress_num_tokens: int, device: torch.device) -> nn.Module:
    config_path = os.path.join(text_model_path, "model_config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    text_config = TransformerConfig(**config_dict)
    model = TransformerForCausalLM(text_config)
    # Add compressor before moving to device
    add_compressor_to_layer(model, layer_idx, compress_deterministic=True, compress_num_tokens=compress_num_tokens)
    model = model.to(device=device, dtype=torch.bfloat16)
    return model


def maybe_load_text_checkpoint(model: nn.Module, ckpt_path: Optional[str]) -> Dict:
    """Load a compressor/text checkpoint into the text model; return the flat state for optional extras.

    Returns the chosen state dict (flat) so callers can inspect e.g. a trained vision_proj.
    """
    if not ckpt_path:
        return {}
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Support multiple key layouts
    state = ckpt.get('model_state_dict') or ckpt.get('text_model_state_dict') or ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"  Missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")
    return state


class VisionHooks:
    def __init__(self, model: nn.Module, hidden_size: int):
        self.model = model
        self.hidden_size = hidden_size
        self.projector = None

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        m = self.model
        original_dims = pixel_values.dim()
        if original_dims == 5:
            b, n, c, h, w = pixel_values.shape
            pixel_values_flat = pixel_values.view(b * n, c, h, w)
        else:
            pixel_values_flat = pixel_values

        if hasattr(m, 'vision_tower') and m.vision_tower is not None:
            out = m.vision_tower(pixel_values_flat, output_hidden_states=True)
            feats = out.last_hidden_state
        elif hasattr(m, 'vision_model') and m.vision_model is not None:
            out = m.vision_model(pixel_values_flat, output_hidden_states=True)
            feats = out.last_hidden_state
        else:
            raise RuntimeError("Could not locate a vision encoder on the SmolVLM2 model.")

        if original_dims == 5:
            feats = feats.view(b, n * feats.size(1), feats.size(2))

        return feats


def build_vision(name_or_path: str, device: torch.device, dtype: torch.dtype, target_hidden_size: int) -> Tuple[VisionHooks, AutoImageProcessor]:
    vlm = AutoModel.from_pretrained(name_or_path).to(device=device, dtype=dtype)
    for p in vlm.parameters():
        p.requires_grad = False
    image_processor = AutoImageProcessor.from_pretrained(name_or_path)
    hooks = VisionHooks(model=vlm, hidden_size=target_hidden_size)
    return hooks, image_processor


def ensure_projection(feats: torch.Tensor, target_hidden: int) -> Tuple[torch.Tensor, Optional[nn.Module]]:
    B, S, D = feats.shape
    if D == target_hidden:
        return feats, None
    proj = nn.Linear(D, target_hidden, bias=True).to(device=feats.device, dtype=feats.dtype)
    with torch.no_grad():
        nn.init.normal_(proj.weight, std=0.02)
        nn.init.zeros_(proj.bias)
    return proj(feats), proj


def compute_compressor_attention(
    compressor: AdaptiveAttention,
    feats: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute attention weights (head-averaged) from compressor queries to input patches.

    Args:
      compressor: AdaptiveAttention module (with q_proj/k_proj and rotary)
      feats: (B, S, D) input patch embeddings (already projected to hidden size)
      attention_mask: (B, S) 1 for valid tokens; optional

    Returns:
      attn_mean: (B, S_q, S_k) where S_q == S_k == active seq len per example
    """
    B, S, D = feats.shape
    device = feats.device

    # Active lengths per batch item
    if attention_mask is None:
        s_lens = torch.full((B,), S, device=device, dtype=torch.long)
    else:
        s_lens = attention_mask.sum(dim=-1).to(torch.long)

    # Appended compressor tokens (queries): one per input token
    appended = compressor.compress_token.view(1, 1, -1).expand(B, S, D).contiguous()

    # Linear projections to heads
    head_dim = compressor.head_dim
    q = torch.einsum('bnd,df->bnf', appended.to(feats.dtype), compressor.q_proj.weight.T) + (
        compressor.q_proj.bias if compressor.q_proj.bias is not None else 0
    )
    k = torch.einsum('bnd,df->bnf', feats, compressor.k_proj.weight.T) + (
        compressor.k_proj.bias if compressor.k_proj.bias is not None else 0
    )
    q = q.view(B, S, compressor.num_heads, head_dim)
    k = k.view(B, S, compressor.num_heads, head_dim)

    if compressor.qk_norm:
        q = compressor.q_norm(q)
        k = compressor.k_norm(k)

    # Rotary positional embedding
    # Q positions: [S .. 2S-1], K positions: [0 .. S-1]
    q, _ = compressor.rotary(q, q, seqlen_offset=S, max_seqlen=2*S, cu_seqlens=None)
    k, _ = compressor.rotary(k, k, seqlen_offset=0, max_seqlen=2*S, cu_seqlens=None)

    # Compute attention logits and softmax over keys
    # logits: (B, H, S_q, S_k)
    qt = q.permute(0, 2, 1, 3).contiguous()  # (B, H, S, d)
    kt = k.permute(0, 2, 1, 3).contiguous()  # (B, H, S, d)
    scale = float(head_dim) ** -0.5
    logits = torch.matmul(qt.float(), kt.float().transpose(-1, -2)) * scale

    if attention_mask is not None:
        # Mask out invalid keys by setting to large negative
        # Expand to (B, 1, 1, S_k)
        key_mask = (attention_mask == 1).unsqueeze(1).unsqueeze(2)
        logits = logits.masked_fill(~key_mask, -1e9)

    attn = torch.softmax(logits, dim=-1)  # (B, H, S_q, S_k)
    attn_mean = attn.mean(dim=1)  # (B, S_q, S_k)
    return attn_mean


def _vec_to_grid(attn_vec: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
    """Convert a 1D attention vector over patches into a (H, W) grid.

    Supports processors that tile images, producing N * (H*W) tokens.
    In that case, average over the tiling dimension to return a single grid.
    """
    S_tile = patch_h * patch_w
    if attn_vec.size == S_tile:
        grid = attn_vec.reshape(patch_h, patch_w)
    elif attn_vec.size % S_tile == 0 and attn_vec.size > S_tile:
        n = attn_vec.size // S_tile
        grid = attn_vec.reshape(n, patch_h, patch_w).mean(axis=0)
    else:
        raise ValueError(
            f"Attention length {attn_vec.size} incompatible with grid {patch_h}x{patch_w}"
        )
    return grid


def overlay_and_save(
    image_path: str,
    attn_vec: np.ndarray,
    patch_h: int,
    patch_w: int,
    out_path: str,
    cmap: str = 'jet',
    alpha: float = 0.45,
):
    # Load image
    img = Image.open(image_path).convert('RGB')
    W, H = img.size

    # Reshape attention vector to grid and upscale to image size
    grid = _vec_to_grid(attn_vec, patch_h, patch_w)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

    # Create heatmap with matplotlib, then overlay with alpha
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(img)
    ax.imshow(grid, cmap=cmap, interpolation='nearest', extent=[0, W, H, 0], alpha=alpha)
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Visualize compressor attention over image patches")
    # Models
    ap.add_argument('--text_model_path', type=str, default='exp/transformer_160M')
    ap.add_argument('--load_text_checkpoint', type=str, default=None,
                    help='Optional .pt checkpoint with compressor weights (from train_compressor.py)')
    ap.add_argument('--vision_model', type=str, default='HuggingFaceTB/SmolVLM2-500M-Instruct')
    ap.add_argument('--layer_idx', type=int, default=6)
    ap.add_argument('--compress_num_tokens', type=int, default=32)

    # Data
    ap.add_argument('--annotations_json', type=str, required=True)
    ap.add_argument('--images_root', type=str, default=None)
    ap.add_argument('--max_images', type=int, default=None)

    # Output
    ap.add_argument('--out_dir', type=str, default='attn_maps')
    ap.add_argument('--max_tokens_per_image', type=int, default=16,
                    help='-1 for all queries (can be large)')
    ap.add_argument('--cmap', type=str, default='jet')
    ap.add_argument('--alpha', type=float, default=0.45)

    # System
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load text model + compressor
    print('Loading text model and compressor...')
    text_model = build_text_model(args.text_model_path, args.layer_idx, args.compress_num_tokens, device)
    text_model.eval()
    ckpt_state = {}
    if args.load_text_checkpoint:
        ckpt_state = maybe_load_text_checkpoint(text_model, args.load_text_checkpoint)
    compressor: AdaptiveAttention = text_model.model.layers[args.layer_idx].compressor

    # Load vision encoder
    print('Loading vision encoder...')
    vision_hooks, image_processor = build_vision(
        args.vision_model, device=device, dtype=torch.bfloat16, target_hidden_size=text_model.config.hidden_size
    )
    vision_hooks.model.eval()

    # Prepare output directory
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(args.annotations_json, 'r') as f:
        data = json.load(f)
    if args.max_images is not None:
        data = data[: args.max_images]

    images_root = Path(args.images_root) if args.images_root else None

    print(f"Processing {len(data)} images...")

    for idx, entry in enumerate(data, 1):
        rel_path = Path(entry['image'])
        img_path = rel_path if rel_path.is_absolute() or images_root is None else images_root / rel_path
        if not Path(img_path).exists():
            print(f"[warn] Missing image: {img_path}; skipping")
            continue

        try:
            # Prepare pixels
            img = Image.open(img_path).convert('RGB')
            pixel_inputs = image_processor(images=img, return_tensors='pt')
            pixel_values = pixel_inputs['pixel_values'].to(device=device, dtype=torch.bfloat16)

            # Determine patch grid size from processed shape and model patch size
            if pixel_values.dim() == 5:
                pv = pixel_values.view(-1, *pixel_values.shape[-3:])
            else:
                pv = pixel_values
            _, C, H, W = pv.shape
            # Try to get patch size from model config
            if hasattr(vision_hooks.model, 'vision_model') and hasattr(vision_hooks.model.vision_model, 'patch_size'):
                patch_size = int(vision_hooks.model.vision_model.patch_size)
            else:
                patch_size = int(getattr(vision_hooks.model.config.vision_config, 'patch_size', 16))
            ph, pw = H // patch_size, W // patch_size

            # Vision features
            with torch.no_grad():
                feats = vision_hooks.forward_features(pixel_values)  # (1, S, Dv)

            # If a trained vision_proj exists in the checkpoint, apply it; else use a fresh projection
            if ckpt_state:
                # Find a matching key
                vp_w_key = None
                for key in ckpt_state.keys():
                    if key.endswith('vision_proj.weight'):
                        vp_w_key = key
                        break
                if vp_w_key is not None:
                    W = ckpt_state[vp_w_key]
                    b = ckpt_state.get(vp_w_key.replace('weight', 'bias'), None)
                    out_features, in_features = W.shape[0], W.shape[1]
                    # If the dims match current feats, use it
                    if feats.size(-1) == in_features:
                        vision_proj = nn.Linear(in_features, out_features, bias=b is not None).to(device=device, dtype=feats.dtype)
                        with torch.no_grad():
                            vision_proj.weight.copy_(W.to(dtype=feats.dtype))
                            if b is not None:
                                vision_proj.bias.copy_(b.to(dtype=feats.dtype))
                        feats = vision_proj(feats)
                    else:
                        # Fallback to simple projection if dims mismatch
                        feats, _ = ensure_projection(feats, target_hidden=text_model.config.hidden_size)
                else:
                    feats, _ = ensure_projection(feats, target_hidden=text_model.config.hidden_size)
            else:
                feats, _ = ensure_projection(feats, target_hidden=text_model.config.hidden_size)
            attention_mask = torch.ones(feats.size()[:2], device=device, dtype=torch.long)

            # Compute attention maps (no grad)
            with torch.no_grad():
                attn_mean = compute_compressor_attention(compressor, feats, attention_mask=attention_mask)  # (1, S, S)
            attn_np = attn_mean[0].detach().to(torch.float32).cpu().numpy()  # (S, S)

            # Select which query rows to visualize
            S = attn_np.shape[0]
            if args.max_tokens_per_image is None or args.max_tokens_per_image < 0 or args.max_tokens_per_image >= S:
                q_indices = list(range(S))
            else:
                # spread samples across the sequence for coverage
                step = max(1, S // args.max_tokens_per_image)
                q_indices = list(range(0, S, step))[: args.max_tokens_per_image]

            # Create per-image directory
            img_out_dir = out_root / Path(img_path).stem
            img_out_dir.mkdir(parents=True, exist_ok=True)

            # Save raw attention matrix
            np.save(img_out_dir / 'attn_patches.npy', attn_np)

            # Render overlays for selected queries
            for q in q_indices:
                vec = attn_np[q]  # (S,)
                out_path = img_out_dir / f"attn_q{q:04d}.png"
                overlay_and_save(str(img_path), vec, ph, pw, str(out_path), cmap=args.cmap, alpha=args.alpha)

            # Also save a small montage of the first few queries
            montage = [_vec_to_grid(attn_np[q], ph, pw) for q in q_indices[:min(9, len(q_indices))]]
            if montage:
                fig, axes = plt.subplots(1, len(montage), figsize=(3 * len(montage), 3))
                if len(montage) == 1:
                    axes = [axes]
                for ax, grid in zip(axes, montage):
                    ax.imshow(grid, cmap=args.cmap)
                    ax.axis('off')
                fig.tight_layout()
                fig.savefig(img_out_dir / 'montage.png', bbox_inches='tight')
                plt.close(fig)

            if idx % 10 == 0:
                print(f"Processed {idx}/{len(data)}: {img_path}")
        except Exception as e:
            print(f"[error] Failed on {img_path}: {e}")

    print(f"Done. Outputs in {out_root}")


if __name__ == '__main__':
    main()
