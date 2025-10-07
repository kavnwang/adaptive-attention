#!/usr/bin/env python3
"""
Train visual compressor: compress image features from a VLM (SmolVLM2-500M) and
inject the compressed memory tokens into a text LM at layer L for summary generation.

Pipeline (mirrors train_compressor.py but with images):
1) Load pre-trained text LM (TransformerForCausalLM) from exp/transformer_160M
2) Load pre-trained VLM (SmolVLM2-500M) and freeze it; extract vision features
3) Add AdaptiveAttention compressor at text layer L (trained from scratch)
4) For each image: run vision encoder, optionally project to text hidden size, compress
5) For each image summary: inject compressed memory at layer L, train on summary
6) Backprop accumulates gradients from summaries into compressor (and optionally the vision->text proj and upper text layers)

CLEVR data format expected:
  annotations_json: JSON with list of items: {"image": "relative/or/absolute/path/to/image.png", "summaries": ["...", "..."]}
  images are loaded from disk and processed with the chosen VLM's image processor.

Note:
- This script tries to support SmolVLM2-500M vision encoder via Transformers. It uses heuristics to find the
  vision tower and any multimodal projector. If it cannot auto-detect, it will raise a clear error indicating
  where to adapt the integration.
- The text LM is our existing TransformerForCausalLM (FLA). Memory injection matches train_compressor.py exactly.
"""

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from fla.layers.adaptive import AdaptiveAttention, check_compressor_gradients
from fla.models.transformer import TransformerForCausalLM, TransformerConfig
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


class ClevrSummariesDataset(Dataset):
    """
    Dataset for CLEVR-like image + summaries.

    annotations_json format: list of entries, each with keys:
      - image: str (path to image)
      - summaries: List[str] (one or more target summaries)
    """

    def __init__(
        self,
        annotations_json: str,
        images_root: Optional[str],
        tokenizer,
        image_processor,
        max_summary_len: int = 64,
        max_summaries_per_image: Optional[int] = None,
        num_items: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_summary_len = max_summary_len
        self.max_summaries_per_image = max_summaries_per_image

        with open(annotations_json, 'r') as f:
            data = json.load(f)
        if num_items is not None:
            data = data[:num_items]

        self.items = []
        images_root = Path(images_root) if images_root else None
        for entry in data:
            img_path = Path(entry['image'])
            if images_root is not None and not img_path.is_absolute():
                img_path = images_root / img_path
            summaries = entry.get('summaries', [])
            if self.max_summaries_per_image is not None:
                summaries = summaries[: self.max_summaries_per_image]
            self.items.append({
                'image_path': str(img_path),
                'summaries': summaries,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(item['image_path']).convert('RGB')
        pixel_inputs = self.image_processor(images=img, return_tensors='pt')
        pixel_values = pixel_inputs['pixel_values'].squeeze(0)  # (C, H, W)

        # Tokenize each summary
        summary_encodings = []
        for text in item['summaries']:
            enc = self.tokenizer(
                text,
                max_length=self.max_summary_len,
                truncation=True,
                return_tensors='pt'
            )
            summary_encodings.append({
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0)
            })

        return {
            'image_path': item['image_path'],
            'pixel_values': pixel_values,
            'summaries': summary_encodings,
        }


def collate_images_fn(batch):
    image_paths = [it['image_path'] for it in batch]
    pixel_values = torch.stack([it['pixel_values'] for it in batch], dim=0)
    summaries = [it['summaries'] for it in batch]  # List[List[encodings]]

    return {
        'image_paths': image_paths,
        'pixel_values': pixel_values,
        'summaries': summaries,
    }


class MemoryDebugger:
    """Save memory states and decode their vocab projections for inspection."""

    def __init__(
        self,
        save_dir: str,
        tokenizer,
        top_k: int = 5,
        every: int = 100,
        max_dumps: int = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.every = max(1, every)
        self.max_dumps = max_dumps if max_dumps is None or max_dumps > 0 else None
        self._dump_count = 0

    def maybe_log(
        self,
        step: int,
        image_paths: List[str],
        memory_states: torch.Tensor,
        memory_mask: torch.Tensor,
        model: nn.Module,
    ) -> None:
        if step % self.every != 0:
            return
        if self.max_dumps is not None and self._dump_count >= self.max_dumps:
            return

        self._dump_count += 1
        step_dir = self.save_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            mem = memory_states.detach()
            mask = memory_mask.detach()

            payload = {
                'step': step,
                'image_paths': image_paths,
                'memory_states': mem.to(dtype=torch.float32).cpu(),
                'memory_mask': mask.to(dtype=torch.int64).cpu(),
            }
            torch.save(payload, step_dir / "memory_states.pt")

            logits = model.lm_head(mem).float()
            k = min(self.top_k, logits.size(-1))
            top_values, top_indices = torch.topk(logits, k=k, dim=-1)

            decoded_records = []
            text_lines = []
            for idx, img_path in enumerate(image_paths):
                valid = int(mask[idx].sum().item())
                if valid == 0:
                    continue

                argmax_token_ids = top_indices[idx, :valid, 0].tolist()
                argmax_text = self.tokenizer.decode(argmax_token_ids)

                positions = []
                for pos in range(valid):
                    top_tokens = []
                    for rank in range(k):
                        token_id = int(top_indices[idx, pos, rank].item())
                        token_text = self.tokenizer.decode([token_id]).replace("\n", "\\n")
                        logit_val = float(top_values[idx, pos, rank].item())
                        top_tokens.append({'token_id': token_id, 'token': token_text, 'logit': logit_val})
                    positions.append({'position': pos, 'top_tokens': top_tokens})

                decoded_records.append({
                    'image_path': img_path,
                    'valid_memory_tokens': valid,
                    'argmax_token_ids': argmax_token_ids,
                    'argmax_text': argmax_text,
                    'positions': positions,
                })

                text_lines.append(f"image_path: {img_path}")
                text_lines.append(f"valid_memory_tokens: {valid}")
                text_lines.append(f"argmax_text: {argmax_text}")
                text_lines.append("")

            if decoded_records:
                with open(step_dir / "decoded.json", 'w') as f:
                    json.dump(decoded_records, f, indent=2)
                (step_dir / "decoded.txt").write_text("\n".join(text_lines))


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
    print(f"Added compressor to layer {layer_idx} (deterministic={compress_deterministic}, num_tokens={compress_num_tokens})")


def setup_optimizer(
    model: nn.Module,
    layer_idx: int,
    compressor_lr: float = 5e-4,
    upper_layers_lr: float = 1e-4,
    weight_decay: float = 0.1,
    freeze_bottom: bool = True,
    extra_trainable: Optional[List[nn.Parameter]] = None,
    only_compressor: bool = False,
) -> torch.optim.Optimizer:
    param_groups = []

    if freeze_bottom:
        for i in range(layer_idx):
            for p in model.model.layers[i].parameters():
                p.requires_grad = False
        print(f"Froze layers 0..{layer_idx-1}")

    if only_compressor:
        # Freeze everything first
        for p in model.parameters():
            p.requires_grad = False

    compressor = model.model.layers[layer_idx].compressor
    # Ensure compressor requires grad
    for p in compressor.parameters():
        p.requires_grad = True
    param_groups.append({"params": compressor.parameters(), "lr": compressor_lr, "name": "compressor"})

    if not only_compressor:
        upper_params = []
        for i in range(layer_idx, len(model.model.layers)):
            for name, p in model.model.layers[i].named_parameters():
                if 'compressor' not in name and p.requires_grad:
                    upper_params.append(p)
        if upper_params:
            param_groups.append({"params": upper_params, "lr": upper_layers_lr, "name": "upper_layers"})

    # Optionally add extra modules (e.g., vision->text projection)
    if (not only_compressor) and extra_trainable:
        param_groups.append({"params": extra_trainable, "lr": compressor_lr, "name": "vision_proj"})

    if not only_compressor:
        param_groups.append({"params": model.lm_head.parameters(), "lr": upper_layers_lr, "name": "lm_head"})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    print("\nOptimizer setup:")
    print(f"  Compressor LR: {compressor_lr}")
    print(f"  Upper layers LR: {upper_layers_lr}")
    print(f"  Weight decay: {weight_decay}")
    if only_compressor:
        print("  Mode: compressor-only training (all other params frozen)")
    if extra_trainable and not only_compressor:
        print("  Extra trainable modules: vision_proj")
    return optimizer


def run_to_layer(model, input_ids, attention_mask, end_layer_exclusive):
    x = model.model.embeddings(input_ids)
    for i in range(end_layer_exclusive):
        layer = model.model.layers[i]
        x = layer(x, attention_mask=attention_mask)[0]
    return x


@dataclass
class VisionHooks:
    model: nn.Module
    projector: Optional[nn.Module]
    hidden_size: int

    @torch.no_grad()
    def infer_dim(self, pixel_values: torch.Tensor) -> int:
        feats = self.forward_features(pixel_values)
        return feats.size(-1)

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Try common attributes used by LLaVA-like and SmolVLM2 models
        m = self.model

        # Some vision stacks (e.g., SmolVLM2) may expect 4D pixel inputs (B, C, H, W).
        # Their processors can return 5D (B, N, C, H, W) even for single-image inputs.
        # Flatten the extra image dimension if present, and merge it back into sequence afterward.
        original_dims = pixel_values.dim()
        if original_dims == 5:
            b, n, c, h, w = pixel_values.shape
            pixel_values_flat = pixel_values.view(b * n, c, h, w)
        else:
            pixel_values_flat = pixel_values

        # Case 1: vision_tower present
        if hasattr(m, 'vision_tower') and m.vision_tower is not None:
            out = m.vision_tower(pixel_values_flat, output_hidden_states=True)
            feats = out.last_hidden_state  # (B', S, Dv)
        elif hasattr(m, 'vision_model') and m.vision_model is not None:
            out = m.vision_model(pixel_values_flat, output_hidden_states=True)
            feats = out.last_hidden_state  # (B', S, Dv)
        else:
            raise RuntimeError("Could not locate a vision encoder on the SmolVLM2 model. Please adapt VisionHooks.forward_features.")

        # If we flattened an extra image dimension, merge it back into the sequence axis
        if original_dims == 5:
            Bp, S, Dv = feats.shape  # B' = b * n
            feats = feats.view(b, n * S, Dv)

        if self.projector is not None:
            feats = self.projector(feats)
        return feats


def build_vision_encoder(
    name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
    target_hidden_size: int,
) -> Tuple[VisionHooks, AutoImageProcessor]:
    vlm = AutoModel.from_pretrained(name_or_path).to(device=device, dtype=dtype)
    for p in vlm.parameters():
        p.requires_grad = False

    # Try find an existing projector to LLM hidden size
    projector = None
    proj = None
    # Common attributes in LLaVA-like models
    for attr_name in [
        'mm_projector', 'multi_modal_projector', 'visual_projection', 'vision_projection', 'projector'
    ]:
        if hasattr(vlm, attr_name):
            candidate = getattr(vlm, attr_name)
            if isinstance(candidate, nn.Module):
                proj = candidate
                break

    vision_dim = None
    # Probe a dummy image? We cannot forge shapes reliably without running; rely on module structure
    # If projector exists, we attempt to inspect last Linear for out_features
    if proj is not None:
        # Try to guess output dim of projector
        last_linear = None
        for m in reversed(list(proj.modules())):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is not None and last_linear.out_features == target_hidden_size:
            projector = proj
        else:
            # Not matching our text hidden size; we will attach a new linear after features
            projector = None

    image_processor = AutoImageProcessor.from_pretrained(name_or_path)

    hooks = VisionHooks(model=vlm, projector=projector, hidden_size=target_hidden_size)
    return hooks, image_processor


def ensure_projection(feats: torch.Tensor, target_hidden: int) -> Tuple[torch.Tensor, Optional[nn.Module]]:
    B, S, D = feats.shape
    if D == target_hidden:
        return feats, None
    # Create a simple linear projector on the fly
    proj = nn.Linear(D, target_hidden, bias=True).to(device=feats.device, dtype=feats.dtype)
    with torch.no_grad():
        nn.init.normal_(proj.weight, std=0.02)
        nn.init.zeros_(proj.bias)
    return proj(feats), proj


def train_step(
    text_model: nn.Module,
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    layer_idx: int,
    device: torch.device,
    vision_hooks: VisionHooks,
    compression_tokens: Optional[int] = None,
    compression_depth: int = 0,
    gradient_clip: float = 1.0,
    memory_debugger: Optional[MemoryDebugger] = None,
    global_step: Optional[int] = None,
    extra_proj_holder: Dict[str, nn.Module] = None,
    freeze_new_projection: bool = False,
) -> Tuple[float, int]:
    text_model.train()

    image_paths = batch['image_paths']
    model_dtype = next(vision_hooks.model.parameters()).dtype
    pixel_values = batch['pixel_values'].to(device=device, dtype=model_dtype)
    summaries = batch['summaries']

    B = pixel_values.size(0)
    compressor = text_model.model.layers[layer_idx].compressor

    # 1) Vision forward (frozen) -> features -> optional projection -> compress
    with torch.enable_grad():
        feats = vision_hooks.forward_features(pixel_values)  # (B, S, Dv or D_text)
        # Apply an attached projection if present; else create one if needed
        if hasattr(text_model, 'vision_proj') and isinstance(text_model.vision_proj, nn.Module):
            feats = text_model.vision_proj(feats)
            local_proj = None
        else:
            feats, local_proj = ensure_projection(feats, target_hidden=text_model.config.hidden_size)
            if local_proj is not None:
                # Optionally freeze newly created projection
                if freeze_new_projection:
                    for p in local_proj.parameters():
                        p.requires_grad = False
                # Attach to text_model for persistence and later usage
                setattr(text_model, 'vision_proj', local_proj)
                if extra_proj_holder is not None and 'vision_proj' not in extra_proj_holder:
                    extra_proj_holder['vision_proj'] = local_proj

        # Build an attention mask of ones (vision features are all valid)
        vision_mask = torch.ones(feats.size()[:2], device=device, dtype=torch.long)

        M_batch, _, meta_batch = compressor.forward_compress(
            feats,
            attention_mask=vision_mask,
            num_tokens=compression_tokens,
            depth=compression_depth,
            return_padded=True,
        )  # (B, m, D)
        M_mask_batch = meta_batch["mask_padded"]

    if memory_debugger is not None and global_step is not None:
        memory_debugger.maybe_log(
            step=global_step,
            image_paths=image_paths,
            memory_states=M_batch,
            memory_mask=M_mask_batch,
            model=text_model,
        )

    # 2) For each summary per image: inject memory into text at layer L
    total_loss = 0.0
    total_summaries = 0
    loss_fct = nn.CrossEntropyLoss()

    for b in range(B):
        M = M_batch[b:b+1]
        M_mask = M_mask_batch[b:b+1]
        image_summaries = summaries[b]

        for sdata in image_summaries:
            hl_ids = sdata['input_ids'].unsqueeze(0).to(device)  # (1, T)
            hl_mask = sdata['attention_mask'].unsqueeze(0).to(device)  # (1, T)

            input_ids = hl_ids[:, :-1]
            labels = hl_ids[:, 1:]
            input_mask = hl_mask[:, :-1]
            if input_ids.size(1) == 0:
                continue

            H_text = run_to_layer(text_model, input_ids, input_mask, layer_idx)
            H_with_mem = torch.cat([M, H_text], dim=1)
            mask_with_mem = torch.cat([M_mask, input_mask], dim=1)

            hidden = H_with_mem
            for i in range(layer_idx, len(text_model.model.layers)):
                layer = text_model.model.layers[i]
                hidden = layer(hidden, attention_mask=mask_with_mem)[0]

            hidden = text_model.model.norm(hidden)
            hidden_hl = hidden[:, M.size(1):, :]
            logits = text_model.lm_head(hidden_hl)

            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss
            total_summaries += 1

    if total_summaries == 0:
        return 0.0, 0

    avg_loss = total_loss / total_summaries
    optimizer.zero_grad(set_to_none=True)
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(text_model.parameters(), gradient_clip)
    optimizer.step()

    return avg_loss.item(), total_summaries


def main():
    parser = argparse.ArgumentParser(description="Train visual compressor (SmolVLM2 vision -> text LM)")

    # Models
    parser.add_argument("--text_model_path", type=str, default="exp/transformer_160M",
                        help="Path to pre-trained FLA text model checkpoint directory")
    parser.add_argument("--vision_model", type=str, default="HuggingFaceTB/SmolVLM2-500M-Instruct",
                        help="SmolVLM2-500M model id or local path for the vision encoder")
    parser.add_argument("--layer_idx", type=int, default=6,
                        help="Layer index in text model for compressor and memory injection")
    parser.add_argument("--compress_num_tokens", type=int, default=32,
                        help="Number of tokens to compress to")
    parser.add_argument("--compress_depth", type=int, default=0,
                        help="Number of refinement iterations (0=single pass, >0 applies [attention+MLP]×depth)")

    # Data
    parser.add_argument("--annotations_json", type=str, required=True,
                        help="Path to CLEVR-style annotations JSON with image paths and summaries")
    parser.add_argument("--images_root", type=str, default=None,
                        help="Optional root to resolve relative image paths")
    parser.add_argument("--auto_prepare_clevr", action="store_true",
                        help="If set and annotations_json is missing, download a small CLEVR subset and create it")
    parser.add_argument("--auto_out_dir", type=str, default="data/clevr-mini",
                        help="Output directory for auto-prepared CLEVR subset")
    parser.add_argument("--auto_split", type=str, default="val", choices=["train", "val", "test"],
                        help="Split to download for auto-preparation")
    parser.add_argument("--auto_num_images", type=int, default=200,
                        help="Number of images to fetch in auto-preparation")
    parser.add_argument("--auto_synthetic_on_fail", action="store_true",
                        help="If downloads fail, generate synthetic CLEVR-like images instead")
    parser.add_argument("--auto_synthetic_only", action="store_true",
                        help="Skip downloads and generate synthetic images only for auto-preparation")
    parser.add_argument("--max_summary_len", type=int, default=64,
                        help="Max summary length in tokens")
    parser.add_argument("--max_summaries_per_image", type=int, default=4,
                        help="Max number of summaries per image to use")
    parser.add_argument("--num_items", type=int, default=None,
                        help="Limit number of dataset items (None = all)")

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--compressor_lr", type=float, default=5e-4)
    parser.add_argument("--upper_layers_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--freeze_bottom", action="store_true")
    parser.add_argument("--train_only_compressor", action="store_true",
                        help="If set, freezes all params except the compression layer. Optimizer updates compressor only.")

    # Debug
    parser.add_argument("--debug_memory_dir", type=str, default=None,
                        help="Directory to dump memory token diagnostics (disabled if omitted)")
    parser.add_argument("--debug_memory_every", type=int, default=0,
                        help="Dump diagnostics every N steps (0 disables dumping)")
    parser.add_argument("--debug_memory_limit", type=int, default=5,
                        help="Maximum number of diagnostics dumps (<=0 for unlimited)")
    parser.add_argument("--debug_memory_top_k", type=int, default=5,
                        help="How many top vocab tokens to record per memory vector")

    # Checkpoint / system
    parser.add_argument("--save_dir", type=str, default="exp/visual_compressor_training")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--check_grads_every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--load_text_checkpoint", type=str, default=None,
                        help="Optional .pt checkpoint (from prior visual compressor run) to initialize weights before training.")

    args = parser.parse_args()

    print("=" * 80)
    print("Visual Compressor Training Configuration")
    print("=" * 80)
    print(f"Text model: {args.text_model_path}")
    print(f"Vision model: {args.vision_model}")
    print(f"Compression layer: {args.layer_idx}")
    print(f"Compression tokens: {args.compress_num_tokens}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Tokenizer for the text LM
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load text model config and weights
    print("\n2. Loading text model...")
    config_path = os.path.join(args.text_model_path, "model_config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    text_config = TransformerConfig(**config_dict)
    text_model = TransformerForCausalLM(text_config)

    checkpoint_dir = os.path.join(args.text_model_path, "checkpoint")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("step-")]
    if checkpoint_files:
        latest_step = max([int(f.split("-")[1]) for f in checkpoint_files])
        checkpoint_path = os.path.join(checkpoint_dir, f"step-{latest_step}")
        print(f"Loading checkpoint from {checkpoint_path} (step {latest_step})")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, "checkpoint.pt")
            print("  Converting distributed checkpoint...")
            dcp_to_torch_save(checkpoint_path, temp_path)
            torch.serialization.add_safe_globals([timedelta, BytesIO])
            checkpoint = torch.load(temp_path, map_location="cpu", weights_only=False)
            model_state_dict = checkpoint.get("model", checkpoint)
            text_model.load_state_dict(model_state_dict, strict=True)
            print("  ✓ Loaded pre-trained text weights")
    else:
        print("  Warning: No text checkpoint found, using random initialization")

    # 3. Load vision model (SmolVLM2-500M) and image processor
    print("\n3. Loading vision model...")
    vision_hooks, image_processor = build_vision_encoder(
        args.vision_model,
        device=torch.device(args.device),
        dtype=torch.bfloat16,
        target_hidden_size=text_config.hidden_size,
    )
    print("  ✓ Vision model loaded and frozen")

    # 4. Add compressor to text layer L
    print("\n4. Adding compressor to text model...")
    add_compressor_to_layer(text_model, args.layer_idx, compress_deterministic=True, compress_num_tokens=args.compress_num_tokens)

    # 4a. Optionally load a previous visual-compressor checkpoint (includes compressor/upper layers/vision_proj)
    if args.load_text_checkpoint:
        ckpt_path = Path(args.load_text_checkpoint)
        if ckpt_path.exists():
            print(f"Loading text+compressor checkpoint: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            state = ckpt.get('text_model_state_dict') or ckpt.get('model_state_dict') or ckpt
            missing, unexpected = text_model.load_state_dict(state, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)} (show up to 10): {missing[:10]}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)} (show up to 10): {unexpected[:10]}")
        else:
            print(f"  Warning: --load_text_checkpoint not found: {ckpt_path}")

    # Move text model to device + bfloat16
    text_model = text_model.to(dtype=torch.bfloat16, device=args.device)
    print(f"  ✓ Text model (with compressor) converted to bfloat16 on {args.device}")

    # 5. Setup optimizer (include optional vision->text projector if created on-the-fly)
    print("\n5. Setting up optimizer...")
    extra_trainable = []
    optimizer = setup_optimizer(
        text_model,
        args.layer_idx,
        compressor_lr=args.compressor_lr,
        upper_layers_lr=args.upper_layers_lr,
        weight_decay=args.weight_decay,
        freeze_bottom=args.freeze_bottom,
        extra_trainable=None if args.train_only_compressor else extra_trainable,
        only_compressor=args.train_only_compressor,
    )

    # 6. Prepare dataset (optionally auto-download a subset)
    ann_path = Path(args.annotations_json)
    if not ann_path.exists():
        if args.auto_prepare_clevr:
            print("\n6a. Annotations not found; auto-preparing CLEVR subset...")
            # Attempt to run the local downloader
            import subprocess, sys as _sys

            # Prefer local CLEVR zip if present to avoid network
            candidate_zips = [
                Path("data/clevr/CLEVR_v1.0.zip"),
                Path("CLEVR_v1.0.zip"),
                Path(args.auto_out_dir) / "CLEVR_v1.0.zip",
            ]
            local_zip = next((p for p in candidate_zips if p.exists()), None)

            cmd = [
                _sys.executable, "utils/prepare_clevr.py",
                "--out_dir", args.auto_out_dir,
                "--split", args.auto_split,
                "--num", str(args.auto_num_images),
            ]
            if local_zip is not None:
                cmd += ["--zip_path", str(local_zip)]
            if args.auto_synthetic_only:
                cmd.append("--synthetic_only")
            elif args.auto_synthetic_on_fail:
                cmd.append("--synthetic_on_fail")

            print("Running:", " ".join(cmd))
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                # If a local zip exists but extraction failed for some reason, try synthetic-only as a fallback
                if not args.auto_synthetic_only:
                    fallback_cmd = cmd + ["--synthetic_only"] if "--synthetic_only" not in cmd else cmd
                    print("Auto-preparation failed; retrying with synthetic-only subset...")
                    try:
                        subprocess.check_call(fallback_cmd)
                    except subprocess.CalledProcessError:
                        raise RuntimeError(
                            "Auto-preparation failed even with synthetic-only fallback. "
                            "Ensure a local CLEVR_v1.0.zip is available or enable --auto_synthetic_only."
                        )
                else:
                    raise RuntimeError(
                        f"Auto-preparation failed (exit {e.returncode}). Ensure network access is available or provide a local CLEVR_v1.0.zip."
                    )

            # Point to newly created annotations and images root
            args.annotations_json = str(Path(args.auto_out_dir) / "clevr_summaries.json")
            if args.images_root is None:
                args.images_root = args.auto_out_dir
            print(f"  ✓ Auto-prepared annotations at {args.annotations_json}")
        else:
            raise FileNotFoundError(
                f"Annotations file not found: {args.annotations_json}. "
                f"Pass --auto_prepare_clevr to auto-create a small subset, or use --annotations_json data/clevr_test/clevr_summaries.json --images_root data/clevr_test."
            )

    # 6b. Load dataset
    print("\n6. Loading dataset...")
    dataset = ClevrSummariesDataset(
        annotations_json=str(args.annotations_json),
        images_root=args.images_root,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_summary_len=args.max_summary_len,
        max_summaries_per_image=args.max_summaries_per_image,
        num_items=args.num_items,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_images_fn,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    memory_debugger = None
    if args.debug_memory_dir and args.debug_memory_every > 0:
        max_dumps = args.debug_memory_limit if args.debug_memory_limit > 0 else None
        top_k = max(1, args.debug_memory_top_k)
        memory_debugger = MemoryDebugger(
            save_dir=args.debug_memory_dir,
            tokenizer=tokenizer,
            top_k=top_k,
            every=args.debug_memory_every,
            max_dumps=max_dumps,
        )
        print(f"  ✓ Memory debugger enabled (dir={args.debug_memory_dir}, every={args.debug_memory_every} steps)")

    print("\n7. Starting training...")
    print("=" * 80)

    global_step = 0
    compressor = text_model.model.layers[args.layer_idx].compressor
    extra_proj_holder: Dict[str, nn.Module] = {}

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        epoch_loss = 0.0
        epoch_summaries = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            current_step = global_step + 1
            loss, num_summaries = train_step(
                text_model=text_model,
                batch=batch,
                optimizer=optimizer,
                layer_idx=args.layer_idx,
                device=torch.device(args.device),
                vision_hooks=vision_hooks,
                compression_tokens=args.compress_num_tokens,
                compression_depth=args.compress_depth,
                gradient_clip=args.gradient_clip,
                memory_debugger=memory_debugger,
                global_step=current_step,
                extra_proj_holder=extra_proj_holder,
                freeze_new_projection=args.train_only_compressor,
            )

            # If we created a new projection in the first step, add to optimizer param group once
            if (not args.train_only_compressor) and 'vision_proj' in extra_proj_holder and not any(
                (isinstance(g, dict) and g.get('name') == 'vision_proj') for g in optimizer.param_groups
            ):
                optimizer.add_param_group({
                    'params': extra_proj_holder['vision_proj'].parameters(),
                    'lr': args.compressor_lr,
                    'name': 'vision_proj',
                })

            epoch_loss += loss * num_summaries
            epoch_summaries += num_summaries
            global_step = current_step

            if num_summaries > 0:
                progress_bar.set_postfix({'loss': f'{loss:.4f}', 'step': global_step})

            if global_step % args.check_grads_every == 0:
                grad_info = check_compressor_gradients(compressor, verbose=(global_step == args.check_grads_every))
                if not grad_info['has_grads']:
                    raise RuntimeError(
                        f"Step {global_step}: No gradients reached compressor! "
                        "Ensure memory is computed inside the step and not detached."
                    )

            if global_step % args.save_every == 0:
                checkpoint_path = os.path.join(args.save_dir, f"checkpoint_step_{global_step}.pt")
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'text_model_state_dict': text_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'text_config': config_dict,
                    'args': vars(args),
                }, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")

        avg_epoch_loss = epoch_loss / epoch_summaries if epoch_summaries > 0 else 0.0
        print(f"Epoch {epoch + 1} complete: avg_loss={avg_epoch_loss:.4f}, summaries={epoch_summaries}")

    final_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save({
        'step': global_step,
        'text_model_state_dict': text_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'text_config': config_dict,
        'args': vars(args),
    }, final_path)
    print(f"\n✓ Training complete! Final model saved to {final_path}")

    print("\nFinal gradient verification:")
    grad_info = check_compressor_gradients(compressor, verbose=True)


if __name__ == "__main__":
    main()
