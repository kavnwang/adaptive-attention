#!/usr/bin/env python3
"""
Evaluate the visual compressor checkpoint by reproducing the training pipeline
without gradient updates and reporting the average loss (and perplexity) over
a subset of the data.

Pipeline mirrors train_visual_compressor.py:
  1) Load pre-trained text LM (TransformerForCausalLM) and attach AdaptiveAttention compressor
  2) Load vision encoder (SmolVLM2-500M) + image processor
  3) For each image: run vision encoder, project to text hidden size if needed, compress via compressor
  4) For each summary: inject memory at layer L, run upper layers, compute token-level CE loss

Notes
- If a learned vision->text projection (vision_proj) was saved under the checkpoint, it will be loaded
  and used. If not found and dims mismatch, a fresh projection is created on-the-fly for evaluation.
- No gradients are computed; model runs in eval() mode with torch.no_grad().
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from fla.layers.adaptive import AdaptiveAttention
from fla.models.transformer import TransformerForCausalLM, TransformerConfig


class ClevrSummariesDataset(Dataset):
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
        if num_items is not None and num_items > 0:
            data = data[:num_items]

        self.items = []
        images_root = Path(images_root) if images_root else None
        for entry in data:
            img_path = Path(entry['image'])
            if images_root is not None and not img_path.is_absolute():
                img_path = images_root / img_path
            summaries = entry.get('summaries', [])
            if self.max_summaries_per_image is not None and self.max_summaries_per_image > 0:
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
        pixel_values = pixel_inputs['pixel_values'].squeeze(0)  # (C, H, W) or (N, C, H, W)

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
            'summaries_text': item['summaries'],
        }


def collate_images_fn(batch):
    image_paths = [it['image_path'] for it in batch]
    pixel_values = torch.stack([it['pixel_values'] for it in batch], dim=0)
    summaries = [it['summaries'] for it in batch]
    summaries_text = [it['summaries_text'] for it in batch]
    return {
        'image_paths': image_paths,
        'pixel_values': pixel_values,
        'summaries': summaries,
        'summaries_text': summaries_text,
    }


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
            raise RuntimeError("Could not locate a vision encoder on the SmolVLM2 model. Please adapt VisionHooks.forward_features.")

        if original_dims == 5:
            Bp, S, Dv = feats.shape
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
    image_processor = AutoImageProcessor.from_pretrained(name_or_path)
    hooks = VisionHooks(model=vlm, projector=None, hidden_size=target_hidden_size)
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


def greedy_decode_with_memory(
    text_model: nn.Module,
    tokenizer,
    M: torch.Tensor,
    M_mask: torch.Tensor,
    layer_idx: int,
    device: torch.device,
    max_new_tokens: int = 64,
    stop_on_eos: bool = True,
):
    text_model.eval()

    bos_id = tokenizer.bos_token_id if getattr(tokenizer, 'bos_token_id', None) is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        input_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        H_text = run_to_layer(text_model, input_ids, input_mask, layer_idx)
        hidden = torch.cat([M, H_text], dim=1)
        mask_with_mem = torch.cat([M_mask, input_mask], dim=1)

        for i in range(layer_idx, len(text_model.model.layers)):
            layer = text_model.model.layers[i]
            hidden = layer(hidden, attention_mask=mask_with_mem)[0]

        hidden = text_model.model.norm(hidden)
        hidden_hl = hidden[:, M.size(1):, :]
        next_logits = text_model.lm_head(hidden_hl[:, -1:, :])  # (1,1,V)
        next_id = int(torch.argmax(next_logits[0, -1], dim=-1).item())
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
        if stop_on_eos and next_id == eos_id:
            break

    # Drop the initial BOS when decoding
    gen_ids = input_ids[0, 1:].tolist()
    gen_text = tokenizer.decode(gen_ids)
    return gen_ids, gen_text


@torch.no_grad()
def eval_step(
    text_model: nn.Module,
    batch: Dict,
    layer_idx: int,
    device: torch.device,
    vision_hooks: VisionHooks,
    compression_tokens: Optional[int] = None,
    compression_depth: int = 0,
    ckpt_state: Optional[Dict] = None,
    tokenizer=None,
    do_comparisons: bool = False,
    max_new_tokens: int = 64,
) -> Tuple[float, int, List[Dict]]:
    text_model.eval()

    image_paths = batch['image_paths']
    pixel_values = batch['pixel_values'].to(device=device, dtype=torch.bfloat16)
    summaries = batch['summaries']
    summaries_text = batch.get('summaries_text', None)

    # 1) Vision features and optional projection
    feats = vision_hooks.forward_features(pixel_values)  # (B, S, Dv)

    # If a trained vision_proj exists in the checkpoint, apply it; else project if needed
    if ckpt_state:
        vp_w_key = None
        for key in ckpt_state.keys():
            if key.endswith('vision_proj.weight'):
                vp_w_key = key
                break
        if vp_w_key is not None:
            W = ckpt_state[vp_w_key]
            b = ckpt_state.get(vp_w_key.replace('weight', 'bias'), None)
            out_features, in_features = W.shape[0], W.shape[1]
            if feats.size(-1) == in_features:
                vision_proj = nn.Linear(in_features, out_features, bias=b is not None).to(device=device, dtype=feats.dtype)
                vision_proj.weight.copy_(W.to(dtype=feats.dtype))
                if b is not None:
                    vision_proj.bias.copy_(b.to(dtype=feats.dtype))
                feats = vision_proj(feats)
            else:
                feats, _ = ensure_projection(feats, target_hidden=text_model.config.hidden_size)
        else:
            feats, _ = ensure_projection(feats, target_hidden=text_model.config.hidden_size)
    else:
        feats, _ = ensure_projection(feats, target_hidden=text_model.config.hidden_size)

    B = feats.size(0)
    loss_fct = nn.CrossEntropyLoss()

    # Create full-ones attention mask for vision features
    vmask = torch.ones(feats.size()[:2], device=device, dtype=torch.long)

    # Compress features with the compressor at layer L
    compressor = text_model.model.layers[layer_idx].compressor
    M_batch, _, meta_batch = compressor.forward_compress(
        feats,
        attention_mask=vmask,
        num_tokens=compression_tokens,
        depth=compression_depth,
        return_padded=True,
    )  # (B, m, D)
    M_mask_batch = meta_batch["mask_padded"]  # (B, m)

    # 2) Inject memory and compute loss for each summary
    total_loss = 0.0
    total_summaries = 0
    comparisons: List[Dict] = []

    for b in range(B):
        M = M_batch[b:b+1]
        M_mask = M_mask_batch[b:b+1]
        image_summaries = summaries[b]
        image_summaries_text = summaries_text[b] if summaries_text is not None else [None] * len(image_summaries)

        for sidx, sdata in enumerate(image_summaries):
            hl_ids = sdata['input_ids'].unsqueeze(0).to(device)
            hl_mask = sdata['attention_mask'].unsqueeze(0).to(device)
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
            logits = text_model.lm_head(hidden_hl).float()

            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += float(loss.item())
            total_summaries += 1

            if do_comparisons and tokenizer is not None:
                # Teacher-forced next-token predictions
                pred_tf_ids = torch.argmax(logits, dim=-1).view(-1).tolist()
                pred_tf_text = tokenizer.decode(pred_tf_ids)
                true_text = image_summaries_text[sidx]

                # Free-running greedy decode with memory
                gen_ids, gen_text = greedy_decode_with_memory(
                    text_model=text_model,
                    tokenizer=tokenizer,
                    M=M,
                    M_mask=M_mask,
                    layer_idx=layer_idx,
                    device=device,
                    max_new_tokens=max_new_tokens,
                )

                comparisons.append({
                    'image_path': image_paths[b],
                    'reference': true_text,
                    'teacher_forced_pred': pred_tf_text,
                    'free_decode_pred': gen_text,
                    'loss': float(loss.item()),
                })

    avg = (total_loss / total_summaries if total_summaries > 0 else 0.0)
    return avg, total_summaries, comparisons


def main():
    ap = argparse.ArgumentParser(description="Evaluate visual compressor by reproducing compression pipeline")

    # Models
    ap.add_argument("--text_model_path", type=str, default="exp/transformer_160M",
                    help="Path to pre-trained FLA text model checkpoint directory")
    ap.add_argument("--load_checkpoint", type=str, default="exp/visual_compressor_training/final_model.pt",
                    help="Path to .pt checkpoint with compressor/text weights from training")
    ap.add_argument("--vision_model", type=str, default="HuggingFaceTB/SmolVLM2-500M-Instruct",
                    help="SmolVLM2-500M model id or local path for the vision encoder")
    ap.add_argument("--layer_idx", type=int, default=6,
                    help="Layer index in text model for compressor and memory injection")
    ap.add_argument("--compress_num_tokens", type=int, default=32,
                    help="Number of tokens to compress to")
    ap.add_argument("--compress_depth", type=int, default=0,
                    help="Number of refinement iterations (as in training)")

    # Data
    ap.add_argument("--annotations_json", type=str, required=True,
                    help="Path to CLEVR-style annotations JSON with image paths and summaries")
    ap.add_argument("--images_root", type=str, default=None,
                    help="Optional root to resolve relative image paths")
    ap.add_argument("--max_summary_len", type=int, default=64)
    ap.add_argument("--max_summaries_per_image", type=int, default=None,
                    help="Limit number of summaries per image (None or <=0 for all)")
    ap.add_argument("--num_items", type=int, default=None,
                    help="Limit number of dataset items (None or <=0 for all)")

    # System
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_comparisons", action="store_true",
                    help="If set, saves per-example predictions vs references to a JSONL file")
    ap.add_argument("--comparisons_out", type=str, default="exp/visual_compressor_training/eval_comparisons.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=64)

    args = ap.parse_args()

    device = torch.device(args.device)

    # Tokenizer
    print("1) Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Text model
    print("2) Loading text model + compressor...")
    config_path = os.path.join(args.text_model_path, "model_config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    text_config = TransformerConfig(**config_dict)
    text_model = TransformerForCausalLM(text_config)
    add_compressor_to_layer(text_model, args.layer_idx, compress_deterministic=True, compress_num_tokens=args.compress_num_tokens)

    # Load checkpoint weights
    ckpt_state = {}
    if args.load_checkpoint and Path(args.load_checkpoint).exists():
        ckpt = torch.load(args.load_checkpoint, map_location="cpu")
        ckpt_state = ckpt.get('text_model_state_dict') or ckpt.get('model_state_dict') or ckpt
        missing, unexpected = text_model.load_state_dict(ckpt_state, strict=False)
        print(f"   ✓ Loaded checkpoint: {args.load_checkpoint}")
        if missing:
            print(f"     Missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
        if unexpected:
            print(f"     Unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")
    else:
        print("   ⚠ No checkpoint found; evaluating random-initialized text/compressor")

    text_model = text_model.to(dtype=torch.bfloat16, device=device)
    text_model.eval()

    # Vision model
    print("3) Loading vision encoder + processor...")
    vision_hooks, image_processor = build_vision_encoder(
        args.vision_model,
        device=device,
        dtype=torch.bfloat16,
        target_hidden_size=text_config.hidden_size,
    )
    print("   ✓ Vision model loaded (frozen)")

    # Dataset + dataloader
    print("4) Loading dataset subset...")
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
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_images_fn,
    )

    # Eval loop
    print("5) Running evaluation (no grad)...")
    total_loss = 0.0
    total_summaries = 0
    writer = None
    if args.save_comparisons:
        Path(Path(args.comparisons_out).parent).mkdir(parents=True, exist_ok=True)
        writer = open(args.comparisons_out, 'w', encoding='utf-8')

    for batch in tqdm(dataloader, desc="Eval"):
        loss, n, comps = eval_step(
            text_model=text_model,
            batch=batch,
            layer_idx=args.layer_idx,
            device=device,
            vision_hooks=vision_hooks,
            compression_tokens=args.compress_num_tokens,
            compression_depth=args.compress_depth,
            ckpt_state=ckpt_state,
            tokenizer=tokenizer,
            do_comparisons=bool(writer is not None),
            max_new_tokens=args.max_new_tokens,
        )
        total_loss += loss * n
        total_summaries += n
        if writer and comps:
            import json as _json
            for rec in comps:
                writer.write(_json.dumps(rec, ensure_ascii=False) + "\n")

    avg_loss = total_loss / total_summaries if total_summaries > 0 else 0.0
    ppl = float(torch.exp(torch.tensor(avg_loss))) if total_summaries > 0 else float('inf')
    print("\n=== Eval Results ===")
    print(f"Summaries evaluated: {total_summaries}")
    print(f"Average loss (CE): {avg_loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")
    if writer:
        writer.close()
        print(f"Comparisons written to: {args.comparisons_out}")


if __name__ == "__main__":
    main()
