#!/usr/bin/env python3
"""
Train compressor for adaptive attention on CNN/DailyMail article-highlight pairs.

Setup:
1. Load pre-trained 160M transformer from exp/transformer_160M
2. Add compressor at layer L (trained from scratch)
3. For each article: compress once, cache M (keep in computation graph)
4. For each highlight: inject M at layer L, train on highlight tokens only
5. Backprop accumulates gradients from all highlights into compressor
"""

import argparse
import json
import os
import tempfile
from datetime import timedelta
from io import BytesIO
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from fla.layers.adaptive import AdaptiveAttention, check_compressor_gradients
from fla.models.transformer import TransformerForCausalLM, TransformerConfig
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


class CNNDailyMailHighlightsDataset(Dataset):
    """
    Dataset that loads article-highlight pairs from extracted_highlights/*.json files.
    Groups highlights by article_id for efficient batching.
    """
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_article_len: int = 2048,
        max_highlight_len: int = 128,
        max_highlights_per_article: int = None,
        num_files: int = None,
    ):
        self.tokenizer = tokenizer
        self.max_article_len = max_article_len
        self.max_highlight_len = max_highlight_len
        self.max_highlights_per_article = max_highlights_per_article
        
        # Load data files
        self.data = []
        data_files = sorted(Path(data_dir).glob("highlights_*.json"))
        
        if num_files is not None:
            data_files = data_files[:num_files]
        
        print(f"Loading data from {len(data_files)} files...")
        for data_file in tqdm(data_files, desc="Loading data"):
            with open(data_file, 'r') as f:
                content = json.load(f)
                self.data.extend(content['results'])
        
        print(f"Loaded {len(self.data)} articles with highlights")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        article = item['article']
        summaries = item['summaries']
        
        # Limit number of highlights per article
        if self.max_highlights_per_article:
            summaries = summaries[:self.max_highlights_per_article]
        
        # Tokenize article
        art_enc = self.tokenizer(
            article,
            max_length=self.max_article_len,
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize each highlight
        highlight_encodings = []
        for summary_data in summaries:
            summary_text = summary_data['text']
            hl_enc = self.tokenizer(
                summary_text,
                max_length=self.max_highlight_len,
                truncation=True,
                return_tensors='pt'
            )
            highlight_encodings.append({
                'input_ids': hl_enc['input_ids'].squeeze(0),
                'attention_mask': hl_enc['attention_mask'].squeeze(0)
            })
        
        return {
            'article_id': item['article_id'],
            'article_input_ids': art_enc['input_ids'].squeeze(0),
            'article_attention_mask': art_enc['attention_mask'].squeeze(0),
            'highlights': highlight_encodings
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
        article_ids: List[str],
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

            # Persist raw tensors for offline analysis
            payload = {
                'step': step,
                'article_ids': article_ids,
                'memory_states': mem.to(dtype=torch.float32).cpu(),
                'memory_mask': mask.to(dtype=torch.int64).cpu(),
            }
            torch.save(payload, step_dir / "memory_states.pt")

            # Project memories into vocab space
            logits = model.lm_head(mem).float()
            k = min(self.top_k, logits.size(-1))
            top_values, top_indices = torch.topk(logits, k=k, dim=-1)

            decoded_records = []
            text_lines = []
            for idx, article_id in enumerate(article_ids):
                valid = int(mask[idx].sum().item())
                if valid == 0:
                    continue

                argmax_token_ids = top_indices[idx, :valid, 0].tolist()
                argmax_text = self.tokenizer.decode(argmax_token_ids)

                per_position = []
                for pos in range(valid):
                    top_tokens = []
                    for rank in range(k):
                        token_id = int(top_indices[idx, pos, rank].item())
                        token_text = self.tokenizer.decode([token_id]).replace("\n", "\\n")
                        logit_val = float(top_values[idx, pos, rank].item())
                        top_tokens.append({
                            'token_id': token_id,
                            'token': token_text,
                            'logit': logit_val,
                        })
                    per_position.append({
                        'position': pos,
                        'top_tokens': top_tokens,
                    })

                decoded_records.append({
                    'article_id': article_id,
                    'valid_memory_tokens': valid,
                    'argmax_token_ids': argmax_token_ids,
                    'argmax_text': argmax_text,
                    'positions': per_position,
                })

                text_lines.append(f"article_id: {article_id}")
                text_lines.append(f"valid_memory_tokens: {valid}")
                text_lines.append(f"argmax_text: {argmax_text}")
                text_lines.append("")

            if decoded_records:
                with open(step_dir / "decoded.json", 'w') as f:
                    json.dump(decoded_records, f, indent=2)
                (step_dir / "decoded.txt").write_text("\n".join(text_lines))

def collate_fn(batch):
    """Custom collate that keeps highlights as lists and pads articles."""
    # Batch articles with padding
    article_ids = [item['article_id'] for item in batch]
    
    # Find max length for articles in this batch
    max_art_len = max(item['article_input_ids'].size(0) for item in batch)
    
    # Pad articles to max length
    article_input_ids = []
    article_attention_mask = []
    
    for item in batch:
        art_ids = item['article_input_ids']
        art_mask = item['article_attention_mask']
        
        pad_len = max_art_len - art_ids.size(0)
        if pad_len > 0:
            art_ids = torch.cat([art_ids, torch.zeros(pad_len, dtype=art_ids.dtype)])
            art_mask = torch.cat([art_mask, torch.zeros(pad_len, dtype=art_mask.dtype)])
        
        article_input_ids.append(art_ids)
        article_attention_mask.append(art_mask)
    
    article_input_ids = torch.stack(article_input_ids)
    article_attention_mask = torch.stack(article_attention_mask)
    
    # Keep highlights as nested list (not batched yet)
    highlights = [item['highlights'] for item in batch]
    
    return {
        'article_ids': article_ids,
        'article_input_ids': article_input_ids,
        'article_attention_mask': article_attention_mask,
        'highlights': highlights
    }


def run_to_layer(model, input_ids, attention_mask, end_layer_exclusive):
    """
    Run bottom stack [0..end_layer_exclusive) to get hidden states at layer L input.
    
    Returns:
        hidden_states: (B, T, D) hidden states at the input to layer end_layer_exclusive
    """
    x = model.model.embeddings(input_ids)
    
    for i in range(end_layer_exclusive):
        layer = model.model.layers[i]
        # Forward returns (hidden_states, attentions, past_key_values)
        x = layer(x, attention_mask=attention_mask)[0]
    
    return x


def add_compressor_to_layer(
    model: nn.Module,
    layer_idx: int,
    compress_deterministic: bool = True,
    compress_num_tokens: int = 32,
) -> None:
    """
    Add an AdaptiveAttention compressor to the specified layer.
    The compressor is trained from scratch.
    """
    layer = model.model.layers[layer_idx]
    config = model.config
    
    # Create compressor with same dims as the attention layer
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
    
    # Attach to layer
    layer.compressor = compressor
    print(f"Added compressor to layer {layer_idx} (deterministic={compress_deterministic}, num_tokens={compress_num_tokens})")


def setup_optimizer(
    model: nn.Module,
    layer_idx: int,
    compressor_lr: float = 5e-4,
    upper_layers_lr: float = 1e-4,
    weight_decay: float = 0.1,
    freeze_bottom: bool = True,
) -> torch.optim.Optimizer:
    """
    Setup optimizer with parameter groups:
    - Compressor (highest LR, trained from scratch)
    - Upper layers [L..end] (lower LR)
    - LM head (lower LR)
    - Optionally freeze bottom layers [0..L)
    """
    param_groups = []
    
    # Freeze bottom layers if requested
    if freeze_bottom:
        for i in range(layer_idx):
            for param in model.model.layers[i].parameters():
                param.requires_grad = False
        print(f"Froze layers 0..{layer_idx-1}")
    
    # Compressor parameters (highest LR)
    compressor = model.model.layers[layer_idx].compressor
    param_groups.append({
        "params": compressor.parameters(),
        "lr": compressor_lr,
        "name": "compressor"
    })
    
    # Upper layers (layer_idx and above)
    upper_params = []
    for i in range(layer_idx, len(model.model.layers)):
        for name, param in model.model.layers[i].named_parameters():
            if 'compressor' not in name and param.requires_grad:
                upper_params.append(param)
    
    if upper_params:
        param_groups.append({
            "params": upper_params,
            "lr": upper_layers_lr,
            "name": "upper_layers"
        })
    
    # LM head
    param_groups.append({
        "params": model.lm_head.parameters(),
        "lr": upper_layers_lr,
        "name": "lm_head"
    })
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    print(f"\nOptimizer setup:")
    print(f"  Compressor LR: {compressor_lr}")
    print(f"  Upper layers LR: {upper_layers_lr}")
    print(f"  Weight decay: {weight_decay}")
    
    return optimizer


def train_step(
    model: nn.Module,
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    layer_idx: int,
    device: torch.device,
    compression_tokens: Optional[int] = None,
    compression_depth: int = 3,
    gradient_clip: float = 1.0,
    memory_debugger: Optional[MemoryDebugger] = None,
    global_step: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Single training step:
    1. Compress each article once (keep in computation graph)
    2. For each highlight in the article: inject memory, forward, compute loss
    3. Accumulate loss across all highlights
    4. Single backward and optimizer step
    
    Returns:
        (total_loss, num_highlights)
    """
    model.train()
    
    article_ids = batch['article_ids']
    art_ids = batch['article_input_ids'].to(device)
    art_mask = batch['article_attention_mask'].to(device)
    highlights = batch['highlights']  # List[List[Dict]]
    
    B = art_ids.size(0)
    compressor = model.model.layers[layer_idx].compressor
    
    # Step 1: Compress each article once (CRITICAL: keep in computation graph)
    with torch.enable_grad():
        H_art = run_to_layer(model, art_ids, art_mask, layer_idx)  # (B, S, D)
        
        # Compress all articles in batch with optional depth refinement
        M_batch, _, meta_batch = compressor.forward_compress(
            H_art,
            attention_mask=art_mask,
            num_tokens=compression_tokens,
            depth=compression_depth,
            return_padded=True
        )  # M_batch: (B, m, D), meta has mask_padded (B, m)
        M_mask_batch = meta_batch["mask_padded"]  # (B, m)

    if memory_debugger is not None and global_step is not None:
        memory_debugger.maybe_log(
            step=global_step,
            article_ids=article_ids,
            memory_states=M_batch,
            memory_mask=M_mask_batch,
            model=model,
        )

    # Step 2: For each article, process all its highlights
    total_loss = 0.0
    total_highlights = 0
    
    for b in range(B):
        M = M_batch[b:b+1]  # (1, m, D) - keep batch dim
        M_mask = M_mask_batch[b:b+1]  # (1, m)
        article_highlights = highlights[b]
        
        # Process each highlight for this article
        for hl_data in article_highlights:
            hl_ids = hl_data['input_ids'].unsqueeze(0).to(device)  # (1, T)
            hl_mask = hl_data['attention_mask'].unsqueeze(0).to(device)  # (1, T)
            
            # Shift for causal LM: input = tokens[:-1], labels = tokens[1:]
            input_ids = hl_ids[:, :-1]
            labels = hl_ids[:, 1:]
            input_mask = hl_mask[:, :-1]
            
            if input_ids.size(1) == 0:
                continue
            
            # Manual forward with memory injection at layer L
            # Run bottom layers [0..L)
            H_hl = run_to_layer(model, input_ids, input_mask, layer_idx)  # (1, T_hl, D)
            
            # Inject memory: concatenate M with H_hl
            # CRITICAL: Don't detach M so gradients flow back
            H_with_mem = torch.cat([M, H_hl], dim=1)  # (1, m + T_hl, D)
            
            # Concatenate masks
            mask_with_mem = torch.cat([M_mask, input_mask], dim=1)  # (1, m + T_hl)
            
            # Run upper layers [L..end]
            hidden = H_with_mem
            for i in range(layer_idx, len(model.model.layers)):
                layer = model.model.layers[i]
                hidden = layer(hidden, attention_mask=mask_with_mem)[0]
            
            # Final norm and LM head
            hidden = model.model.norm(hidden)
            
            # Only compute logits for the highlight tokens (skip memory tokens)
            hidden_hl = hidden[:, M.size(1):, :]  # Skip memory tokens, keep highlight part
            logits = model.lm_head(hidden_hl)
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss
            total_highlights += 1
    
    if total_highlights == 0:
        return 0.0, 0
    
    # Step 3: Single backward pass (accumulates gradients from all highlights)
    avg_loss = total_loss / total_highlights
    
    optimizer.zero_grad(set_to_none=True)
    avg_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    
    optimizer.step()
    
    return avg_loss.item(), total_highlights


def main():
    parser = argparse.ArgumentParser(description="Train compressor on CNN/DailyMail")
    
    # Model args
    parser.add_argument("--model_path", type=str, default="exp/transformer_160M",
                        help="Path to pre-trained model checkpoint")
    parser.add_argument("--layer_idx", type=int, default=6,
                        help="Layer index where compressor is inserted (middle layer recommended)")
    parser.add_argument("--compress_num_tokens", type=int, default=32,
                        help="Number of tokens to compress to")
    parser.add_argument("--compress_depth", type=int, default=0,
                        help="Number of refinement iterations (0=single pass, >0 applies [attention+MLP]×depth)")
    
    # Data args
    parser.add_argument("--data_dir", type=str, default="extracted_highlights",
                        help="Directory with highlights_*.json files")
    parser.add_argument("--max_article_len", type=int, default=2048,
                        help="Max article length in tokens")
    parser.add_argument("--max_highlight_len", type=int, default=128,
                        help="Max highlight length in tokens")
    parser.add_argument("--max_highlights_per_article", type=int, default=4,
                        help="Max number of highlights to use per article")
    parser.add_argument("--num_files", type=int, default=None,
                        help="Number of data files to load (None = all)")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Number of articles per batch")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--compressor_lr", type=float, default=5e-4,
                        help="Learning rate for compressor")
    parser.add_argument("--upper_layers_lr", type=float, default=1e-4,
                        help="Learning rate for upper layers and lm_head")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="Max gradient norm")
    parser.add_argument("--freeze_bottom", action="store_true",
                        help="Freeze bottom layers [0..L)")

    # Debug args
    parser.add_argument("--debug_memory_dir", type=str, default=None,
                        help="Directory to dump memory token diagnostics (disabled if omitted)")
    parser.add_argument("--debug_memory_every", type=int, default=0,
                        help="Dump diagnostics every N steps (0 disables dumping)")
    parser.add_argument("--debug_memory_limit", type=int, default=5,
                        help="Maximum number of diagnostics dumps (<=0 for unlimited)")
    parser.add_argument("--debug_memory_top_k", type=int, default=5,
                        help="How many top vocab tokens to record per memory vector")
    
    # Checkpoint args
    parser.add_argument("--save_dir", type=str, default="exp/compressor_training",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--check_grads_every", type=int, default=100,
                        help="Check gradients every N steps")
    
    # System args
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Compressor Training Configuration")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Compression layer: {args.layer_idx}")
    print(f"Compression tokens: {args.compress_num_tokens}")
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model config
    print("\n2. Loading model...")
    config_path = os.path.join(args.model_path, "model_config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = TransformerConfig(**config_dict)
    model = TransformerForCausalLM(config)
    
    # Load checkpoint from distributed format
    checkpoint_dir = os.path.join(args.model_path, "checkpoint")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("step-")]
    if checkpoint_files:
        # Load latest checkpoint
        latest_step = max([int(f.split("-")[1]) for f in checkpoint_files])
        checkpoint_path = os.path.join(checkpoint_dir, f"step-{latest_step}")
        print(f"Loading checkpoint from {checkpoint_path} (step {latest_step})")
        
        # Convert distributed checkpoint to regular format
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, "checkpoint.pt")
            print("  Converting distributed checkpoint...")
            dcp_to_torch_save(checkpoint_path, temp_path)
            
            # Load checkpoint
            torch.serialization.add_safe_globals([timedelta, BytesIO])
            checkpoint = torch.load(temp_path, map_location="cpu", weights_only=False)
            
            # Load model weights
            model_state_dict = checkpoint.get("model", checkpoint)
            model.load_state_dict(model_state_dict, strict=True)
            print(f"  ✓ Loaded pre-trained weights from step {latest_step}")
    else:
        print("  Warning: No checkpoint found, using random initialization")
    
    # Add compressor to layer L (before moving to device)
    print("\n3. Adding compressor...")
    add_compressor_to_layer(
        model,
        args.layer_idx,
        compress_deterministic=True,
        compress_num_tokens=args.compress_num_tokens
    )
    
    # Convert to bfloat16 and move to device (after adding compressor)
    model = model.to(dtype=torch.bfloat16, device=args.device)
    print(f"  ✓ Model (with compressor) converted to bfloat16 on {args.device}")
    
    # Setup optimizer
    print("\n4. Setting up optimizer...")
    optimizer = setup_optimizer(
        model,
        args.layer_idx,
        compressor_lr=args.compressor_lr,
        upper_layers_lr=args.upper_layers_lr,
        weight_decay=args.weight_decay,
        freeze_bottom=args.freeze_bottom
    )
    
    # Load dataset
    print("\n5. Loading dataset...")
    dataset = CNNDailyMailHighlightsDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_article_len=args.max_article_len,
        max_highlight_len=args.max_highlight_len,
        max_highlights_per_article=args.max_highlights_per_article,
        num_files=args.num_files
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # Create save directory
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

    # Training loop
    print("\n6. Starting training...")
    print("=" * 80)
    
    global_step = 0
    compressor = model.model.layers[args.layer_idx].compressor
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        epoch_loss = 0.0
        epoch_highlights = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
            current_step = global_step + 1
            loss, num_highlights = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                layer_idx=args.layer_idx,
                device=args.device,
                compression_tokens=args.compress_num_tokens,
                compression_depth=args.compress_depth,
                gradient_clip=args.gradient_clip,
                memory_debugger=memory_debugger,
                global_step=current_step,
            )
            
            epoch_loss += loss * num_highlights
            epoch_highlights += num_highlights
            global_step = current_step
            
            # Update progress
            if num_highlights > 0:
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'step': global_step
                })
            
            # Check gradients periodically
            if global_step % args.check_grads_every == 0:
                grad_info = check_compressor_gradients(compressor, verbose=(global_step == args.check_grads_every))
                if not grad_info['has_grads']:
                    raise RuntimeError(
                        f"Step {global_step}: No gradients reached compressor! "
                        "Check that memory_bank uses freeze_memory=False"
                    )
            
            # Save checkpoint
            if global_step % args.save_every == 0:
                checkpoint_path = os.path.join(args.save_dir, f"checkpoint_step_{global_step}.pt")
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config_dict,
                    'args': vars(args)
                }, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / epoch_highlights if epoch_highlights > 0 else 0.0
        print(f"Epoch {epoch + 1} complete: avg_loss={avg_epoch_loss:.4f}, highlights={epoch_highlights}")
    
    # Final save
    final_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config_dict,
        'args': vars(args)
    }, final_path)
    
    print(f"\n✓ Training complete! Final model saved to {final_path}")
    
    # Final gradient check
    print("\nFinal gradient verification:")
    grad_info = check_compressor_gradients(compressor, verbose=True)


if __name__ == "__main__":
    main()
