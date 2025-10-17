"""
Test of AdaptiveTransformer with RECURRENT compression.
- Compression: 2 state tokens + 1 input token at each step
- Final output: 2 compressed tokens that encode the entire sequence
- Process: sequential/recurrent (like an RNN but with attention)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent / "3rdparty" / "bento"))

from fla.models.adaptive import AdaptiveTransformerConfig, AdaptiveTransformerForCausalLM
from modular_arithmetic_dataset import SimpleSumDataset, collate_fn

# WandB setup
try:
    import wandb
    USE_WANDB = True
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "adaptive-attention"),
        name="recurrent-compression-test",
        config={
            "architecture": "AdaptiveTransformer",
            "compression_type": "recurrent",
            "dataset": "simple_sum",
        }
    )
except ImportError:
    USE_WANDB = False
    print("⚠️  wandb not installed. Install with: pip install wandb")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

print("=" * 80)
print("RECURRENT COMPRESSION TEST: AdaptiveTransformer")
print("=" * 80)
print(f"Device: {device}")
print(f"Dtype: {dtype}")
print()
print("Compression Strategy (same structure as standard adaptive attention):")
print("  - At each step:")
print("    1. Input: N cached tokens + 1 new token = N+1 tokens")
print("    2. Append N compress tokens")
print("    3. Attention: Q from compress tokens, K,V from input tokens only")
print("       - Compress tokens attend ONLY to input tokens (not to each other)")
print("    4. MLP on combined sequence (input + compress tokens)")
print("    5. Output N compress tokens become new cache")
print("  - Process all input tokens sequentially for each depth level")
print("  - Final: N compressed tokens encoding entire sequence")
print("=" * 80)

# Create datasets
print("\nCreating datasets...")
train_dataset = SimpleSumDataset(
    num_samples=50000,
    sequence_length=8,
    modulo=23,
    value_range=10,
    seed=12345
)

val_dataset = SimpleSumDataset(
    num_samples=2000,
    sequence_length=8,
    modulo=23,
    value_range=10,
    seed=54321
)

test_dataset = SimpleSumDataset(
    num_samples=1000,
    sequence_length=8,
    modulo=23,
    value_range=10,
    seed=99999
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, collate_fn=collate_fn)

vocab_size = train_dataset.get_vocab_size()
value_offset = train_dataset.VALUE_OFFSET

print(f"Vocab size: {vocab_size}")
print(f"Train samples: {len(train_dataset):,} ({len(train_loader):,} batches)")
print(f"Val samples: {len(val_dataset):,} ({len(val_loader):,} batches)")
print(f"Test samples: {len(test_dataset):,} ({len(test_loader):,} batches)")
print(f"Task: sum of 8 numbers mod 23")

# Model with recurrent compression
print("\nCreating model with RECURRENT compression...")
config = AdaptiveTransformerConfig(
    vocab_size=vocab_size,
    hidden_size=256,
    num_hidden_layers=1,
    num_heads=4,
    num_kv_heads=4,
    intermediate_size=512,
    max_position_embeddings=16,
    compress_layers=(0,),
    compress_target_tokens=1,
    compress_return_padded=True,
    compress_deterministic=True,
    compress_num_tokens=1,
    recurrent_depth=0,
)

# Update wandb config with model details
if USE_WANDB:
    wandb.config.update({
        "vocab_size": vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_heads": config.num_heads,
        "intermediate_size": config.intermediate_size,
        "compress_layers": config.compress_layers,
        "compress_num_tokens": config.compress_num_tokens,
        "recurrent_depth": config.recurrent_depth,
        "sequence_length": 8,
        "modulo": 23,
        "value_range": 10,
        "task": "simple_sum",
        "batch_size": 64,
        "learning_rate": 3e-4,
        "num_epochs": 10,
    })

model = AdaptiveTransformerForCausalLM(config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")
print(f"Compress at layer: {config.compress_layers}")
print(f"Compression type: RECURRENT")
print(f"  - Per step: {config.compress_num_tokens} cached + 1 new = {config.compress_num_tokens + 1} input tokens")
print(f"  - Attention: Q={config.compress_num_tokens} compress → K,V={config.compress_num_tokens + 1} input")
print(f"  - Compress tokens attend only to input (not each other)")
print(f"  - MLP processing on combined sequence")
print(f"  - Output: {config.compress_num_tokens} compress tokens as new cache")
print(f"Final compressed tokens: {config.compress_num_tokens}")
print(f"Task: Simple sum of {8} numbers modulo {23}")
print("Input format: [SEP] x1 x2 ... x8 [SEP]")

# Training setup with stability improvements
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, eps=1e-8)

print("\n" + "=" * 80)
print("Training for 10 epochs with recurrent compression...")
print("=" * 80)

num_epochs = 10
best_val_acc = 0.0

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        
        # Forward with RECURRENT compression
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=dtype):
            seq_len = input_ids.size(1)
            outputs = model(
                input_ids=input_ids,
                compress_layers=(0,),
                compress_recurrent=True,
                compress_prefix_len=seq_len,
                compress_num_tokens=config.compress_num_tokens,  # Number of state tokens
                recurrent_depth=0,
            )
            logits = outputs.logits[:, -1, :]  # Last position
            target_token_ids = targets + value_offset
            loss = nn.functional.cross_entropy(logits, target_token_ids)
        
        # Check for NaN BEFORE backward to prevent corrupting gradients.
        if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"  WARNING: NaN/Inf detected at batch {batch_idx+1}, skipping backward pass...")
            continue
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients for NaN before updating weights.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"  WARNING: NaN/Inf gradients at batch {batch_idx+1}, skipping optimizer step...")
            optimizer.zero_grad()
            continue
        
        optimizer.step()
        
        # Track metrics
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct = (predictions == target_token_ids).sum().item()
            total_correct += correct
            total_samples += len(targets)
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 200 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            train_acc_running = 100.0 * total_correct / total_samples
            print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}: "
                  f"Loss={avg_loss:.4f}, Acc={train_acc_running:.1f}%")
            
            # Log to wandb
            if USE_WANDB:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/accuracy": train_acc_running,
                    "train/batch": batch_idx + 1,
                    "epoch": epoch + 1,
                }, step=epoch * len(train_loader) + batch_idx)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_samples = 0
        val_loss = 0
        
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=dtype):
                seq_len = input_ids.size(1)
                outputs = model(
                    input_ids=input_ids,
                    compress_layers=(0,),
                    compress_recurrent=True,
                    compress_prefix_len=seq_len,
                    compress_num_tokens=config.compress_num_tokens,
                    recurrent_depth=0,
                )
                logits = outputs.logits[:, -1, :]
                target_token_ids = targets + value_offset
                loss = nn.functional.cross_entropy(logits, target_token_ids)
                predictions = logits.argmax(dim=-1)
                correct = (predictions == target_token_ids).sum().item()
                val_correct += correct
                val_samples += len(targets)
                val_loss += loss.item()
        
        val_acc = 100.0 * val_correct / val_samples
        avg_val_loss = val_loss / len(val_loader)
    
    train_acc = 100.0 * total_correct / total_samples
    avg_train_loss = total_loss / len(train_loader)
    
    print(f"\n{'='*80}")
    print(f"Epoch {epoch+1}/{num_epochs} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print(f"  Gap: {train_acc - val_acc:+.2f}%")
    
    # Log epoch summary to wandb
    if USE_WANDB:
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": avg_train_loss,
            "train/epoch_accuracy": train_acc,
            "val/loss": avg_val_loss,
            "val/accuracy": val_acc,
            "val/train_gap": train_acc - val_acc,
        }, step=(epoch + 1) * len(train_loader))
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"  ✓ New best validation accuracy: {val_acc:.2f}%")
        if USE_WANDB:
            wandb.run.summary["best_val_accuracy"] = val_acc
            wandb.run.summary["best_val_epoch"] = epoch + 1

print("\n" + "=" * 80)
print("FINAL EVALUATION on fresh test set")
print("=" * 80)

# Final test evaluation
model.eval()
with torch.no_grad():
    test_correct = 0
    test_samples = 0
    test_loss = 0
    
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=dtype):
            seq_len = input_ids.size(1)
            outputs = model(
                input_ids=input_ids,
                compress_layers=(0,),
                compress_recurrent=True,
                compress_prefix_len=seq_len,
                compress_num_tokens=config.compress_num_tokens,
                recurrent_depth=0,
            )
            logits = outputs.logits[:, -1, :]
            target_token_ids = targets + value_offset
            loss = nn.functional.cross_entropy(logits, target_token_ids)
            predictions = logits.argmax(dim=-1)
            correct = (predictions == target_token_ids).sum().item()
            test_correct += correct
            test_samples += len(targets)
            test_loss += loss.item()
    
    test_acc = 100.0 * test_correct / test_samples
    avg_test_loss = test_loss / len(test_loader)

print(f"\nTest Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Best Val Accuracy: {best_val_acc:.2f}%")
print(f"Final Train Accuracy: {train_acc:.2f}%")

# Log final test metrics to wandb
if USE_WANDB:
    wandb.log({
        "test/loss": avg_test_loss,
        "test/accuracy": test_acc,
    })
    wandb.run.summary.update({
        "final_test_accuracy": test_acc,
        "final_test_loss": avg_test_loss,
        "final_train_accuracy": train_acc,
        "best_val_accuracy": best_val_acc,
    })

print("\n" + "=" * 80)
print("ANALYSIS - RECURRENT COMPRESSION")
print("=" * 80)
print(f"Model parameters: {total_params:,}")
print(f"Compression: {8+2} tokens → {config.compress_num_tokens} tokens (recurrent)")
print(f"  - Method: Sequential processing with recurrent depth {config.recurrent_depth}")
print(f"  - Per step: Q={config.compress_num_tokens} compress → K,V={config.compress_num_tokens + 1} input")
print(f"  - Compress tokens attend only to input tokens (not each other)")
print(f"  - MLP processing on combined sequence")
print(f"  - Output: {config.compress_num_tokens} compress tokens")
print(f"Task: sum of {8} numbers mod {23}")
print(f"Train accuracy: {train_acc:.2f}%")
print(f"Val accuracy:   {best_val_acc:.2f}%")
print(f"Test accuracy:  {test_acc:.2f}%")

if test_acc > 80:
    print("\n✓ Model GENERALIZES well with recurrent compression!")
    print("  → Sequential processing maintains good performance")
elif test_acc > 50:
    print("\n⚠ Model shows PARTIAL generalization")
    print("  → Recurrent approach works but could be improved")
else:
    print("\n✗ Model shows POOR generalization")
    print("  → Recurrent compression may need tuning")

print("=" * 80)

# Finish wandb run
if USE_WANDB:
    wandb.finish()
