"""
BASELINE TEST: Standard Transformer (no compression).
- 2-layer standard transformer for comparison with recurrent compression
- Same dataset, hyperparameters, and training setup as test_compression_recurrent.py
- This establishes a baseline to compare against compression approaches
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent / "3rdparty" / "bento"))

from fla.models.transformer import TransformerConfig, TransformerForCausalLM
from modular_arithmetic_dataset import ModularArithmeticDataset, collate_fn

# WandB setup
try:
    import wandb
    USE_WANDB = True
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "adaptive-attention"),
        name="baseline-1layer-transformer",
        config={
            "architecture": "StandardTransformer",
            "compression_type": "none",
            "dataset": "modular_arithmetic",
        }
    )
except ImportError:
    USE_WANDB = False
    print("⚠️  wandb not installed. Install with: pip install wandb")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

print("=" * 80)
print("BASELINE TEST: Standard Transformer (NO compression)")
print("=" * 80)
print(f"Device: {device}")
print(f"Dtype: {dtype}")
print()
print("Architecture:")
print("  - Standard 1-layer Transformer")
print("  - No compression or sequence reduction")
print("  - Baseline for comparison with compression approaches")
print("=" * 80)

# Create datasets (SAME as recurrent test)
print("\nCreating datasets...")
train_dataset = ModularArithmeticDataset(
    num_samples=50000,
    sequence_length=8,
    modulo=97,
    value_range=10,
    seed=12345
)

val_dataset = ModularArithmeticDataset(
    num_samples=2000,
    sequence_length=8,
    modulo=97,
    value_range=10,
    seed=54321
)

test_dataset = ModularArithmeticDataset(
    num_samples=1000,
    sequence_length=8,
    modulo=97,
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
print(f"Task: (sum of 10 numbers) * (sum of 10 numbers) mod 97")

# Model: Standard 1-layer Transformer (BASELINE)
print("\nCreating BASELINE model (1-layer standard Transformer)...")
config = TransformerConfig(
    vocab_size=vocab_size,
    hidden_size=256,
    num_hidden_layers=1,  # 1 layer baseline
    num_heads=4,
    num_kv_heads=4,
    intermediate_size=512,
    max_position_embeddings=16,
)

# Update wandb config with model details
if USE_WANDB:
    wandb.config.update({
        "vocab_size": vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_heads": config.num_heads,
        "intermediate_size": config.intermediate_size,
        "sequence_length": 8,
        "modulo": 97,
        "value_range": 10,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "num_epochs": 10,
    })

model = TransformerForCausalLM(config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")
print(f"Layers: {config.num_hidden_layers}")
print(f"Architecture: Standard Transformer (no compression)")

# Training setup with stability improvements (SAME as recurrent test)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, eps=1e-8)

print("\n" + "=" * 80)
print("Training for 10 epochs (baseline)...")
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
        
        # Forward (standard transformer, no compression)
        with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=dtype):
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]  # Last position
            target_token_ids = targets + value_offset
            loss = nn.functional.cross_entropy(logits, target_token_ids)
        
        # Check for NaN BEFORE backward to prevent corrupting gradients
        if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"  WARNING: NaN/Inf detected at batch {batch_idx+1}, skipping backward pass...")
            continue
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients for NaN before updating weights
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"  WARNING: NaN/Inf gradients at batch {batch_idx+1}, skipping optimizer step...")
            optimizer.zero_grad()  # Clear the bad gradients
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
                outputs = model(input_ids=input_ids)
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
            outputs = model(input_ids=input_ids)
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
print("ANALYSIS - BASELINE (1-layer Transformer)")
print("=" * 80)
print(f"Model parameters: {total_params:,}")
print(f"Architecture: 1-layer standard Transformer")
print(f"  - No compression or sequence reduction")
print(f"  - Full attention over entire sequence")
print(f"Train accuracy: {train_acc:.2f}%")
print(f"Val accuracy:   {best_val_acc:.2f}%")
print(f"Test accuracy:  {test_acc:.2f}%")

if test_acc > 80:
    print("\n✓ Model GENERALIZES well (baseline performance)")
elif test_acc > 50:
    print("\n⚠ Model shows PARTIAL generalization")
else:
    print("\n✗ Model shows POOR generalization")

print("\n" + "=" * 80)
print("COMPARISON NOTE:")
print("Compare this baseline with recurrent compression results:")
print("  - Parameter efficiency")
print("  - Training speed")
print("  - Generalization performance")
print("=" * 80)

# Finish wandb run
if USE_WANDB:
    wandb.finish()

