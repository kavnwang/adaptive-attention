#!/usr/bin/env python3
"""
Track Top-5 Token Activations for Memory Layer Keys

This script processes text through a 2-layer memory model and tracks the top 5 tokens
that maximally activate each of the (num_keys)² * num_layers keys in the model.

Model Structure:
- 2 memory layers (positions 0 and 1)
- 1024² = 1,048,576 keys per layer via product quantization
- 4 memory heads per layer
- Total keys to track: 2,097,152
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
from tqdm import tqdm
import heapq
from typing import Dict, List, Tuple
from collections import defaultdict
from datasets import load_dataset

# Import memory model components
import sys

sys.path.append("/home/kevin/LLMonade_interp/3rdparty/bento")
from bento.layers.feature_mixers.memory import HashingMemoryRetrieval as HashingMemory


class KeyActivationTracker:
    """Tracks top-5 token activations for each memory key."""

    def __init__(self, num_layers: int, num_keys_per_layer: int, num_heads: int):
        """
        Initialize tracker for memory key activations.

        Args:
            num_layers: Number of memory layers (2)
            num_keys_per_layer: Number of keys per layer (1024² = 1,048,576)
            num_heads: Number of memory heads per layer (4)
        """
        self.num_layers = num_layers
        self.num_keys_per_layer = num_keys_per_layer
        self.num_heads = num_heads

        # Track top-5 activations per key: (layer, head, key_idx) -> min_heap of (activation, token_id, token_str)
        self.key_activations: Dict[Tuple[int, int, int], List] = defaultdict(list)
        self.token_counter = 0

    def update_activations(
        self,
        layer_idx: int,
        head_idx: int,
        key_indices: torch.Tensor,  # (batch_size * seq_len, knn)
        similarities: torch.Tensor,  # (batch_size * seq_len, knn)
        token_ids: torch.Tensor,  # (batch_size, seq_len)
        tokenizer,
    ):
        """
        Update top-5 activations for retrieved keys.

        Args:
            layer_idx: Memory layer index (0 or 1)
            head_idx: Memory head index (0-3)
            key_indices: Retrieved key indices from k-NN search
            similarities: Similarity scores (post-softmax probabilities)
            token_ids: Input token IDs
            tokenizer: Tokenizer for converting IDs to strings
        """
        batch_size, seq_len = token_ids.shape

        # Flatten token_ids to match key_indices shape
        flat_token_ids = token_ids.view(-1)  # (batch_size * seq_len,)

        # Process each position
        for pos_idx in range(len(flat_token_ids)):
            token_id = flat_token_ids[pos_idx].item()
            token_str = tokenizer.decode([token_id])

            # Get retrieved keys and their similarities for this position
            pos_key_indices = key_indices[pos_idx]  # (knn,)
            pos_similarities = similarities[pos_idx]  # (knn,)

            # Update activations for each retrieved key
            for key_rel_idx, similarity in zip(pos_key_indices, pos_similarities):
                key_abs_idx = key_rel_idx.item()
                activation_score = similarity.item()

                key_tuple = (layer_idx, head_idx, key_abs_idx)

                # Use min-heap to maintain top-5 activations
                heap = self.key_activations[key_tuple]

                if len(heap) < 5:
                    heapq.heappush(
                        heap,
                        (activation_score, token_id, token_str, self.token_counter),
                    )
                elif (
                    activation_score > heap[0][0]
                ):  # New activation is higher than minimum
                    heapq.heapreplace(
                        heap,
                        (activation_score, token_id, token_str, self.token_counter),
                    )

            self.token_counter += 1

    def get_top_activations(self) -> Dict:
        """Get top-5 activations for all keys in a structured format."""
        result = {}

        for (layer_idx, head_idx, key_idx), heap in self.key_activations.items():
            # Sort heap to get top activations in descending order
            top_activations = sorted(heap, key=lambda x: x[0], reverse=True)

            key_name = f"layer_{layer_idx}_head_{head_idx}_key_{key_idx}"
            result[key_name] = [
                {
                    "activation": float(activation),
                    "token_id": int(token_id),
                    "token": token_str,
                    "position": int(position),
                }
                for activation, token_id, token_str, position in top_activations
            ]

        return result


def hook_memory_layer(layer_idx: int, tracker: KeyActivationTracker, tokenizer):
    """Create forward hook for memory layer to capture key activations."""

    def hook_fn(module, input, output):
        """Hook function to capture memory key activations."""
        if not isinstance(module, HashingMemory):
            return

        # Get current token_ids from the global context (set during processing)
        if hasattr(hook_fn, "current_token_ids"):
            token_ids = hook_fn.current_token_ids
        else:
            return  # Skip if no token_ids available

        # Access the module's stored indices and scores (available in eval mode)
        if hasattr(module, "last_indices") and hasattr(module, "last_scores"):
            # last_indices: (batch_size * seq_len, num_heads, knn)
            # last_scores: (batch_size * seq_len, num_heads, knn)
            retrieved_indices = module.last_indices  # Already moved to CPU
            similarities = module.last_scores  # Already moved to CPU

            batch_size, seq_len = token_ids.shape

            # Update tracker for each head
            for head_idx in range(module.heads):
                head_indices = retrieved_indices[
                    :, head_idx, :
                ]  # (batch_size * seq_len, knn)
                head_similarities = similarities[
                    :, head_idx, :
                ]  # (batch_size * seq_len, knn)

                tracker.update_activations(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    key_indices=head_indices,
                    similarities=head_similarities,
                    token_ids=token_ids,
                    tokenizer=tokenizer,
                )

    return hook_fn


def load_model_and_tokenizer(model_path: str):
    """Load the memory model and tokenizer."""
    print(f"Loading model from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True
    )
    model.eval()

    return model, tokenizer


def load_ultra_fine_web_sample(num_samples: int = 1000, max_length: int = 512):
    """Load a sample from Ultra_Fine_Web dataset."""
    print(f"Loading {num_samples} samples from Ultra_Fine_Web dataset")

    # Load dataset
    try:
        dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
        samples = []

        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            text = sample["text"]
            if len(text.strip()) > 50:  # Skip very short texts
                samples.append(text[: max_length * 4])  # Rough character limit

        print(f"Loaded {len(samples)} text samples")
        return samples

    except Exception as e:
        print(f"Failed to load Ultra_Fine_Web: {e}")
        print("Using fallback dummy data")

        # Fallback: Generate some dummy text
        dummy_texts = [
            "The quick brown fox jumps over the lazy dog. This is a test sentence for memory activation tracking.",
            "Machine learning models use neural networks to process information and make predictions.",
            "Natural language processing involves understanding and generating human language using computational methods.",
            "Memory networks can store and retrieve information efficiently using key-value mechanisms.",
            "Artificial intelligence systems are becoming increasingly sophisticated and capable.",
        ] * (num_samples // 5 + 1)

        return dummy_texts[:num_samples]


def process_texts_with_model(
    model,
    tokenizer,
    texts: List[str],
    tracker: KeyActivationTracker,
    batch_size: int = 4,
    max_length: int = 512,
):
    """Process texts through the model while tracking memory activations."""

    # Register hooks for memory layers
    hooks = []
    memory_layer_indices = [0, 1]  # Based on config: memory_layers = [0, 1]

    for layer_idx in memory_layer_indices:
        # Access the memory layer within the model
        # Structure: model.transformer.h[layer_idx].mlp (which is a HashingMemory)
        memory_module = model.transformer.h[layer_idx].mlp

        hook_fn = hook_memory_layer(layer_idx, tracker, tokenizer)
        hook_handle = memory_module.register_forward_hook(hook_fn)
        hooks.append((hook_handle, hook_fn))

    try:
        print(f"Processing {len(texts)} texts in batches of {batch_size}")

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i : i + batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # Move to model device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Set current token_ids in hook functions
            for _, hook_fn in hooks:
                hook_fn.current_token_ids = inputs["input_ids"]

            # Forward pass through model
            with torch.no_grad():
                pass  # outputs = model(**inputs, output_hidden_states=True)

        print(
            f"Completed processing. Tracked activations for {len(tracker.key_activations)} unique keys."
        )

    finally:
        # Remove hooks
        for hook_handle, _ in hooks:
            hook_handle.remove()


def main():
    parser = argparse.ArgumentParser(description="Track memory layer key activations")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/kevin/LLMonade_interp/exp/memory_2layer_general_1M/hfcheckpoint/step-25000",
        help="Path to the memory model",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="memory_key_activations.json",
        help="Output file for activation results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of text samples to process",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for processing"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # Initialize activation tracker
    # Based on config: 2 layers, 1024² keys per layer, 1 head per layer
    tracker = KeyActivationTracker(
        num_layers=2,
        num_keys_per_layer=1024 * 1024,  # 1024² = 1,048,576
        num_heads=1,  # From config: mem_heads = 1
    )

    # Load text data
    texts = load_ultra_fine_web_sample(args.num_samples, args.max_length)

    # Process texts and track activations
    process_texts_with_model(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        tracker=tracker,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Save results
    print(f"Saving results to {args.output_file}")
    results = {
        "config": {
            "model_path": args.model_path,
            "num_samples": args.num_samples,
            "num_layers": 2,
            "num_keys_per_layer": 1024 * 1024,
            "num_heads": 4,
            "total_keys_tracked": len(tracker.key_activations),
        },
        "top_activations": tracker.get_top_activations(),
    }

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"✅ Completed! Tracked activations for {len(tracker.key_activations)} unique keys"
    )
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
