#!/usr/bin/env python3
"""
Extract key activations from memory model for each answer in the synthetic QA dataset.
For each answer value, record the top 5 keys with highest average activations across
BOTH memory layers, all answer tokens, and all heads.

This version properly:
1. Extracts activations for answer tokens only
2. Averages across tokens, heads, and layers
3. Correctly aligns batch processing
"""

import json
import torch
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset


class MemoryActivationExtractor:
    """Hook-based extractor for memory layer activations."""

    def __init__(self, model):
        self.model = model
        self.current_activations = {}
        self.hooks = []

    def __enter__(self):
        # Register hooks for memory layers
        for layer_idx, layer in enumerate(self.model.model.layers):
            if self._is_memory_layer(layer):
                hook = layer.mlp.register_forward_hook(self._create_hook(layer_idx))
                self.hooks.append(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove all hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _is_memory_layer(self, layer):
        """Check if layer contains a HashingMemory module."""
        return (
            hasattr(layer, "mlp")
            and hasattr(layer.mlp, "__class__")
            and "HashingMemory" in str(layer.mlp.__class__)
        )

    def _create_hook(self, layer_idx):
        """Create a forward hook to capture memory activations."""

        def hook(module, input, output):
            if hasattr(module, "keys"):
                # Get input tensor - maintain batch structure
                input_tensor = input[0]  # Shape: (B, T, input_dim)
                B, T, C = input_tensor.shape

                # Flatten for processing but remember the shape
                input_flat = input_tensor.view(-1, module.input_dim)  # (B*T, input_dim)

                # Compute query vectors
                query = module.query_proj(input_flat)  # (B*T*heads, k_dim)
                query = query.view(
                    B * T, module.heads, module.k_dim
                )  # (B*T, heads, k_dim)

                # Prepare keys for product quantization
                half = module.k_dim // 2
                keys = module.keys.view(module.heads, 2, -1, half)
                keys1 = keys[:, 0, :, :]  # (heads, n_keys, half)
                keys2 = keys[:, 1, :, :]  # (heads, n_keys, half)

                # Split queries
                q1 = query[:, :, :half]  # (B*T, heads, half)
                q2 = query[:, :, half:]  # (B*T, heads, half)

                # Compute raw dot product scores
                scores1 = torch.einsum(
                    "blh, lkh->blk", q1, keys1
                )  # (B*T, heads, n_keys)
                scores2 = torch.einsum(
                    "blh, lkh->blk", q2, keys2
                )  # (B*T, heads, n_keys)

                # Combine scores for product quantization
                n_keys = scores1.shape[-1]
                scores1_expanded = scores1.view(B * T, module.heads, n_keys, 1).expand(
                    B * T, module.heads, n_keys, n_keys
                )
                scores2_expanded = scores2.view(B * T, module.heads, 1, n_keys).expand(
                    B * T, module.heads, n_keys, n_keys
                )
                combined_scores = (
                    scores1_expanded + scores2_expanded
                )  # (B*T, heads, n_keys, n_keys)

                # Reshape to (B, T, heads, n_keys^2)
                combined_scores = combined_scores.view(B, T, module.heads, -1)

                self.current_activations[layer_idx] = {
                    "scores": combined_scores.cpu().float(),  # Keep original shape
                }

        return hook

    def get_activations(self):
        """Get and clear current activations."""
        activations = self.current_activations
        self.current_activations = {}
        return activations


def get_answer_token_positions(tokenizer, question, answer):
    """
    Find the token positions corresponding to the answer in the full Q&A text.
    Returns a list of position indices where answer tokens appear.
    """
    # Tokenize the full Q&A pair
    full_text = f"{question} {answer}"
    full_encoding = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
    full_ids = full_encoding["input_ids"][0]

    # Tokenize just the answer (without special tokens)
    answer_encoding = tokenizer(answer, return_tensors="pt", add_special_tokens=False)
    answer_ids = answer_encoding["input_ids"][0]

    # Find where answer tokens start in the full sequence
    # This is a simple search - could be made more robust
    answer_len = len(answer_ids)
    answer_positions = []

    for i in range(len(full_ids) - answer_len + 1):
        if torch.equal(full_ids[i : i + answer_len], answer_ids):
            answer_positions = list(range(i, i + answer_len))
            break

    return answer_positions


def extract_key_activations(model, tokenizer, dataset, batch_size=32):
    """
    Run model over dataset and extract key activations for each answer.
    Properly averages across answer tokens, heads, and layers.
    """
    model.eval()
    device = next(model.parameters()).device

    # Dictionary to accumulate scores for each answer
    # answer -> key_index -> {'sum': float, 'count': int}
    answer_key_stats = defaultdict(
        lambda: defaultdict(lambda: {"sum": 0.0, "count": 0})
    )

    # Track Q&A examples
    qa_examples = {}

    # Create activation extractor
    extractor = MemoryActivationExtractor(model)

    # Process dataset in batches
    with torch.no_grad(), extractor:
        for batch_start in tqdm(
            range(0, len(dataset), batch_size), desc="Processing batches"
        ):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_texts = []
            batch_qa_info = []  # Store (question, answer, answer_positions) for each sample

            # Prepare batch
            for i in range(batch_start, batch_end):
                sample = dataset[i]
                text = sample["text"]

                # Extract Q&A
                parts = text.split("? ")
                if len(parts) == 2:
                    question = parts[0] + "?"
                    answer = parts[1].strip()

                    # Find answer token positions
                    answer_positions = get_answer_token_positions(
                        tokenizer, question, answer
                    )

                    if answer_positions:
                        batch_texts.append(text)
                        batch_qa_info.append(
                            {
                                "question": question,
                                "answer": answer,
                                "answer_positions": answer_positions,
                            }
                        )

                        # Store example
                        if answer not in qa_examples:
                            qa_examples[answer] = question

            if not batch_texts:
                continue

            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            # Forward pass
            # input_ids = inputs["input_ids"].to(device)
            # attention_mask = inputs["attention_mask"].to(device)
            # outputs = model(input_ids, attention_mask=attention_mask)

            # Get activations from all layers
            layer_activations = extractor.get_activations()

            # Process each sample in the batch
            for sample_idx, qa_info in enumerate(batch_qa_info):
                answer = qa_info["answer"]
                answer_positions = qa_info["answer_positions"]

                # Accumulate scores across layers
                for layer_idx, activations in layer_activations.items():
                    if "scores" in activations and activations["scores"] is not None:
                        # Extract scores for this sample's answer tokens
                        # Shape: (B, T, heads, n_keys^2)
                        sample_scores = activations["scores"][
                            sample_idx
                        ]  # (T, heads, n_keys^2)

                        # Extract only answer token positions
                        answer_token_scores = sample_scores[
                            answer_positions
                        ]  # (n_answer_tokens, heads, n_keys^2)

                        # Average across answer tokens and heads
                        avg_scores = answer_token_scores.mean(dim=(0, 1))  # (n_keys^2,)

                        # Accumulate for this answer
                        for key_idx, score in enumerate(avg_scores):
                            answer_key_stats[answer][key_idx]["sum"] += score.item()
                            answer_key_stats[answer][key_idx]["count"] += 1

    # Now compute final averages and select top-5 keys per answer
    results = []
    for answer, key_stats in answer_key_stats.items():
        # Compute average score for each key (averaged across all layers and occurrences)
        key_avg_scores = []
        for key_idx, stats in key_stats.items():
            if stats["count"] > 0:
                avg_score = stats["sum"] / stats["count"]
                key_avg_scores.append((key_idx, avg_score))

        # Sort by average score and take top 5
        key_avg_scores.sort(key=lambda x: x[1], reverse=True)
        top_5_keys = key_avg_scores[:5]

        result = {
            "question": qa_examples[answer],
            "answer": answer,
            "top_keys": [
                {"key_index": key_idx, "avg_score": score}
                for key_idx, score in top_5_keys
            ],
        }
        results.append(result)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract memory key activations for each answer"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="exp/memory_2layer_qa_1M",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./synthetic_qa_data",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="memory_key_activations_per_answer.json",
        help="Path to save the output JSON file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to process (-1 for all)",
    )

    args = parser.parse_args()

    # Model and dataset paths from arguments
    model_path = args.model_path
    # dataset_path = args.dataset_path  # Currently unused

    # Add model directory and bento to path
    import sys
    import os

    sys.path.insert(0, model_path)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "3rdparty/bento"))

    print("Loading model and tokenizer...")
    try:
        from bento.models.memory.modeling_memory import MemoryForCausalLM
    except ImportError:
        from modeling_memory import MemoryForCausalLM

    model = MemoryForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    dataset = load_dataset("synthetic_qa_data", split="test")
    test_dataset = dataset

    # Limit number of samples if specified
    if args.num_samples > 0 and args.num_samples < len(test_dataset):
        test_dataset = test_dataset.select(range(args.num_samples))
        print(f"Limited to {args.num_samples} samples")

    print(f"Extracting key activations from {len(test_dataset)} samples...")
    # Extract activations
    results = extract_key_activations(
        model, tokenizer, test_dataset, batch_size=args.batch_size
    )

    # Save results
    output_path = args.output_path
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved key activations to {output_path}")
    print(f"Total unique answers analyzed: {len(results)}")

    # Print example
    if results:
        example = results[0]
        print("\nExample entry:")
        print(f"Question: {example['question']}")
        print(f"Answer: {example['answer']}")
        print("Top 5 keys (averaged across both layers, all heads, all answer tokens):")
        for i, key_data in enumerate(example["top_keys"]):
            print(
                f"  #{i + 1}: Key {key_data['key_index']} (avg_score={key_data['avg_score']:.6f})"
            )


if __name__ == "__main__":
    main()
