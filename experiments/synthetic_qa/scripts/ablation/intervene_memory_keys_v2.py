#!/usr/bin/env python3
"""
Memory key intervention analysis - Version 2.

This script:
1. First pass: Builds a database of (idx1, idx2, position, token) from natural generation
2. Second pass: Tests if forcing those keys at those positions reproduces the tokens
3. Creates position-digit heatmap showing intervention success rates
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import sys
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

# Add bento path for custom model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../3rdparty/bento"))
from bento.models.memory.modeling_memory import MemoryForCausalLM, MemoryLayer


class KeyPositionTokenCollector:
    """Collects (idx1, idx2, position, token) quadruples during natural generation."""

    def __init__(self):
        self.quadruples = set()  # Store unique quadruples
        self.current_position = 0

    def reset_position(self):
        self.current_position = 0

    def collect_hook(self, memory_layer, layer_idx):
        """Create a hook that collects key selections during forward pass."""
        memory_retrieval = memory_layer.memory_retrieval
        original_forward = memory_retrieval.forward

        def collecting_forward(query, knn, return_individual_probs=False):
            # Call original forward
            if return_individual_probs:
                scores, indices, probs1, probs2 = original_forward(
                    query, knn, return_individual_probs=True
                )
            else:
                scores, indices = original_forward(
                    query, knn, return_individual_probs=False
                )
                probs1, probs2 = None, None

            # Extract selected keys
            n_keys = memory_retrieval.mem_n_keys
            bs_times_heads = indices.shape[0]
            heads = memory_retrieval.heads
            bs = bs_times_heads // heads

            # We're interested in the last position (current generation step)
            if bs > 0:
                for head_idx in range(heads):
                    pos_idx = (bs - 1) * heads + head_idx

                    # Get top-1 key for this head
                    if indices.dim() == 1:
                        combined_idx = indices[pos_idx].item()
                    else:
                        combined_idx = indices[pos_idx, 0].item()  # Top-1

                    # Decode to idx1, idx2
                    idx1 = combined_idx // n_keys
                    idx2 = combined_idx % n_keys

                    # Store for later (we'll add the token after generation)
                    if not hasattr(self, "_pending_keys"):
                        self._pending_keys = []
                    self._pending_keys.append((idx1, idx2, layer_idx, head_idx))

            if return_individual_probs:
                return scores, indices, probs1, probs2
            else:
                return scores, indices

        memory_retrieval.forward = collecting_forward
        return original_forward


def build_key_position_token_database(
    model, tokenizer, qa_pairs, num_samples, device="cuda"
):
    """
    Phase 1: Build database of (idx1, idx2, position, token) from natural generation.
    """
    print(
        f"\nPhase 1: Building key-position-token database from {num_samples} QA pairs..."
    )

    collector = KeyPositionTokenCollector()
    database = []  # List of quadruples with metadata

    # Sample QA pairs for database building
    sampled_pairs = random.sample(qa_pairs, min(num_samples, len(qa_pairs)))

    for qa_idx, (question, answer) in enumerate(
        tqdm(sampled_pairs, desc="Building database")
    ):
        # Tokenize
        inputs = tokenizer(question, return_tensors="pt").to(device)
        value_tokens = tokenizer(value, add_special_tokens=False, return_tensors="pt")
        value_token_ids = value_tokens["input_ids"][0].to(device)

        # Map token positions to digit positions
        token_to_digit_pos = {}
        digit_pos = 0
        for token_pos, token_id in enumerate(value_token_ids):
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            if token_text and token_text[0].isdigit():
                token_to_digit_pos[token_pos] = digit_pos
                digit_pos += 1

        # Hook all memory layers
        original_methods = []
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "mlp") and isinstance(layer.mlp, MemoryLayer):
                original_method = collector.collect_hook(layer.mlp, layer_idx)
                original_methods.append((layer.mlp, original_method))

        # Generate answer token by token (teacher forcing)
        current_ids = inputs["input_ids"]

        for pos in range(len(answer_token_ids)):
            # Reset pending keys
            collector._pending_keys = []

            # Generate next token
            with torch.no_grad():
                outputs = model(input_ids=current_ids)
                logits = outputs.logits[0, -1, :]
                predicted_token = logits.argmax().item()

            # Decode token
            predicted_text = tokenizer.decode(
                [predicted_token], skip_special_tokens=True
            )

            # Record quadruples if this is a digit position
            if (
                pos in token_to_digit_pos
                and predicted_text
                and predicted_text[0].isdigit()
            ):
                digit_position = token_to_digit_pos[pos]
                predicted_digit = int(predicted_text[0])

                # Add all keys that were used for this position
                for idx1, idx2, layer_idx, head_idx in collector._pending_keys:
                    quadruple = {
                        "idx1": idx1,
                        "idx2": idx2,
                        "digit_position": digit_position,
                        "token_position": pos,
                        "generated_token": predicted_token,
                        "generated_digit": predicted_digit,
                        "layer": layer_idx,
                        "head": head_idx,
                        "qa_idx": qa_idx,
                        "question": question,
                        "answer": answer,
                    }
                    database.append(quadruple)

            # Use true token for next position (teacher forcing)
            current_ids = torch.cat(
                [current_ids, answer_token_ids[pos : pos + 1].unsqueeze(0)], dim=1
            )

        # Restore original methods
        for memory_layer, original_method in original_methods:
            memory_layer.memory_retrieval.forward = original_method

    # Deduplicate based on (idx1, idx2, digit_position, generated_digit)
    unique_quadruples = {}
    for quad in database:
        key = (
            quad["idx1"],
            quad["idx2"],
            quad["digit_position"],
            quad["generated_digit"],
        )
        if key not in unique_quadruples:
            unique_quadruples[key] = quad

    print(f"\nCollected {len(database)} total quadruples")
    print(f"Unique quadruples: {len(unique_quadruples)}")

    # Show digit distribution
    digit_counts = defaultdict(int)
    for quad in unique_quadruples.values():
        digit_counts[quad["generated_digit"]] += 1
    print(f"Digit distribution: {dict(digit_counts)}")

    return list(unique_quadruples.values())


def hook_memory_retrieval_for_intervention(
    memory_layer, layer_idx, forced_key, position_idx
):
    """
    Hook to force a specific key at a specific position.
    """
    memory_retrieval = memory_layer.memory_retrieval
    original_forward = memory_retrieval.forward

    def intervened_forward(query, knn, return_individual_probs=False):
        # Call original to get natural results
        if return_individual_probs:
            scores, indices, probs1, probs2 = original_forward(
                query, knn, return_individual_probs=True
            )
        else:
            scores, indices = original_forward(
                query, knn, return_individual_probs=False
            )
            probs1, probs2 = None, None

        # Convert forced key to combined index
        idx1, idx2 = forced_key
        n_keys = memory_retrieval.mem_n_keys
        combined_idx = idx1 * n_keys + idx2

        # Force this key for all heads
        # The hook is only installed for the specific position we want to intervene at,
        # so we always apply the intervention when this function is called
        heads = memory_retrieval.heads
        for head_idx in range(heads):
            if indices.dim() == 1:
                # Shape: (heads,) - directly index by head
                indices[head_idx] = combined_idx
            else:
                # Shape: (heads, knn) - replace the top-1 selection with our forced key
                indices[head_idx, 0] = combined_idx

        if return_individual_probs:
            return scores, indices, probs1, probs2
        else:
            return scores, indices

    memory_retrieval.forward = intervened_forward
    return original_forward


def test_intervention(
    model,
    tokenizer,
    question,
    answer,
    forced_key,
    position,
    expected_token,
    device="cuda",
):
    """
    Test if forcing a specific key at a specific position produces the expected token.
    """
    inputs = tokenizer(question, return_tensors="pt").to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
    answer_token_ids = answer_tokens["input_ids"][0].to(device)

    # Hook memory layers
    original_methods = []
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, MemoryLayer):
            original_method = hook_memory_retrieval_for_intervention(
                layer.mlp, layer_idx, forced_key, position
            )
            original_methods.append((layer.mlp, original_method))

    # Generate with teacher forcing up to the intervention position
    current_ids = inputs["input_ids"]

    for pos in range(min(position + 1, len(answer_token_ids))):
        with torch.no_grad():
            outputs = model(input_ids=current_ids)
            logits = outputs.logits[0, -1, :]
            predicted_token = logits.argmax().item()

        if pos == position:
            # This is where we check the intervention result
            result = {
                "predicted_token": predicted_token,
                "expected_token": expected_token,
                "success": predicted_token == expected_token,
                "predicted_text": tokenizer.decode(
                    [predicted_token], skip_special_tokens=True
                ),
                "expected_text": tokenizer.decode(
                    [expected_token], skip_special_tokens=True
                ),
            }
            break

        # Use true token for next position
        if pos < len(answer_token_ids):
            current_ids = torch.cat(
                [current_ids, answer_token_ids[pos : pos + 1].unsqueeze(0)], dim=1
            )

    # Restore original methods
    for memory_layer, original_method in original_methods:
        memory_layer.memory_retrieval.forward = original_method

    return result


def test_interventions_on_new_pairs(
    model, tokenizer, quadruples, qa_pairs, num_samples, device="cuda"
):
    """
    Phase 2: Test if forcing keys from the database produces expected tokens on new QA pairs.
    """
    print(f"\nPhase 2: Testing interventions on {num_samples} new QA pairs...")

    # Sample different QA pairs for testing
    test_pairs = random.sample(qa_pairs, min(num_samples, len(qa_pairs)))

    # Track results by (digit_position, expected_digit)
    position_digit_success = defaultdict(lambda: defaultdict(int))
    position_digit_attempts = defaultdict(lambda: defaultdict(int))
    detailed_results = []

    for qa_idx, (question, answer) in enumerate(
        tqdm(test_pairs, desc="Testing interventions")
    ):
        # Skip if answer has no digits
        answer_digits = [int(c) for c in answer if c.isdigit()]
        if not answer_digits:
            continue

        # Map token positions to digit positions
        value_tokens = tokenizer(value, add_special_tokens=False, return_tensors="pt")
        value_token_ids = value_tokens["input_ids"][0].to(device)

        token_to_digit_pos = {}
        digit_pos = 0
        for token_pos, token_id in enumerate(value_token_ids):
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            if token_text and token_text[0].isdigit():
                token_to_digit_pos[token_pos] = digit_pos
                digit_pos += 1

        # Test each relevant quadruple on this QA pair
        for quad in quadruples:
            digit_position = quad["digit_position"]
            expected_digit = quad["generated_digit"]

            # Find token position for this digit position
            token_position = None
            for t_pos, d_pos in token_to_digit_pos.items():
                if d_pos == digit_position:
                    token_position = t_pos
                    break

            if token_position is not None and token_position < len(answer_token_ids):
                # Test intervention
                forced_key = (quad["idx1"], quad["idx2"])
                expected_token = quad["generated_token"]

                result = test_intervention(
                    model,
                    tokenizer,
                    question,
                    answer,
                    forced_key,
                    token_position,
                    expected_token,
                    device,
                )

                # Track statistics
                position_digit_attempts[digit_position][expected_digit] += 1
                if result["success"]:
                    position_digit_success[digit_position][expected_digit] += 1

                # Store detailed result
                detailed_results.append(
                    {
                        "qa_pair": {"question": question, "answer": answer},
                        "quadruple": quad,
                        "intervention_result": result,
                        "digit_position": digit_position,
                        "token_position": token_position,
                    }
                )

    # Compute success rates
    position_digit_rates = {}
    for pos in range(10):  # Max 10 digit positions
        position_digit_rates[pos] = {}
        for digit in range(10):
            attempts = position_digit_attempts[pos][digit]
            if attempts > 0:
                success = position_digit_success[pos][digit]
                position_digit_rates[pos][digit] = success / attempts
            else:
                position_digit_rates[pos][digit] = 0.0

    return position_digit_rates, detailed_results


def visualize_position_digit_mapping(position_digit_rates, output_path):
    """Create heatmap of position-digit intervention success rates."""
    # Find max position with data
    max_pos = (
        max(
            [
                p
                for p in position_digit_rates.keys()
                if any(position_digit_rates[p].values())
            ]
        )
        + 1
    )

    # Create matrix
    matrix = np.zeros((max_pos, 10))

    for pos in range(max_pos):
        for digit in range(10):
            matrix[pos, digit] = position_digit_rates[pos][digit]

    # Create heatmap
    plt.figure(figsize=(12, max(6, max_pos * 0.5)))

    sns.heatmap(
        matrix * 100,  # Convert to percentage
        cmap="RdYlBu_r",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "Success Rate (%)"},
        xticklabels=range(10),
        yticklabels=[f"Position {i}" for i in range(max_pos)],
        square=True,
    )

    plt.title("Key Intervention Success: Position-Digit Mapping", fontsize=14)
    plt.xlabel("Expected Digit", fontsize=12)
    plt.ylabel("Digit Position in Answer", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Memory key intervention analysis v2")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to tokenizer"
    )
    parser.add_argument(
        "--dataset_size",
        type=str,
        choices=["200", "2K", "10K", "50K"],
        required=True,
        help="Dataset size to use",
    )
    parser.add_argument(
        "--num_samples_phase1",
        type=int,
        default=100,
        help="Number of QA pairs for building database",
    )
    parser.add_argument(
        "--num_samples_phase2",
        type=int,
        default=100,
        help="Number of QA pairs for testing interventions",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output path for visualization",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading model and tokenizer...")
    model = MemoryForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"\nLoading dataset (size: {args.dataset_size})...")
    dataset_path = (
        f"experiments/synthetic_qa/datasets/synthetic_qa_data_{args.dataset_size}"
    )
    dataset = load_dataset(dataset_path, split="test")

    # Extract QA pairs
    qa_pairs = []
    for sample in dataset:
        text = sample["text"]
        parts = text.split(":")
        if len(parts) == 2:
            key = parts[0].strip() + ":"
            value = parts[1].strip()
            qa_pairs.append((question, answer))

    print(f"Loaded {len(qa_pairs)} QA pairs")

    # Phase 1: Build database
    quadruples = build_key_position_token_database(
        model, tokenizer, qa_pairs, args.num_samples_phase1, device=args.device
    )

    # Phase 2: Test interventions
    position_digit_rates, detailed_results = test_interventions_on_new_pairs(
        model,
        tokenizer,
        quadruples,
        qa_pairs,
        args.num_samples_phase2,
        device=args.device,
    )

    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Visualize results
    visualize_position_digit_mapping(position_digit_rates, output_path)

    # Save comprehensive results
    results_data = {
        "config": {
            "model_path": args.model_path,
            "dataset_size": args.dataset_size,
            "num_samples_phase1": args.num_samples_phase1,
            "num_samples_phase2": args.num_samples_phase2,
            "seed": args.seed,
        },
        "summary": {
            "num_unique_quadruples": len(quadruples),
            "position_digit_success_rates": position_digit_rates,
            "total_interventions_tested": len(detailed_results),
        },
        "quadruples_database": quadruples,
        "detailed_intervention_results": detailed_results[
            :100
        ],  # Limit to first 100 for file size
    }

    json_path = output_path.parent / f"{output_path.stem}_comprehensive.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    # Save summary
    summary_data = {
        "config": results_data["config"],
        "summary": results_data["summary"],
    }

    summary_path = output_path.parent / f"{output_path.stem}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved comprehensive data to {json_path}")
    print(f"Saved summary to {summary_path}")

    # Print summary statistics
    print("\n=== Summary ===")
    print(
        f"Built database of {len(quadruples)} unique (key, position, digit) associations"
    )
    print(f"Tested {len(detailed_results)} interventions")

    # Find strongest associations
    print("\nStrongest position-digit mappings (>80% success):")
    for pos in sorted(position_digit_rates.keys()):
        for digit in range(10):
            rate = position_digit_rates[pos][digit]
            if rate > 0.8:
                attempts = sum(
                    1
                    for r in detailed_results
                    if r["digit_position"] == pos
                    and r["quadruple"]["generated_digit"] == digit
                )
                print(
                    f"  Position {pos} -> Digit {digit}: {rate * 100:.1f}% ({attempts} attempts)"
                )


if __name__ == "__main__":
    main()
