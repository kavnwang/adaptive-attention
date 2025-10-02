#!/usr/bin/env python3
"""
Intervene on memory keys to force specific retrievals and measure position-digit mapping.

This script:
1. Loads all keys that have been activated at least once
2. For random QA pairs, forces each key to be retrieved at each position
3. Measures how often forcing a key causes generation of its associated digit
4. Creates a heatmap showing position-digit mapping strength
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
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

# Add bento path for custom model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../3rdparty/bento"))
from bento.models.memory.modeling_memory import MemoryForCausalLM, MemoryLayer


def load_activated_keys(activation_file):
    """Load all unique activated keys and build key-to-digit mapping."""
    with open(activation_file, "r") as f:
        data = json.load(f)

    # Extract all unique keys and their associated digits
    key_to_digits = defaultdict(list)  # (idx1, idx2) -> list of digits (with positions)
    all_keys = set()

    for entry in data:
        value = entry["value"]
        # Extract digits from value
        value_digits = [int(c) for c in value if c.isdigit()]

        if not value_digits:
            continue

        # Get keys from each layer
        for layer_name, top_keys in entry["top_keys_per_layer"].items():
            for key_info in top_keys:
                key = (key_info["idx1"], key_info["idx2"])
                all_keys.add(key)
                # Associate this key with all digits in this value
                key_to_digits[key].extend(value_digits)

    # Convert to primary digit mapping (most common digit for each key)
    key_to_primary_digit = {}
    for key, digit_list in key_to_digits.items():
        if digit_list:
            # Count occurrences of each digit
            digit_counts = Counter(digit_list)
            # Take the most common digit
            most_common_digit = digit_counts.most_common(1)[0][0]
            key_to_primary_digit[key] = most_common_digit

    print(f"Loaded {len(all_keys)} unique keys")
    print("Key-digit mapping examples:")
    for i, (key, digit) in enumerate(list(key_to_primary_digit.items())[:10]):
        print(f"  Key {key} -> Digit {digit}")

    # Show digit distribution
    digit_distribution = Counter(key_to_primary_digit.values())
    print(f"\nDigit distribution: {dict(digit_distribution)}")

    return all_keys, key_to_primary_digit


def hook_memory_retrieval_for_intervention(
    memory_layer, layer_idx, forced_key, position_idx
):
    """
    Hook into memory retrieval to force a specific key at a specific position.
    If forced_key is None, no intervention is performed (natural generation).
    Returns the original forward method.
    """
    memory_retrieval = memory_layer.memory_retrieval
    original_forward = memory_retrieval.forward

    # If no forced key, just return original method (no intervention)
    if forced_key is None:
        return original_forward

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

        # Convert forced key (idx1, idx2) to combined index
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


def generate_with_forced_key(model, tokenizer, key, value, forced_key, device="cuda"):
    """
    Generate value with a forced key and track which digits are generated.
    Uses teacher forcing to isolate effects at each position.
    """
    inputs = tokenizer(key, return_tensors="pt").to(device)
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

    results = []
    current_ids = inputs["input_ids"]

    # Generate each position with teacher forcing
    for pos in range(len(value_token_ids)):
        # Set up intervention for this position
        original_methods = []
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "mlp") and isinstance(layer.mlp, MemoryLayer):
                original_method = hook_memory_retrieval_for_intervention(
                    layer.mlp, layer_idx, forced_key, pos
                )
                original_methods.append((layer.mlp, original_method))

        # Generate next token
        with torch.no_grad():
            outputs = model(input_ids=current_ids)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            predicted_token = logits.argmax().item()

        # Decode to check if it's a digit
        predicted_text = tokenizer.decode([predicted_token], skip_special_tokens=True)
        if predicted_text and predicted_text[0].isdigit():
            predicted_digit = int(predicted_text[0])
        else:
            predicted_digit = -1  # Not a digit

        # Get digit position if this token position should contain a digit
        digit_position = token_to_digit_pos.get(pos, -1)

        results.append(
            {
                "token_position": pos,
                "digit_position": digit_position,
                "predicted_token": predicted_token,
                "predicted_digit": predicted_digit,
                "true_token": value_token_ids[pos].item(),
            }
        )

        # Restore original methods
        for memory_layer, original_method in original_methods:
            memory_layer.memory_retrieval.forward = original_method

        # Use true token for next position (teacher forcing)
        current_ids = torch.cat(
            [current_ids, value_token_ids[pos : pos + 1].unsqueeze(0)], dim=1
        )

    return results


def analyze_position_digit_mapping(
    model,
    tokenizer,
    qa_pairs,
    all_keys,
    key_to_digit,
    num_samples=100,
    max_positions=10,
    device="cuda",
):
    """
    Analyze how forcing keys affects digit generation at each position.
    Returns both summary statistics and detailed intervention results.
    """
    # Initialize tracking: position -> digit -> success count
    position_digit_success = defaultdict(lambda: defaultdict(int))
    position_digit_attempts = defaultdict(lambda: defaultdict(int))

    # Detailed tracking of all interventions
    all_interventions = []

    # Group keys by their associated digit
    digit_to_keys = defaultdict(list)
    for key, digit in key_to_digit.items():
        digit_to_keys[digit].append(key)

    print(f"\nAnalyzing {num_samples} random QA pairs...")
    print(f"Keys per digit: {[(d, len(keys)) for d, keys in digit_to_keys.items()]}")

    # Sample random QA pairs
    sampled_pairs = random.sample(qa_pairs, min(num_samples, len(qa_pairs)))

    for qa_idx, (key, value) in enumerate(
        tqdm(sampled_pairs, desc="Processing key-value pairs")
    ):
        # Skip if value has no digits
        value_digits = [int(c) for c in value if c.isdigit()]
        if not value_digits:
            continue

        # Store natural generation first (no intervention)
        natural_results = generate_with_forced_key(
            model, tokenizer, key, value, None, device
        )

        # For each digit that has associated keys
        for target_digit, keys_for_digit in digit_to_keys.items():
            # Sample a few keys for this digit (to avoid too many interventions)
            sampled_keys = random.sample(keys_for_digit, min(5, len(keys_for_digit)))

            for forced_key in sampled_keys:
                # Generate with this forced key
                intervention_results = generate_with_forced_key(
                    model, tokenizer, key, value, forced_key, device
                )

                # Create detailed intervention record
                intervention_record = {
                    "qa_pair": {
                        "key": key,
                        "value": value,
                        "value_digits": value_digits,
                    },
                    "forced_key": {
                        "idx1": forced_key[0],
                        "idx2": forced_key[1],
                        "associated_digit": target_digit,
                    },
                    "results_by_position": [],
                }

                # Track results at each position
                for i, (nat_res, int_res) in enumerate(
                    zip(natural_results, intervention_results)
                ):
                    position_record = {
                        "token_position": int_res["token_position"],
                        "digit_position": int_res["digit_position"],
                        "natural": {
                            "predicted_token": nat_res["predicted_token"],
                            "predicted_digit": nat_res["predicted_digit"],
                            "predicted_text": tokenizer.decode(
                                [nat_res["predicted_token"]], skip_special_tokens=True
                            ),
                        },
                        "intervened": {
                            "predicted_token": int_res["predicted_token"],
                            "predicted_digit": int_res["predicted_digit"],
                            "predicted_text": tokenizer.decode(
                                [int_res["predicted_token"]], skip_special_tokens=True
                            ),
                        },
                        "true_token": int_res["true_token"],
                        "true_text": tokenizer.decode(
                            [int_res["true_token"]], skip_special_tokens=True
                        ),
                        "changed": nat_res["predicted_token"]
                        != int_res["predicted_token"],
                    }

                    # Add to detailed record
                    intervention_record["results_by_position"].append(position_record)

                    # Update summary statistics for digit positions
                    digit_position = int_res["digit_position"]
                    if digit_position >= 0 and digit_position < max_positions:
                        predicted_digit = int_res["predicted_digit"]

                        # Count attempt for this digit-position-target-digit pair
                        position_digit_attempts[digit_position][target_digit] += 1

                        # Count success if predicted digit matches the key's digit
                        if predicted_digit == target_digit:
                            position_digit_success[digit_position][target_digit] += 1

                # Add summary to intervention record
                intervention_record["summary"] = {
                    "num_positions_changed": sum(
                        1
                        for p in intervention_record["results_by_position"]
                        if p["changed"]
                    ),
                    "digit_positions_matched_target": sum(
                        1
                        for p in intervention_record["results_by_position"]
                        if p["digit_position"] >= 0
                        and p["intervened"]["predicted_digit"] == target_digit
                    ),
                    "total_digit_positions": sum(
                        1
                        for p in intervention_record["results_by_position"]
                        if p["digit_position"] >= 0
                    ),
                }

                all_interventions.append(intervention_record)

    # Compute success rates
    position_digit_rates = {}
    for pos in range(max_positions):
        position_digit_rates[pos] = {}
        for digit in range(10):
            attempts = position_digit_attempts[pos][digit]
            if attempts > 0:
                success = position_digit_success[pos][digit]
                position_digit_rates[pos][digit] = success / attempts
            else:
                position_digit_rates[pos][digit] = 0.0

    return position_digit_rates, all_interventions


def visualize_position_digit_mapping(position_digit_rates, output_path):
    """Create heatmap of position-digit mapping strength."""
    # Find max position with data
    max_pos = max(position_digit_rates.keys()) + 1

    # Create matrix for visualization
    matrix = np.zeros((max_pos, 10))

    for pos in range(max_pos):
        for digit in range(10):
            if pos in position_digit_rates and digit in position_digit_rates[pos]:
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

    plt.title(
        "Position-Digit Mapping: Success Rate of Forced Key Interventions", fontsize=14
    )
    plt.xlabel("Target Digit", fontsize=12)
    plt.ylabel("Position in Answer", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Intervene on memory keys to measure position-digit mapping"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to tokenizer"
    )
    parser.add_argument(
        "--activation_file",
        type=str,
        required=True,
        help="JSON file with memory key activations",
    )
    parser.add_argument(
        "--dataset_size",
        type=str,
        choices=["200", "2K", "10K", "50K"],
        required=True,
        help="Dataset size to use",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of random QA pairs to test",
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

    # Load activated keys
    print(f"\nLoading activated keys from {args.activation_file}...")
    all_keys, key_to_digit = load_activated_keys(args.activation_file)

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
            qa_pairs.append((key, value))

    print(f"Loaded {len(qa_pairs)} QA pairs")

    # Analyze position-digit mapping
    position_digit_rates, all_interventions = analyze_position_digit_mapping(
        model,
        tokenizer,
        qa_pairs,
        all_keys,
        key_to_digit,
        num_samples=args.num_samples,
        device=args.device,
    )

    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Visualize results
    visualize_position_digit_mapping(position_digit_rates, output_path)

    # Compute additional statistics
    intervention_stats = {
        "total_interventions": len(all_interventions),
        "avg_positions_changed_per_intervention": np.mean(
            [i["summary"]["num_positions_changed"] for i in all_interventions]
        )
        if all_interventions
        else 0,
        "interventions_by_digit": {},
    }

    # Stats by target digit
    for digit in range(10):
        digit_interventions = [
            i for i in all_interventions if i["forced_key"]["associated_digit"] == digit
        ]
        if digit_interventions:
            intervention_stats["interventions_by_digit"][str(digit)] = {
                "count": len(digit_interventions),
                "avg_positions_changed": np.mean(
                    [i["summary"]["num_positions_changed"] for i in digit_interventions]
                ),
                "avg_digit_positions_matched": np.mean(
                    [
                        i["summary"]["digit_positions_matched_target"]
                        for i in digit_interventions
                    ]
                ),
                "unique_keys_used": len(
                    set(
                        (i["forced_key"]["idx1"], i["forced_key"]["idx2"])
                        for i in digit_interventions
                    )
                ),
            }

    # Save comprehensive results
    results_data = {
        "config": {
            "model_path": args.model_path,
            "activation_file": args.activation_file,
            "dataset_size": args.dataset_size,
            "num_samples": args.num_samples,
            "seed": args.seed,
        },
        "summary": {
            "num_unique_keys": len(all_keys),
            "keys_per_digit": {
                str(d): len([k for k, v in key_to_digit.items() if v == d])
                for d in range(10)
            },
            "position_digit_success_rates": position_digit_rates,
            "intervention_statistics": intervention_stats,
        },
        "detailed_interventions": all_interventions,
    }

    # Save main comprehensive JSON
    json_path = output_path.parent / f"{output_path.stem}_comprehensive.json"
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    # Also save a lighter summary version without full intervention details
    summary_data = {
        "config": results_data["config"],
        "summary": results_data["summary"],
    }

    summary_json_path = output_path.parent / f"{output_path.stem}_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved comprehensive data to {json_path}")
    print(f"Saved summary data to {summary_json_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Total unique keys analyzed: {len(all_keys)}")
    print(f"Total interventions performed: {len(all_interventions)}")
    print(
        f"Average positions changed per intervention: {intervention_stats['avg_positions_changed_per_intervention']:.2f}"
    )
    print(f"\nKeys per digit: {results_data['summary']['keys_per_digit']}")

    # Find strongest position-digit mappings
    print("\nStrongest position-digit mappings (>80% success rate):")
    for pos in sorted(position_digit_rates.keys()):
        for digit in range(10):
            rate = position_digit_rates[pos][digit]
            if rate > 0.8:
                print(f"  Position {pos} -> Digit {digit}: {rate * 100:.1f}%")


if __name__ == "__main__":
    main()
