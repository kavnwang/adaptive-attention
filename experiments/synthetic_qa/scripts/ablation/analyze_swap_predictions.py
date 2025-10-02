#!/usr/bin/env python3
"""
Enhanced swapping analysis that shows, for each (digit, position) pair in QA1,
after values are swapped, what is the maximally likely generated next digit
and its probability, displayed as heatmaps.
"""

import argparse
import json
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import sys
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add bento path for custom model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../3rdparty/bento"))
from bento.models.memory.modeling_memory import MemoryForCausalLM, MemoryLayer


def extract_activations_for_qa(model, tokenizer, question, answer, device="cuda"):
    """Extract memory activations for a single QA pair."""
    inputs = tokenizer(question, return_tensors="pt").to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
    answer_token_ids = answer_tokens["input_ids"][0].to(device)

    activations = {}
    original_methods = []

    # Hook into memory layers
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, MemoryLayer):
            memory_retrieval = layer.mlp.memory_retrieval
            original_forward = memory_retrieval.forward

            def make_tracked_forward(layer_idx):
                def tracked_forward(query, knn, return_individual_probs=False):
                    scores, indices, probs1, probs2 = original_forward(
                        query, knn, return_individual_probs=True
                    )

                    if layer_idx not in activations:
                        activations[layer_idx] = []

                    # Store activation info
                    n_keys = memory_retrieval.mem_n_keys
                    bs_times_heads = indices.shape[0]
                    heads = memory_retrieval.heads
                    bs = bs_times_heads // heads

                    if bs > 0:
                        for head_idx in range(heads):
                            pos_idx = (bs - 1) * heads + head_idx
                            selected_indices = indices[pos_idx]
                            selected_scores = scores[pos_idx]

                            if selected_indices.dim() == 0:
                                selected_indices = selected_indices.unsqueeze(0)
                                selected_scores = selected_scores.unsqueeze(0)

                            for k in range(len(selected_indices)):
                                combined_idx = selected_indices[k].item()
                                activations[layer_idx].append(
                                    {
                                        "combined_idx": combined_idx,
                                        "score": selected_scores[k].item(),
                                        "head": head_idx,
                                    }
                                )

                    if return_individual_probs:
                        return scores, indices, probs1, probs2
                    else:
                        return scores, indices

                return tracked_forward

            memory_retrieval.forward = make_tracked_forward(i)
            original_methods.append((memory_retrieval, original_forward))

    # Forward pass with answer tokens
    current_ids = inputs["input_ids"]
    with torch.no_grad():
        for i in range(len(answer_token_ids)):
            outputs = model(input_ids=current_ids)
            current_ids = torch.cat(
                [current_ids, answer_token_ids[i : i + 1].unsqueeze(0)], dim=1
            )

    # Restore original methods
    for memory_retrieval, original_method in original_methods:
        memory_retrieval.forward = original_method

    return activations, answer_token_ids.tolist()


def forward_with_swapped_keys_detailed(
    model,
    tokenizer,
    qa1,
    qa2,
    swap_mapping_by_layer,
    device="cuda",
    use_teacher_forcing=False,
):
    """
    Perform forward pass with swapped keys and return detailed probability distributions.

    Args:
        use_teacher_forcing: If True, use true tokens for conditioning. If False, use predicted tokens.
    """
    question1, answer1 = qa1
    inputs1 = tokenizer(question1, return_tensors="pt").to(device)
    answer_tokens1 = tokenizer(answer1, add_special_tokens=False, return_tensors="pt")
    answer_token_ids1 = answer_tokens1["input_ids"][0].to(device)

    # Patch memory layers for swapping
    original_methods = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, MemoryLayer):
            if i in swap_mapping_by_layer:
                memory_retrieval = layer.mlp.memory_retrieval
                original_forward = memory_retrieval.forward

                def make_swapped_forward(layer_idx, mapping):
                    def swapped_forward(query, knn, return_individual_probs=False):
                        scores, indices, probs1, probs2 = original_forward(
                            query, knn, return_individual_probs=True
                        )

                        # Apply swapping
                        indices_swapped = indices.clone()
                        for i in range(indices.shape[0]):
                            for j in range(
                                indices.shape[1] if indices.dim() > 1 else 1
                            ):
                                idx = (
                                    indices[i, j].item()
                                    if indices.dim() > 1
                                    else indices[i].item()
                                )
                                if idx in mapping:
                                    if indices.dim() == 1:
                                        indices_swapped[i] = mapping[idx]
                                    else:
                                        indices_swapped[i, j] = mapping[idx]

                        if return_individual_probs:
                            return scores, indices_swapped, probs1, probs2
                        else:
                            return scores, indices_swapped

                    return swapped_forward

                memory_retrieval.forward = make_swapped_forward(
                    i, swap_mapping_by_layer[i]
                )
                original_methods.append((memory_retrieval, original_forward))

    # Generate with detailed probability tracking
    results = {
        "generated_tokens": [],
        "probability_distributions": [],
        "top_k_predictions": [],
    }

    current_ids = inputs1["input_ids"]
    with torch.no_grad():
        for pos in range(len(answer_token_ids1)):
            outputs = model(input_ids=current_ids)
            logits = outputs.logits[0, -1, :]

            # Get probability distribution
            probs = torch.softmax(
                logits.float(), dim=-1
            )  # Convert to float32 for softmax

            # Get top-k predictions
            top_k = 10
            top_probs, top_indices = torch.topk(probs, k=top_k)

            # Get predicted token
            predicted_token = logits.argmax().item()
            predicted_prob = probs[predicted_token].item()

            results["generated_tokens"].append(predicted_token)
            results["probability_distributions"].append(
                probs.cpu().float().numpy()
            )  # Ensure float32
            results["top_k_predictions"].append(
                {
                    "tokens": top_indices.tolist(),
                    "probs": top_probs.tolist(),
                    "predicted_token": predicted_token,
                    "predicted_prob": predicted_prob,
                }
            )

            # Continue generation
            if use_teacher_forcing:
                # Use true token for next prediction
                next_token = answer_token_ids1[pos].item()
            else:
                # Use predicted token for next prediction
                next_token = predicted_token

            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token]], device=device)], dim=1
            )

    # Restore original methods
    for memory_retrieval, original_method in original_methods:
        memory_retrieval.forward = original_method

    return results


def analyze_swap_effects_by_digit_position(
    model, tokenizer, qa_pairs, num_swaps=50, device="cuda", use_teacher_forcing=False
):
    """
    Analyze swap effects for different digit values at each position.
    Returns matrices showing the most likely generated digit and its probability
    for each (original_digit, position) pair.

    Args:
        use_teacher_forcing: Whether to use teacher forcing during generation
    """
    # First, group QA pairs by their answer patterns
    qa_by_answer_pattern = defaultdict(list)

    for qa in qa_pairs:
        question, answer = qa
        # Extract digits from answer
        answer_digits = []
        for char in answer:
            if char.isdigit():
                answer_digits.append(int(char))

        if answer_digits:
            # Use tuple of digits as key
            pattern_key = tuple(answer_digits)
            qa_by_answer_pattern[pattern_key].append(qa)

    # Find the most common answer length
    answer_lengths = [len(pattern) for pattern in qa_by_answer_pattern.keys()]
    if not answer_lengths:
        print("No QA pairs with digit answers found")
        return None

    max_length = max(answer_lengths)

    # Initialize result matrices
    # For each position, track what happens when we swap from digit X
    position_results = {}

    for pos in range(max_length):
        position_results[pos] = {
            "most_likely_digit": np.full((10,), -1, dtype=int),  # -1 means no data
            "probability": np.zeros((10,)),
            "sample_counts": np.zeros((10,), dtype=int),
            "all_predictions": defaultdict(
                list
            ),  # digit -> list of (predicted_digit, prob)
        }

    # Perform swaps and analyze
    print(f"Performing {num_swaps} swap experiments...")

    for swap_idx in tqdm(range(num_swaps)):
        # Select two random QA pairs with digit answers
        valid_patterns = [p for p in qa_by_answer_pattern.keys() if len(p) > 0]
        if len(valid_patterns) < 2:
            continue

        # Select two different patterns
        pattern1, pattern2 = random.sample(valid_patterns, 2)
        qa1 = random.choice(qa_by_answer_pattern[pattern1])
        qa2 = random.choice(qa_by_answer_pattern[pattern2])

        # Extract activations
        qa1_activations, qa1_answer_tokens = extract_activations_for_qa(
            model, tokenizer, qa1[0], qa1[1], device
        )
        qa2_activations, qa2_answer_tokens = extract_activations_for_qa(
            model, tokenizer, qa2[0], qa2[1], device
        )

        # Build swap mapping
        swap_mapping_by_layer = defaultdict(dict)
        for layer_idx in qa1_activations:
            qa1_keys = {act["combined_idx"] for act in qa1_activations[layer_idx]}
            qa2_keys = {act["combined_idx"] for act in qa2_activations[layer_idx]}

            qa1_list = sorted(list(qa1_keys))
            qa2_list = sorted(list(qa2_keys))

            # Create bidirectional mapping
            for i, key1 in enumerate(qa1_list):
                if qa2_list:
                    key2 = qa2_list[i % len(qa2_list)]
                    swap_mapping_by_layer[layer_idx][key1] = key2

        # Perform swapped forward pass
        swap_results = forward_with_swapped_keys_detailed(
            model,
            tokenizer,
            qa1,
            qa2,
            swap_mapping_by_layer,
            device,
            use_teacher_forcing,
        )

        # Analyze results by position and original digit
        qa1_digits = list(pattern1)

        # First, we need to know which positions in the FULL answer contain digits
        # Tokenize the answer to understand the structure
        answer_tokens = tokenizer(qa1[1], add_special_tokens=False, return_tensors="pt")
        answer_token_ids = answer_tokens["input_ids"][0]

        # Map token positions to digit positions
        token_to_digit_pos = {}
        digit_pos = 0
        for token_pos, token_id in enumerate(answer_token_ids):
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            if token_text and token_text[0].isdigit():
                token_to_digit_pos[token_pos] = digit_pos
                digit_pos += 1

        # Now analyze predictions
        for pred_pos, prediction_info in enumerate(swap_results["top_k_predictions"]):
            # Check if this token position should contain a digit
            if pred_pos in token_to_digit_pos:
                digit_position = token_to_digit_pos[pred_pos]
                if digit_position < len(qa1_digits):
                    original_digit = qa1_digits[digit_position]

                    # Get the predicted digit
                    predicted_token = prediction_info["predicted_token"]
                    predicted_text = tokenizer.decode(
                        [predicted_token], skip_special_tokens=True
                    )

                    if predicted_text and predicted_text[0].isdigit():
                        predicted_digit = int(predicted_text[0])
                        predicted_prob = prediction_info["predicted_prob"]

                        # Update statistics
                        position_results[digit_position]["all_predictions"][
                            original_digit
                        ].append((predicted_digit, predicted_prob))
                        position_results[digit_position]["sample_counts"][
                            original_digit
                        ] += 1

    # Compute final statistics
    for pos in range(max_length):
        for digit in range(10):
            predictions = position_results[pos]["all_predictions"][digit]
            if predictions:
                # Find most common predicted digit
                digit_counts = defaultdict(float)
                for pred_digit, prob in predictions:
                    digit_counts[pred_digit] += prob

                # Get digit with highest total probability
                most_likely = max(digit_counts.items(), key=lambda x: x[1])
                most_likely_digit = most_likely[0]

                # Average probability when predicting this digit
                avg_prob = np.mean(
                    [prob for d, prob in predictions if d == most_likely_digit]
                )

                position_results[pos]["most_likely_digit"][digit] = most_likely_digit
                position_results[pos]["probability"][digit] = avg_prob

    return position_results, max_length


def visualize_swap_predictions(
    position_results, max_length, output_path, title_suffix=""
):
    """Create heatmap visualizations of swap predictions."""
    # Create figure with subplots for each position
    fig, axes = plt.subplots(1, max_length, figsize=(5 * max_length, 6))
    if max_length == 1:
        axes = [axes]

    for pos in range(max_length):
        ax = axes[pos]

        # Create matrices for visualization
        prediction_matrix = np.full(
            (10, 10), np.nan
        )  # rows: original digit, cols: predicted digit
        probability_matrix = np.zeros((10, 10))

        for original_digit in range(10):
            if position_results[pos]["sample_counts"][original_digit] > 0:
                predicted_digit = position_results[pos]["most_likely_digit"][
                    original_digit
                ]
                if predicted_digit >= 0:
                    prediction_matrix[original_digit, predicted_digit] = 1
                    probability_matrix[original_digit, predicted_digit] = (
                        position_results[pos]["probability"][original_digit]
                    )

        # Create heatmap with probabilities as values
        mask = np.isnan(prediction_matrix)

        # Use probability values for color intensity
        sns.heatmap(
            probability_matrix,
            ax=ax,
            mask=mask,
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Probability"},
            xticklabels=range(10),
            yticklabels=range(10),
            square=True,
        )

        # Add diagonal line to show identity mapping
        ax.plot([0, 10], [0, 10], "k--", alpha=0.3, linewidth=1)

        ax.set_title(f"Position {pos}", fontsize=14)
        ax.set_xlabel("Predicted Digit", fontsize=12)
        if pos == 0:
            ax.set_ylabel("Original Digit in QA1", fontsize=12)

    plt.suptitle(
        f"Swap Predictions: Most Likely Digit After Memory Value Swapping{title_suffix}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Create a second visualization showing prediction confidence
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Average probability across all positions
    avg_probs = np.zeros((10, 10))
    counts = np.zeros((10, 10))

    for pos in range(max_length):
        for original_digit in range(10):
            if position_results[pos]["sample_counts"][original_digit] > 0:
                predicted_digit = position_results[pos]["most_likely_digit"][
                    original_digit
                ]
                if predicted_digit >= 0:
                    avg_probs[original_digit, predicted_digit] += position_results[pos][
                        "probability"
                    ][original_digit]
                    counts[original_digit, predicted_digit] += 1

    # Compute average
    mask = counts == 0
    avg_probs[~mask] /= counts[~mask]

    sns.heatmap(
        avg_probs,
        ax=ax,
        mask=mask,
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Average Probability"},
        xticklabels=range(10),
        yticklabels=range(10),
        square=True,
    )

    ax.plot([0, 10], [0, 10], "k--", alpha=0.3, linewidth=1)
    ax.set_title(
        f"Average Swap Predictions Across All Positions{title_suffix}", fontsize=14
    )
    ax.set_xlabel("Predicted Digit", fontsize=12)
    ax.set_ylabel("Original Digit in QA1", fontsize=12)

    plt.tight_layout()
    avg_output_path = output_path.parent / f"{output_path.stem}_average.png"
    plt.savefig(avg_output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved visualizations to {output_path} and {avg_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze swap predictions with detailed probability heatmaps"
    )
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
        "--num_swaps",
        type=int,
        default=100,
        help="Number of swap experiments to perform",
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

    print(f"Loading dataset (size: {args.dataset_size})...")
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

    # Debug: Print some example QA pairs and their digit patterns
    print("\nExample QA pairs:")
    for i, (q, a) in enumerate(qa_pairs[:5]):
        digits = [c for c in a if c.isdigit()]
        print(f"  {i}: Q='{q}' A='{a}' Digits={digits}")

    # Run analysis with both teacher forcing modes
    results_by_mode = {}

    for use_teacher_forcing in [False, True]:
        mode_name = "teacher_forcing" if use_teacher_forcing else "autoregressive"
        print(f"\n{'=' * 60}")
        print(f"Running analysis with {mode_name} mode")
        print(f"{'=' * 60}")

        # Analyze swap effects
        position_results, max_length = analyze_swap_effects_by_digit_position(
            model,
            tokenizer,
            qa_pairs,
            num_swaps=args.num_swaps,
            device=args.device,
            use_teacher_forcing=use_teacher_forcing,
        )

        if position_results is None:
            print(f"No valid QA pairs found for analysis in {mode_name} mode")
            continue

        results_by_mode[mode_name] = (position_results, max_length)

        # Create output directory
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Modify output filename to include mode
        mode_output_path = output_path.parent / f"{output_path.stem}_{mode_name}.png"

        # Visualize results
        title_suffix = f" ({mode_name.replace('_', ' ').title()})"
        visualize_swap_predictions(
            position_results, max_length, mode_output_path, title_suffix
        )

        # Print detailed statistics for debugging
        print(f"\nDetailed statistics for {mode_name}:")
        for pos in range(max_length):
            print(f"\nPosition {pos}:")
            total_samples = np.sum(position_results[pos]["sample_counts"])
            print(f"  Total samples: {int(total_samples)}")

            # Show which digits have data
            for digit in range(10):
                count = position_results[pos]["sample_counts"][digit]
                if count > 0:
                    predicted = position_results[pos]["most_likely_digit"][digit]
                    prob = position_results[pos]["probability"][digit]
                    print(
                        f"    Digit {digit}: {int(count)} samples, most likely → {predicted} (prob: {prob:.3f})"
                    )

    # Save raw results for both modes
    all_results_data = {
        "config": {
            "model_path": args.model_path,
            "dataset_size": args.dataset_size,
            "num_swaps": args.num_swaps,
            "seed": args.seed,
        },
        "results_by_mode": {},
    }

    for mode_name, (position_results, max_length) in results_by_mode.items():
        all_results_data["results_by_mode"][mode_name] = {
            "position_results": {
                str(pos): {
                    "most_likely_digit": position_results[pos][
                        "most_likely_digit"
                    ].tolist(),
                    "probability": position_results[pos]["probability"].tolist(),
                    "sample_counts": position_results[pos]["sample_counts"].tolist(),
                }
                for pos in range(max_length)
            },
            "max_answer_length": max_length,
        }

    json_path = output_path.parent / f"{output_path.stem}_data.json"
    with open(json_path, "w") as f:
        json.dump(all_results_data, f, indent=2)

    print(f"Saved raw data to {json_path}")

    # Print summary comparison
    print("\n=== Summary Comparison ===")
    print(f"Analyzed {args.num_swaps} swap experiments")

    if results_by_mode:
        for mode_name, (position_results, max_length) in results_by_mode.items():
            print(f"\n{mode_name.upper()} MODE:")
            print(f"Maximum answer length: {max_length} positions")

            # Count strong substitutions
            strong_substitutions = 0
            for pos in range(max_length):
                for digit in range(10):
                    if position_results[pos]["sample_counts"][digit] > 0:
                        predicted = position_results[pos]["most_likely_digit"][digit]
                        prob = position_results[pos]["probability"][digit]
                        if predicted != digit and prob > 0.5:
                            strong_substitutions += 1

            print(f"Strong substitutions (>50% probability): {strong_substitutions}")


if __name__ == "__main__":
    main()
