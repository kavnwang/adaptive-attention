#!/usr/bin/env python3
"""
Analyze memory swapping effects as a matrix showing:
- X-axis: digit of QA1 at position
- Y-axis: digit of QA2 at position
- Cell value: most likely generated digit
- Cell shading: probability of that digit
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


def forward_with_swapped_keys(
    model,
    tokenizer,
    qa1,
    qa2,
    swap_mapping_by_layer,
    device="cuda",
    use_teacher_forcing=False,
):
    """
    Perform forward pass with swapped keys and return predictions.
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
    predictions = []

    current_ids = inputs1["input_ids"]
    with torch.no_grad():
        for pos in range(len(answer_token_ids1)):
            outputs = model(input_ids=current_ids)
            logits = outputs.logits[0, -1, :]

            # Get probability distribution
            probs = torch.softmax(logits.float(), dim=-1)

            # Get predicted token and its probability
            predicted_token = logits.argmax().item()
            predicted_prob = probs[predicted_token].item()
            predicted_text = tokenizer.decode(
                [predicted_token], skip_special_tokens=True
            )

            predictions.append(
                {
                    "token": predicted_token,
                    "text": predicted_text,
                    "prob": predicted_prob,
                }
            )

            # Continue generation
            if use_teacher_forcing:
                next_token = answer_token_ids1[pos].item()
            else:
                next_token = predicted_token

            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token]], device=device)], dim=1
            )

    # Restore original methods
    for memory_retrieval, original_method in original_methods:
        memory_retrieval.forward = original_method

    return predictions


def analyze_swap_matrix(
    model, tokenizer, qa_pairs, num_swaps=1000, device="cuda", use_teacher_forcing=False
):
    """
    Create matrix analysis of swap effects.
    For each position, create a 10x10 matrix where:
    - X-axis: digit of QA1 at that position
    - Y-axis: digit of QA2 at that position
    - Cell: most likely generated digit and its probability
    """
    # Extract all QA pairs with their digit patterns
    qa_with_digits = []
    for qa in qa_pairs:
        question, answer = qa
        digits = []
        for char in answer:
            if char.isdigit():
                digits.append(int(char))
        if digits:
            qa_with_digits.append((qa, digits))

    if not qa_with_digits:
        print("No QA pairs with digit answers found")
        return None

    # Determine max answer length
    max_length = max(len(digits) for _, digits in qa_with_digits)

    # Initialize result storage
    # For each position: matrix[qa1_digit][qa2_digit] -> list of (predicted_digit, prob)
    position_results = {}
    for pos in range(max_length):
        position_results[pos] = defaultdict(lambda: defaultdict(list))

    print(f"Performing {num_swaps} swap experiments...")

    for _ in tqdm(range(num_swaps)):
        # Select two random QA pairs
        if len(qa_with_digits) < 2:
            continue

        (qa1, digits1), (qa2, digits2) = random.sample(qa_with_digits, 2)

        # Extract activations
        qa1_activations, _ = extract_activations_for_qa(
            model, tokenizer, qa1[0], qa1[1], device
        )
        qa2_activations, _ = extract_activations_for_qa(
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
        predictions = forward_with_swapped_keys(
            model,
            tokenizer,
            qa1,
            qa2,
            swap_mapping_by_layer,
            device,
            use_teacher_forcing,
        )

        # Analyze predictions by position
        digit_position = 0
        for pred_idx, pred in enumerate(predictions):
            if pred["text"] and pred["text"][0].isdigit():
                if digit_position < len(digits1) and digit_position < len(digits2):
                    qa1_digit = digits1[digit_position]
                    qa2_digit = digits2[digit_position]
                    predicted_digit = int(pred["text"][0])
                    prob = pred["prob"]

                    position_results[digit_position][qa1_digit][qa2_digit].append(
                        (predicted_digit, prob)
                    )

                digit_position += 1

    # Compute final statistics
    final_results = {}
    for pos in range(max_length):
        final_results[pos] = {
            "most_likely": np.full((10, 10), -1, dtype=int),
            "probability": np.zeros((10, 10)),
            "counts": np.zeros((10, 10), dtype=int),
            "raw_predictions": position_results[pos],
        }

        for qa1_digit in range(10):
            for qa2_digit in range(10):
                predictions = position_results[pos][qa1_digit][qa2_digit]
                if predictions:
                    # Count occurrences of each predicted digit
                    digit_counts = defaultdict(float)
                    for pred_digit, prob in predictions:
                        digit_counts[pred_digit] += prob

                    # Find most likely digit
                    most_likely = max(digit_counts.items(), key=lambda x: x[1])
                    most_likely_digit = most_likely[0]

                    # Average probability when predicting this digit
                    avg_prob = np.mean(
                        [prob for d, prob in predictions if d == most_likely_digit]
                    )

                    final_results[pos]["most_likely"][qa1_digit, qa2_digit] = (
                        most_likely_digit
                    )
                    final_results[pos]["probability"][qa1_digit, qa2_digit] = avg_prob
                    final_results[pos]["counts"][qa1_digit, qa2_digit] = len(
                        predictions
                    )

    return final_results, max_length


def visualize_swap_matrix(results, max_length, output_path, title_suffix=""):
    """Create matrix visualizations for each position."""
    # Create figure with subplots for each position
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for pos in range(min(max_length, 6)):  # Show up to 6 positions
        ax = axes[pos]

        # Create visualization data
        matrix_data = results[pos]["most_likely"].copy()
        prob_data = results[pos]["probability"].copy()
        counts = results[pos]["counts"]

        # Mask cells with no data
        mask = counts == 0

        # Create custom annotation
        annot = np.empty_like(matrix_data, dtype=object)
        for i in range(10):
            for j in range(10):
                if not mask[i, j]:
                    digit = matrix_data[i, j]
                    prob = prob_data[i, j]
                    annot[i, j] = f"{digit}\n{prob:.2f}"
                else:
                    annot[i, j] = ""

        # Create heatmap
        sns.heatmap(
            prob_data,
            ax=ax,
            mask=mask,
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1,
            annot=annot,
            fmt="",
            cbar_kws={"label": "Probability"},
            xticklabels=range(10),
            yticklabels=range(10),
            square=True,
            linewidths=0.5,
            linecolor="gray",
        )

        ax.set_title(f"Position {pos}", fontsize=14)
        ax.set_xlabel("QA1 Digit", fontsize=12)
        ax.set_ylabel("QA2 Digit", fontsize=12)

    # Hide unused subplots
    for pos in range(max_length, 6):
        axes[pos].set_visible(False)

    plt.suptitle(
        f"Memory Swap Effects: Generated Digit by QA1/QA2 Digit Combination{title_suffix}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze memory swap effects as interaction matrix"
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
        default=1000,
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
        parts = text.split("? ")
        if len(parts) == 2:
            question = parts[0] + "?"
            answer = parts[1].strip()
            qa_pairs.append((question, answer))

    print(f"Loaded {len(qa_pairs)} QA pairs")

    # Run analysis with both modes
    for use_teacher_forcing in [False, True]:
        mode_name = "teacher_forcing" if use_teacher_forcing else "autoregressive"
        print(f"\n{'=' * 60}")
        print(f"Running analysis with {mode_name} mode")
        print(f"{'=' * 60}")

        # Analyze swap effects
        results, max_length = analyze_swap_matrix(
            model,
            tokenizer,
            qa_pairs,
            num_swaps=args.num_swaps,
            device=args.device,
            use_teacher_forcing=use_teacher_forcing,
        )

        if results is None:
            print(f"No valid QA pairs found for analysis in {mode_name} mode")
            continue

        # Create output directory
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Modify output filename to include mode
        mode_output_path = output_path.parent / f"{output_path.stem}_{mode_name}.png"

        # Visualize results
        title_suffix = f" ({mode_name.replace('_', ' ').title()})"
        visualize_swap_matrix(results, max_length, mode_output_path, title_suffix)

        # Save raw results
        json_data = {
            "config": {
                "model_path": args.model_path,
                "dataset_size": args.dataset_size,
                "num_swaps": args.num_swaps,
                "mode": mode_name,
                "seed": args.seed,
            },
            "results": {},
        }

        for pos in range(max_length):
            json_data["results"][f"position_{pos}"] = {
                "most_likely": results[pos]["most_likely"].tolist(),
                "probability": results[pos]["probability"].tolist(),
                "counts": results[pos]["counts"].tolist(),
            }

        json_path = output_path.parent / f"{output_path.stem}_{mode_name}_data.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"Saved raw data to {json_path}")


if __name__ == "__main__":
    main()
