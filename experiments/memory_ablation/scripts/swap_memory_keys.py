#!/usr/bin/env python3
"""
Swap memory keys between QA pairs to measure intervention effects.

For num_samples iterations:
1. Randomly select 2 QA pairs from the dataset
2. Extract memory key activations for each pair
3. Swap the memory values between the pairs for each key
4. Generate answers and track which tokens swap
5. Calculate percentage of answer tokens that swapped
"""

import argparse
import json
import torch
import random
from pathlib import Path
from tqdm import tqdm
from transformers impoyou srt AutoTokenizer
from datasets import load_dataset
import sys
import os
from collections import defaultdict

# Add bento path for custom model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "3rdparty/bento"))
from bento.models.memory.modeling_memory import MemoryForCausalLM, MemoryLayer


def extract_dual_activations(memory_layer, layer_idx, qa1_activations, qa2_activations, current_qa):
    """Patch a MemoryLayer to track activations for both QA pairs."""
    memory_retrieval = memory_layer.memory_retrieval
    original_forward = memory_retrieval.forward

    def tracked_forward(query, knn, return_individual_probs=False):
        # Call original with return_individual_probs=True to get full info
        scores, indices, probs1, probs2 = original_forward(
            query, knn, return_individual_probs=True
        )

        # Extract information about selected keys
        n_keys = memory_retrieval.mem_n_keys
        bs_times_heads = indices.shape[0]
        heads = memory_retrieval.heads
        bs = bs_times_heads // heads

        # Determine which QA pair we're processing
        target_dict = qa1_activations if current_qa == 1 else qa2_activations

        # Store activations for the last position (answer token)
        if layer_idx not in target_dict:
            target_dict[layer_idx] = []

        # Process only the last position
        if bs > 0:
            for head_idx in range(heads):
                # Get indices for last position, this head
                pos_idx = (bs - 1) * heads + head_idx
                selected_indices = indices[pos_idx]  # shape: (knn,)
                selected_scores = scores[pos_idx]  # shape: (knn,)

                # Decode combined indices to get idx1 and idx2
                if selected_indices.dim() == 0:  # knn=1 case
                    selected_indices = selected_indices.unsqueeze(0)
                    selected_scores = selected_scores.unsqueeze(0)

                for k in range(len(selected_indices)):
                    combined_idx = selected_indices[k].item()
                    idx1 = combined_idx // n_keys
                    idx2 = combined_idx % n_keys
                    score = selected_scores[k].item()

                    target_dict[layer_idx].append({
                        "idx1": idx1,
                        "idx2": idx2,
                        "score": score,
                        "position": len(target_dict[layer_idx]) // heads,
                        "head": head_idx,
                        "combined_idx": combined_idx,
                    })

        # Return the original result format
        if return_individual_probs:
            return scores, indices, probs1, probs2
        else:
            return scores, indices

    # Replace the method
    memory_retrieval.forward = tracked_forward
    return original_forward


def patch_memory_layer_for_swapping(
    memory_layer,
    layer_idx,
    swap_mapping,
    activations_dict,
):
    """Patch a MemoryLayer to swap memory values between QA pairs.

    Args:
        memory_layer: The MemoryLayer to patch
        layer_idx: Index of the layer
        swap_mapping: Dict mapping combined indices from QA1 to QA2 and vice versa
        activations_dict: Dictionary to store activations during swapped generation
    """
    memory_retrieval = memory_layer.memory_retrieval
    original_forward = memory_retrieval.forward

    def swapped_forward(query, knn, return_individual_probs=False):
        # Call original with return_individual_probs=True
        scores, indices, probs1, probs2 = original_forward(
            query, knn, return_individual_probs=True
        )

        # Apply swapping: replace indices according to swap_mapping
        indices_swapped = indices.clone()
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1] if indices.dim() > 1 else 1):
                if indices.dim() == 1:
                    idx = indices[i].item()
                else:
                    idx = indices[i, j].item()

                if idx in swap_mapping:
                    if indices.dim() == 1:
                        indices_swapped[i] = swap_mapping[idx]
                    else:
                        indices_swapped[i, j] = swap_mapping[idx]

        # Track what we're doing
        n_keys = memory_retrieval.mem_n_keys
        bs_times_heads = indices.shape[0]
        heads = memory_retrieval.heads
        bs = bs_times_heads // heads

        if layer_idx not in activations_dict:
            activations_dict[layer_idx] = []

        # Process only the last position for tracking
        if bs > 0:
            for head_idx in range(heads):
                pos_idx = (bs - 1) * heads + head_idx
                original_indices = indices[pos_idx]
                swapped_indices = indices_swapped[pos_idx]
                selected_scores = scores[pos_idx]

                if original_indices.dim() == 0:
                    original_indices = original_indices.unsqueeze(0)
                    swapped_indices = swapped_indices.unsqueeze(0)
                    selected_scores = selected_scores.unsqueeze(0)

                for k in range(len(original_indices)):
                    orig_idx = original_indices[k].item()
                    swap_idx = swapped_indices[k].item()

                    # Decode swapped index
                    idx1 = swap_idx // n_keys
                    idx2 = swap_idx % n_keys
                    score = selected_scores[k].item()

                    activations_dict[layer_idx].append({
                        "idx1": idx1,
                        "idx2": idx2,
                        "score": score,
                        "position": len(activations_dict[layer_idx]) // heads,
                        "head": head_idx,
                        "original_combined_idx": orig_idx,
                        "swapped_combined_idx": swap_idx,
                        "was_swapped": orig_idx != swap_idx,
                    })

        # Return with swapped indices
        if return_individual_probs:
            return scores, indices_swapped, probs1, probs2
        else:
            return scores, indices_swapped

    # Replace the method
    memory_retrieval.forward = swapped_forward
    return original_forward


def clean_forward_passes_dual(model, tokenizer, qa1, qa2, device="cuda"):
    """
    Do clean forward passes for two QA pairs to extract their key activations.
    Returns activations and answer token IDs for both pairs.
    """
    question1, answer1 = qa1
    question2, answer2 = qa2

    # Tokenize both QA pairs
    inputs1 = tokenizer(question1, return_tensors="pt").to(device)
    inputs2 = tokenizer(question2, return_tensors="pt").to(device)

    answer_tokens1 = tokenizer(answer1, add_special_tokens=False, return_tensors="pt")
    answer_tokens2 = tokenizer(answer2, add_special_tokens=False, return_tensors="pt")

    answer_token_ids1 = answer_tokens1["input_ids"][0].to(device)
    answer_token_ids2 = answer_tokens2["input_ids"][0].to(device)

    # Dictionaries to store activations
    qa1_activations = {}
    qa2_activations = {}

    # First, process QA1
    original_methods = []
    found_memory_layers = []

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, MemoryLayer):
            found_memory_layers.append(i)
            original_method = extract_dual_activations(
                layer.mlp, i, qa1_activations, qa2_activations, current_qa=1
            )
            original_methods.append((layer.mlp, original_method))

    # Forward pass for QA1
    current_ids = inputs1["input_ids"]
    with torch.no_grad():
        for i in range(len(answer_token_ids1)):
            outputs = model(input_ids=current_ids)
            current_ids = torch.cat(
                [current_ids, answer_token_ids1[i : i + 1].unsqueeze(0)], dim=1
            )

    # Restore original methods
    for memory_layer, original_method in original_methods:
        memory_layer.memory_retrieval.forward = original_method

    # Now process QA2
    original_methods = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, MemoryLayer):
            original_method = extract_dual_activations(
                layer.mlp, i, qa1_activations, qa2_activations, current_qa=2
            )
            original_methods.append((layer.mlp, original_method))

    # Forward pass for QA2
    current_ids = inputs2["input_ids"]
    with torch.no_grad():
        for i in range(len(answer_token_ids2)):
            outputs = model(input_ids=current_ids)
            current_ids = torch.cat(
                [current_ids, answer_token_ids2[i : i + 1].unsqueeze(0)], dim=1
            )

    # Restore original methods
    for memory_layer, original_method in original_methods:
        memory_layer.memory_retrieval.forward = original_method

    return {
        "qa1_activations": qa1_activations,
        "qa2_activations": qa2_activations,
        "answer_tokens1": answer_token_ids1.tolist(),
        "answer_tokens2": answer_token_ids2.tolist(),
        "memory_layers": found_memory_layers,
    }


def forward_pass_with_swapped_keys(
    model, tokenizer, qa1, qa2, swap_mapping_by_layer, device="cuda"
):
    """
    Do forward passes with swapped memory keys and track which tokens get swapped.

    Returns:
        Dict with results for both QA pairs including generated tokens and swap analysis
    """
    question1, answer1 = qa1
    question2, answer2 = qa2

    # Tokenize
    inputs1 = tokenizer(question1, return_tensors="pt").to(device)
    inputs2 = tokenizer(question2, return_tensors="pt").to(device)

    answer_tokens1 = tokenizer(answer1, add_special_tokens=False, return_tensors="pt")
    answer_tokens2 = tokenizer(answer2, add_special_tokens=False, return_tensors="pt")

    answer_token_ids1 = answer_tokens1["input_ids"][0].to(device)
    answer_token_ids2 = answer_tokens2["input_ids"][0].to(device)

    results = {
        "qa1": {"generated_tokens": [], "activations": {}},
        "qa2": {"generated_tokens": [], "activations": {}},
    }

    # Process QA1 with swapped keys
    original_methods = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, MemoryLayer):
            if i in swap_mapping_by_layer:
                original_method = patch_memory_layer_for_swapping(
                    layer.mlp,
                    i,
                    swap_mapping_by_layer[i],
                    results["qa1"]["activations"],
                )
                original_methods.append((layer.mlp, original_method))

    # Generate for QA1
    current_ids = inputs1["input_ids"]
    with torch.no_grad():
        for i in range(len(answer_token_ids1)):
            outputs = model(input_ids=current_ids)
            next_logits = outputs.logits[0, -1, :]
            predicted_token = next_logits.argmax().item()
            results["qa1"]["generated_tokens"].append(predicted_token)
            current_ids = torch.cat(
                [current_ids, torch.tensor([[predicted_token]], device=device)], dim=1
            )

    # Restore original methods
    for memory_layer, original_method in original_methods:
        memory_layer.memory_retrieval.forward = original_method

    # Process QA2 with swapped keys
    original_methods = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, MemoryLayer):
            if i in swap_mapping_by_layer:
                original_method = patch_memory_layer_for_swapping(
                    layer.mlp,
                    i,
                    swap_mapping_by_layer[i],
                    results["qa2"]["activations"],
                )
                original_methods.append((layer.mlp, original_method))

    # Generate for QA2
    current_ids = inputs2["input_ids"]
    with torch.no_grad():
        for i in range(len(answer_token_ids2)):
            outputs = model(input_ids=current_ids)
            next_logits = outputs.logits[0, -1, :]
            predicted_token = next_logits.argmax().item()
            results["qa2"]["generated_tokens"].append(predicted_token)
            current_ids = torch.cat(
                [current_ids, torch.tensor([[predicted_token]], device=device)], dim=1
            )

    # Restore original methods
    for memory_layer, original_method in original_methods:
        memory_layer.memory_retrieval.forward = original_method

    # Analyze swaps
    results["qa1"]["answer_tokens"] = answer_token_ids1.tolist()
    results["qa2"]["answer_tokens"] = answer_token_ids2.tolist()

    # Check if tokens swapped (QA1 generated QA2's token and vice versa)
    qa1_swapped_positions = []
    qa2_swapped_positions = []

    for i in range(len(answer_token_ids1)):
        if i < len(answer_token_ids2):
            # Check if QA1 generated QA2's answer token
            if results["qa1"]["generated_tokens"][i] == answer_token_ids2[i].item():
                qa1_swapped_positions.append(i)

    for i in range(len(answer_token_ids2)):
        if i < len(answer_token_ids1):
            # Check if QA2 generated QA1's answer token
            if results["qa2"]["generated_tokens"][i] == answer_token_ids1[i].item():
                qa2_swapped_positions.append(i)

    results["qa1_swapped_positions"] = qa1_swapped_positions
    results["qa2_swapped_positions"] = qa2_swapped_positions

    return results


def check_memorization(model, tokenizer, question, answer, device="cuda"):
    """
    Check if a QA pair has been memorized by generating an answer and comparing.
    Returns (is_memorized, predicted_answer)
    """
    # Tokenize just the question
    inputs = tokenizer(question, return_tensors="pt").to(device)

    # Get expected answer length
    answer_tokens = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
    answer_length = answer_tokens["input_ids"].shape[1]

    # Generate answer with exact length
    with torch.no_grad():
        generated = model.generate(
            inputs["input_ids"],
            max_new_tokens=answer_length,
            min_new_tokens=answer_length,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Extract only the generated part
    generated_answer_ids = generated[0][inputs["input_ids"].shape[1]:]
    predicted_answer = tokenizer.decode(generated_answer_ids, skip_special_tokens=True).strip()

    # Check if memorized (case insensitive)
    is_memorized = predicted_answer.lower() == answer.lower()

    return is_memorized, predicted_answer


def load_dataset_with_frequencies(dataset_path):
    """Load dataset and extract frequency information for each QA pair."""
    # Load metadata to understand frequency buckets
    metadata_path = f"{dataset_path}/metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    freq_less = metadata["freq_less"]
    freq_more = metadata["freq_more"]

    # Load test dataset
    dataset = load_dataset(dataset_path, split="test")

    # Extract QA pairs with frequency information
    qa_pairs_by_freq = {
        "less_frequent": [],  # freq_less (typically 1)
        "more_frequent": []   # freq_more (typically 10)
    }

    # Process dataset to categorize by frequency
    # In the synthetic dataset, the first half are less frequent, second half are more frequent
    num_pairs = metadata["num_pairs"]
    pairs_per_bucket = metadata["pairs_per_bucket"]

    all_qa_pairs = []
    for sample in dataset:
        text = sample["text"]
        parts = text.split("? ")
        if len(parts) == 2:
            question = parts[0] + "?"
            answer = parts[1].strip()
            all_qa_pairs.append((question, answer))

    # Assign frequencies based on dataset structure
    # First 1000 pairs are freq=1, next 1000 are freq=10
    for i, qa_pair in enumerate(all_qa_pairs):
        # Determine frequency based on position in dataset
        pair_idx = i % num_pairs  # Handle repeated samples
        if pair_idx < pairs_per_bucket[str(freq_less)]:
            qa_pairs_by_freq["less_frequent"].append(qa_pair)
        else:
            qa_pairs_by_freq["more_frequent"].append(qa_pair)

    return qa_pairs_by_freq, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Swap memory keys between QA pairs and measure intervention effects"
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
        help="Dataset size to use (200, 2K, 10K, or 50K)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of random QA pair swaps to perform",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON file with swap results",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

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
    dataset_path = f"experiments/synthetic_qa/datasets/synthetic_qa_data_{args.dataset_size}"
    qa_pairs_by_freq, metadata = load_dataset_with_frequencies(dataset_path)

    print(f"\nDataset loaded:")
    print(f"  - Less frequent pairs (freq={metadata['freq_less']}): {len(qa_pairs_by_freq['less_frequent'])}")
    print(f"  - More frequent pairs (freq={metadata['freq_more']}): {len(qa_pairs_by_freq['more_frequent'])}")
    print(f"  - Total pairs: {len(qa_pairs_by_freq['less_frequent']) + len(qa_pairs_by_freq['more_frequent'])}")

    # Results storage by swap type
    swap_types = [
        ("less_less", "less_frequent", "less_frequent"),
        ("more_more", "more_frequent", "more_frequent"),
        ("less_more", "less_frequent", "more_frequent"),
        ("more_less", "more_frequent", "less_frequent")
    ]

    results_by_type = {}
    for swap_name, _, _ in swap_types:
        results_by_type[swap_name] = {
            "results": [],
            "total_swapped_tokens_qa1": 0,
            "total_swapped_tokens_qa2": 0,
            "total_answer_tokens_qa1": 0,
            "total_answer_tokens_qa2": 0,
            "skipped_count": 0
        }

    # Determine samples per swap type
    samples_per_type = args.num_samples // 4
    remaining_samples = args.num_samples % 4

    print(f"\nPerforming {args.num_samples} QA pair swaps:")
    print(f"  - {samples_per_type} swaps per type (4 types)")
    if remaining_samples > 0:
        print(f"  - {remaining_samples} additional swaps distributed among types")

    overall_sample_idx = 0

    for swap_idx, (swap_name, freq1_type, freq2_type) in enumerate(swap_types):
        # Calculate number of samples for this swap type
        num_samples_this_type = samples_per_type
        if swap_idx < remaining_samples:
            num_samples_this_type += 1

        print(f"\n{swap_name.upper()} swaps ({freq1_type} ↔ {freq2_type}): {num_samples_this_type} samples")

        for type_sample_idx in tqdm(range(num_samples_this_type), desc=swap_name):
            # Select QA pairs from appropriate frequency buckets
            qa1_candidates = qa_pairs_by_freq[freq1_type]
            qa2_candidates = qa_pairs_by_freq[freq2_type]

            # Handle same-frequency swaps (need different pairs)
            if freq1_type == freq2_type:
                if len(qa1_candidates) < 2:
                    print(f"\nWarning: Not enough {freq1_type} pairs for swapping")
                    results_by_type[swap_name]["skipped_count"] += 1
                    continue
                idx1, idx2 = random.sample(range(len(qa1_candidates)), 2)
                qa1 = qa1_candidates[idx1]
                qa2 = qa1_candidates[idx2]
            else:
                # Different frequency swaps
                idx1 = random.randint(0, len(qa1_candidates) - 1)
                idx2 = random.randint(0, len(qa2_candidates) - 1)
                qa1 = qa1_candidates[idx1]
                qa2 = qa2_candidates[idx2]

            # Check if both QA pairs are memorized
            is_memorized1, pred1 = check_memorization(model, tokenizer, qa1[0], qa1[1], args.device)
            is_memorized2, pred2 = check_memorization(model, tokenizer, qa2[0], qa2[1], args.device)

            if not (is_memorized1 and is_memorized2):
                # Skip this pair if either is not memorized
                results_by_type[swap_name]["skipped_count"] += 1
                if not is_memorized1:
                    print(f"\nQA1 not memorized: {qa1[0]} Expected: {qa1[1]}, Got: {pred1}")
                if not is_memorized2:
                    print(f"\nQA2 not memorized: {qa2[0]} Expected: {qa2[1]}, Got: {pred2}")
                continue

            # Extract clean activations for both QA pairs
            clean_results = clean_forward_passes_dual(model, tokenizer, qa1, qa2, args.device)

            # Build swap mapping for each layer
            swap_mapping_by_layer = defaultdict(dict)

            # For each layer, create bidirectional mapping
            for layer_idx in clean_results["memory_layers"]:
                qa1_keys = set()
                qa2_keys = set()

                # Collect unique combined indices for each QA
                if layer_idx in clean_results["qa1_activations"]:
                    for act in clean_results["qa1_activations"][layer_idx]:
                        qa1_keys.add(act["combined_idx"])

                if layer_idx in clean_results["qa2_activations"]:
                    for act in clean_results["qa2_activations"][layer_idx]:
                        qa2_keys.add(act["combined_idx"])

                # Create bidirectional mapping
                qa1_list = sorted(list(qa1_keys))
                qa2_list = sorted(list(qa2_keys))

                # Map QA1 keys to QA2 keys (cyclically if different lengths)
                for i, key1 in enumerate(qa1_list):
                    if qa2_list:
                        key2 = qa2_list[i % len(qa2_list)]
                        swap_mapping_by_layer[layer_idx][key1] = key2

                # Map QA2 keys to QA1 keys
                for i, key2 in enumerate(qa2_list):
                    if qa1_list:
                        key1 = qa1_list[i % len(qa1_list)]
                        swap_mapping_by_layer[layer_idx][key2] = key1

            # Perform forward pass with swapped keys
            swap_results = forward_pass_with_swapped_keys(
                model, tokenizer, qa1, qa2, swap_mapping_by_layer, args.device
            )

            # Update statistics for this swap type
            results_by_type[swap_name]["total_swapped_tokens_qa1"] += len(swap_results["qa1_swapped_positions"])
            results_by_type[swap_name]["total_swapped_tokens_qa2"] += len(swap_results["qa2_swapped_positions"])
            results_by_type[swap_name]["total_answer_tokens_qa1"] += len(swap_results["qa1"]["answer_tokens"])
            results_by_type[swap_name]["total_answer_tokens_qa2"] += len(swap_results["qa2"]["answer_tokens"])

            # Store results
            result_entry = {
                "sample_idx": overall_sample_idx,
                "swap_type": swap_name,
                "qa1_frequency": metadata["freq_less"] if freq1_type == "less_frequent" else metadata["freq_more"],
                "qa2_frequency": metadata["freq_less"] if freq2_type == "less_frequent" else metadata["freq_more"],
                "qa1": {
                    "question": qa1[0],
                    "answer": qa1[1],
                    "answer_tokens": swap_results["qa1"]["answer_tokens"],
                    "generated_tokens": swap_results["qa1"]["generated_tokens"],
                    "swapped_positions": swap_results["qa1_swapped_positions"],
                    "num_swapped": len(swap_results["qa1_swapped_positions"]),
                },
                "qa2": {
                    "question": qa2[0],
                    "answer": qa2[1],
                    "answer_tokens": swap_results["qa2"]["answer_tokens"],
                    "generated_tokens": swap_results["qa2"]["generated_tokens"],
                    "swapped_positions": swap_results["qa2_swapped_positions"],
                    "num_swapped": len(swap_results["qa2_swapped_positions"]),
                },
                "swap_mapping_sizes": {
                    str(layer_idx): len(mapping)
                    for layer_idx, mapping in swap_mapping_by_layer.items()
                },
            }

            results_by_type[swap_name]["results"].append(result_entry)
            overall_sample_idx += 1

    # Calculate statistics for each swap type
    swap_type_summaries = {}
    overall_total_swapped = 0
    overall_total_tokens = 0
    overall_skipped = 0

    for swap_name, data in results_by_type.items():
        total_swapped = data["total_swapped_tokens_qa1"] + data["total_swapped_tokens_qa2"]
        total_tokens = data["total_answer_tokens_qa1"] + data["total_answer_tokens_qa2"]

        swap_percentage = (total_swapped / total_tokens * 100) if total_tokens > 0 else 0
        qa1_swap_percentage = (data["total_swapped_tokens_qa1"] / data["total_answer_tokens_qa1"] * 100) if data["total_answer_tokens_qa1"] > 0 else 0
        qa2_swap_percentage = (data["total_swapped_tokens_qa2"] / data["total_answer_tokens_qa2"] * 100) if data["total_answer_tokens_qa2"] > 0 else 0

        swap_type_summaries[swap_name] = {
            "swap_percentage": swap_percentage,
            "qa1_swap_percentage": qa1_swap_percentage,
            "qa2_swap_percentage": qa2_swap_percentage,
            "total_swapped_tokens": total_swapped,
            "total_answer_tokens": total_tokens,
            "total_swapped_tokens_qa1": data["total_swapped_tokens_qa1"],
            "total_swapped_tokens_qa2": data["total_swapped_tokens_qa2"],
            "total_answer_tokens_qa1": data["total_answer_tokens_qa1"],
            "total_answer_tokens_qa2": data["total_answer_tokens_qa2"],
            "samples_processed": len(data["results"]),
            "samples_skipped": data["skipped_count"]
        }

        overall_total_swapped += total_swapped
        overall_total_tokens += total_tokens
        overall_skipped += data["skipped_count"]

    overall_swap_percentage = (overall_total_swapped / overall_total_tokens * 100) if overall_total_tokens > 0 else 0

    # Prepare final output
    all_results = []
    for swap_name in ["less_less", "more_more", "less_more", "more_less"]:
        all_results.extend(results_by_type[swap_name]["results"])

    output_data = {
        "config": {
            "model_path": args.model_path,
            "dataset_size": args.dataset_size,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "freq_less": metadata["freq_less"],
            "freq_more": metadata["freq_more"],
        },
        "summary": {
            "overall": {
                "swap_percentage": overall_swap_percentage,
                "total_swapped_tokens": overall_total_swapped,
                "total_answer_tokens": overall_total_tokens,
                "samples_processed": len(all_results),
                "samples_skipped": overall_skipped,
            },
            "by_swap_type": swap_type_summaries
        },
        "results": all_results,
    }

    # Save results
    print(f"\nSaving results to {args.output_file}...")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total samples requested: {args.num_samples}")
    print(f"Total samples skipped (not memorized): {overall_skipped}")
    print(f"Total samples successfully processed: {len(all_results)}")
    print(f"\nOVERALL swap percentage: {overall_swap_percentage:.2f}%")
    print(f"Total swapped tokens: {overall_total_swapped}/{overall_total_tokens}")

    print("\n" + "-"*70)
    print("BY SWAP TYPE:")
    print("-"*70)

    for swap_name, summary in swap_type_summaries.items():
        freq1, freq2 = swap_name.split("_")
        freq1_val = metadata["freq_less"] if freq1 == "less" else metadata["freq_more"]
        freq2_val = metadata["freq_less"] if freq2 == "less" else metadata["freq_more"]

        print(f"\n{swap_name.upper()} (freq {freq1_val} ↔ freq {freq2_val}):")
        print(f"  - Samples processed: {summary['samples_processed']}")
        print(f"  - Samples skipped: {summary['samples_skipped']}")
        print(f"  - Swap percentage: {summary['swap_percentage']:.2f}%")
        print(f"  - QA1→QA2 swap %: {summary['qa1_swap_percentage']:.2f}%")
        print(f"  - QA2→QA1 swap %: {summary['qa2_swap_percentage']:.2f}%")
        print(f"  - Swapped tokens: {summary['total_swapped_tokens']}/{summary['total_answer_tokens']}")

    print("="*70)


if __name__ == "__main__":
    main()
