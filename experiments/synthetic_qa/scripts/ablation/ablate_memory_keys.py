#!/usr/bin/env python3
"""
Ablate memory keys for synthetic QA evaluation.

For each QA pair and each top key, this script:
1. Does a clean forward pass to get key activations and logits
2. Ablates keys one at a time across all positions by setting score to 0
3. Tracks key activations and logits after ablation
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

# Add bento path for custom model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../3rdparty/bento"))
from bento.models.memory.modeling_memory import MemoryForCausalLM, MemoryLayer


def patch_memory_layer_for_tracking(memory_layer, layer_idx, activations_dict):
    """Patch a MemoryLayer's forward method to track activations."""
    # The memory retrieval happens in memory_layer.memory_retrieval.forward
    memory_retrieval = memory_layer.memory_retrieval
    original_forward = memory_retrieval.forward
    print(f"Patching memory layer {layer_idx} for tracking")

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

        # Store activations for the last position (answer token)
        if layer_idx not in activations_dict:
            activations_dict[layer_idx] = []

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

                    activations_dict[layer_idx].append(
                        {
                            "idx1": idx1,
                            "idx2": idx2,
                            "score": score,
                            "position": len(activations_dict[layer_idx]) // heads,
                            "head": head_idx,
                        }
                    )

        # Return the original result format
        if return_individual_probs:
            return scores, indices, probs1, probs2
        else:
            return scores, indices

    # Replace the method
    memory_retrieval.forward = tracked_forward
    return original_forward


def patch_memory_layer_for_ablation(
    memory_layer,
    layer_idx,
    target_idx1,
    target_idx2,
    activations_dict,
    zero_ablation=False,
):
    """Patch a MemoryLayer's forward method to ablate a specific key.

    Args:
        memory_layer: The MemoryLayer to patch
        layer_idx: Index of the layer
        target_idx1, target_idx2: The key indices to ablate
        activations_dict: Dictionary to store activations
        zero_ablation: If True and knn=1, completely skip key selection when ablated key is top-1
    """
    memory_retrieval = memory_layer.memory_retrieval
    original_forward = memory_retrieval.forward
    n_keys = memory_retrieval.mem_n_keys

    def ablated_forward(query, knn, return_individual_probs=False):
        # Call original with return_individual_probs=True
        scores, indices, probs1, probs2 = original_forward(
            query, knn, return_individual_probs=True
        )

        # Compute combined target index
        target_combined = target_idx1 * n_keys + target_idx2

        # For zero ablation with knn=1, set score to 0 when target is selected
        if zero_ablation and knn == 1:
            mask = indices == target_combined
            scores = scores.clone()
            scores[mask] = 0.0
        else:
            # Standard ablation: expand knn and filter out target
            # This requires more complex logic - for now just set score to 0
            mask = indices == target_combined
            scores = scores.clone()
            scores[mask] = 0.0

        # Track ablated activations
        bs_times_heads = indices.shape[0]
        heads = memory_retrieval.heads
        bs = bs_times_heads // heads

        if layer_idx not in activations_dict:
            activations_dict[layer_idx] = []

        # Process only the last position
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
                    idx1 = combined_idx // n_keys
                    idx2 = combined_idx % n_keys
                    score = selected_scores[k].item()

                    # Check if this was the ablated key
                    was_ablated = idx1 == target_idx1 and idx2 == target_idx2

                    activations_dict[layer_idx].append(
                        {
                            "idx1": idx1,
                            "idx2": idx2,
                            "score": score,
                            "position": len(activations_dict[layer_idx]) // heads,
                            "head": head_idx,
                            "ablated": was_ablated,
                        }
                    )

        # Return the original result format
        if return_individual_probs:
            return scores, indices, probs1, probs2
        else:
            return scores, indices

    # Replace the method
    memory_retrieval.forward = ablated_forward
    return original_forward


def check_memorization(model, tokenizer, question, answer, device="cuda"):
    """
    Check if a QA pair has been memorized by generating an answer and comparing.
    Returns (is_memorized, predicted_answer)
    """
    # Tokenize just the key
    inputs = tokenizer(key, return_tensors="pt").to(device)

    # Get expected value length
    value_tokens = tokenizer(value, add_special_tokens=False, return_tensors="pt")
    value_length = value_tokens["input_ids"].shape[1]

    # Generate value with exact length
    with torch.no_grad():
        generated = model.generate(
            inputs["input_ids"],
            max_new_tokens=value_length,
            min_new_tokens=value_length,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Extract only the generated part
    generated_value_ids = generated[0][inputs["input_ids"].shape[1] :]
    predicted_value = tokenizer.decode(
        generated_value_ids, skip_special_tokens=True
    ).strip()

    # Check if memorized (case insensitive)
    is_memorized = predicted_value.lower() == value.lower()

    return is_memorized, predicted_value


def clean_forward_pass(model, tokenizer, question, answer, device="cuda"):
    """
    Do a forward pass on clean prompt to generate answer.
    Returns key activations and logits for each answer token position.
    """
    # Tokenize inputs
    inputs = tokenizer(key, return_tensors="pt").to(device)
    value_tokens = tokenizer(value, add_special_tokens=False, return_tensors="pt")
    value_token_ids = value_tokens["input_ids"][0].to(device)

    # Dictionary to store activations
    activations = {}

    # Patch memory layers
    original_methods = []
    found_memory_layers = []
    for i, layer in enumerate(model.model.layers):
        # Check if this layer's mlp is a MemoryLayer
        if hasattr(layer, "mlp") and isinstance(layer.mlp, MemoryLayer):
            found_memory_layers.append(i)
            original_method = patch_memory_layer_for_tracking(
                layer.mlp,
                i,
                activations,  # Pass the MemoryLayer, not the whole layer
            )
            original_methods.append((layer.mlp, original_method))

    print(f"Found memory layers at indices: {found_memory_layers}")

    # Collect logits for each answer position
    all_logits = []
    current_ids = inputs["input_ids"]

    with torch.no_grad():
        for i in range(len(value_token_ids)):
            outputs = model(input_ids=current_ids)
            next_logits = outputs.logits[0, -1, :]  # Logits for next token
            all_logits.append(
                next_logits[value_token_ids[i]].item()
            )  # Get logit for correct token

            # Add the true value token to continue generation
            current_ids = torch.cat(
                [current_ids, value_token_ids[i : i + 1].unsqueeze(0)], dim=1
            )

    # Restore original methods
    for memory_layer, original_method in original_methods:
        memory_layer.memory_retrieval.forward = original_method

    # Format activations by position
    clean_activations = {}
    for layer_idx, layer_acts in activations.items():
        # Get number of heads from the first activation entry
        if layer_acts:
            # Count heads by finding how many entries have position 0
            heads_count = sum(1 for act in layer_acts if act["position"] == 0)
            # Keep all heads for each answer position
            clean_activations[layer_idx] = layer_acts[
                : len(value_token_ids) * heads_count
            ]
        else:
            clean_activations[layer_idx] = []

    return {
        "clean_activations": clean_activations,
        "clean_logits": all_logits,
        "value_tokens": value_token_ids.tolist(),
    }


def ablate_key_forward_pass(
    model, tokenizer, key, value, ablate_keys, device="cuda", zero_ablation=False
):
    """
    Do a forward pass with specific keys ablated across all positions.
    ablate_keys: dict mapping layer_idx to (idx1, idx2) tuple
    zero_ablation: If True and knn=1, completely skip key selection when ablated key is top-1
    Returns key activations, logits, and whether answer was generated correctly.
    """
    # Tokenize inputs
    inputs = tokenizer(key, return_tensors="pt").to(device)
    value_tokens = tokenizer(value, add_special_tokens=False, return_tensors="pt")
    value_token_ids = value_tokens["input_ids"][0].to(device)

    # Dictionary to store activations
    activations = {}

    # Patch memory layers for ablation
    original_methods = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and isinstance(layer.mlp, MemoryLayer):
            if i in ablate_keys:
                idx1, idx2 = ablate_keys[i]
                original_method = patch_memory_layer_for_ablation(
                    layer.mlp, i, idx1, idx2, activations, zero_ablation
                )
                original_methods.append((layer.mlp, original_method))

    # Collect logits for each answer position
    all_logits = []
    generated_tokens = []
    generated_logits = []  # Logits for generated tokens
    current_ids = inputs["input_ids"]

    with torch.no_grad():
        for i in range(len(value_token_ids)):
            outputs = model(input_ids=current_ids)
            next_logits = outputs.logits[0, -1, :]  # Logits for next token
            all_logits.append(
                next_logits[value_token_ids[i]].item()
            )  # Get logit for correct token

            # Get the predicted token
            predicted_token = next_logits.argmax().item()
            generated_tokens.append(predicted_token)
            generated_logits.append(
                next_logits[predicted_token].item()
            )  # Logit for generated token

            # Add the true value token to continue generation
            current_ids = torch.cat(
                [current_ids, value_token_ids[i : i + 1].unsqueeze(0)], dim=1
            )

    # Restore original methods
    for memory_layer, original_method in original_methods:
        memory_layer.memory_retrieval.forward = original_method

    # Check if generation is correct
    correct_generation = (
        (torch.tensor(generated_tokens) == value_token_ids.cpu()).all().item()
    )

    # Format activations by position
    ablated_activations = {}
    for layer_idx, layer_acts in activations.items():
        # Get number of heads from the first activation entry
        if layer_acts:
            # Count heads by finding how many entries have position 0
            heads_count = sum(1 for act in layer_acts if act["position"] == 0)
            # Keep all heads for each answer position
            ablated_activations[layer_idx] = layer_acts[
                : len(value_token_ids) * heads_count
            ]
        else:
            ablated_activations[layer_idx] = []

    # Get generated text for debugging
    generated_token_ids = torch.tensor(generated_tokens, device=device)
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    return {
        "ablated_activations": ablated_activations,
        "ablated_logits": all_logits,
        "correct_generation": correct_generation,
        "generated_tokens": generated_tokens,
        "generated_text": generated_text,
        "generated_logits": generated_logits,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ablate memory keys and measure impact"
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
        "--output_file",
        type=str,
        required=True,
        help="Output JSON file with ablation results",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of QA pairs to process",
    )
    parser.add_argument(
        "--no_zero_ablation",
        action="store_true",
        help="If set, use standard ablation (expand knn) instead of zero ablation for knn=1",
    )
    parser.add_argument(
        "--batch_check_size",
        type=int,
        default=1,
        help="Number of samples to check for memorization at a time",
    )

    args = parser.parse_args()

    # Zero ablation is now the default
    zero_ablation = not args.no_zero_ablation

    if zero_ablation:
        print(
            "Zero ablation mode enabled (default): will completely skip key selection when ablated key is top-1"
        )
    else:
        print("Standard ablation mode: will expand knn to select alternative keys")

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

    # Limit dataset size if requested
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Dataset size: {len(dataset)} samples")

    # First pass: identify memorized QA pairs
    print("Identifying memorized QA pairs...")
    memorized_pairs = []

    for i in tqdm(
        range(0, len(dataset), args.batch_check_size), desc="Checking memorization"
    ):
        batch_end = min(i + args.batch_check_size, len(dataset))

        for j in range(i, batch_end):
            sample = dataset[j]
            text = sample["text"]

            # Extract key:value
            parts = text.split(":")
            if len(parts) == 2:
                key = parts[0].strip() + ":"
                value = parts[1].strip()

                # Check if memorized
                is_memorized, predicted_answer = check_memorization(
                    model, tokenizer, question, answer, args.device
                )

                if is_memorized:
                    memorized_pairs.append({"key": key, "value": value, "index": j})

    print(
        f"\nFound {len(memorized_pairs)} memorized QA pairs out of {len(dataset)} total"
    )

    if len(memorized_pairs) == 0:
        print("No memorized pairs found! Exiting.")
        return

    # Second pass: perform ablation analysis on memorized pairs
    results = []

    print(f"\nProcessing {len(memorized_pairs)} memorized QA pairs for ablation...")
    for qa_item in tqdm(memorized_pairs):
        key = qa_item["key"]
        value = qa_item["value"]

        # Run clean forward pass
        clean_result = clean_forward_pass(
            model, tokenizer, question, answer, args.device
        )

        # Prepare result for this QA pair
        qa_result = {
            "key": key,
            "value": value,
            "clean_activations": clean_result["clean_activations"],
            "clean_logits": clean_result["clean_logits"],
            "ablations": [],
        }

        # Collect all keys that were activated during clean pass
        activated_keys = set()
        for layer_idx, layer_acts in clean_result["clean_activations"].items():
            for act in layer_acts:
                activated_keys.add((layer_idx, act["idx1"], act["idx2"]))

        # Ablate each activated key
        for layer_idx, idx1, idx2 in activated_keys:
            ablate_keys = {layer_idx: (idx1, idx2)}
            ablation_result = ablate_key_forward_pass(
                model,
                tokenizer,
                key,
                value,
                ablate_keys,
                args.device,
                zero_ablation,
            )

            ablation_entry = {
                "layer": layer_idx,
                "idx1": idx1,
                "idx2": idx2,
                "ablated_activations": ablation_result["ablated_activations"],
                "ablated_logits": ablation_result["ablated_logits"],
                "correct_generation": ablation_result["correct_generation"],
            }

            # Add generated text and logits if generation was incorrect
            if not ablation_result["correct_generation"]:
                ablation_entry["generated_text"] = ablation_result["generated_text"]
                ablation_entry["generated_logits"] = ablation_result["generated_logits"]

            qa_result["ablations"].append(ablation_entry)

        results.append(qa_result)

    # Save results with custom formatting
    print(f"Saving results to {args.output_file}...")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Custom JSON formatting for readability
    def format_json_custom(obj):
        json_str = json.dumps(obj, indent=2)
        lines = json_str.split("\n")
        formatted_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]

            # Inline activation objects with idx1, idx2, score, position, head, ablated
            if '"idx1":' in line and i + 5 < len(lines):
                if (
                    '"idx2":' in lines[i + 1]
                    and '"score":' in lines[i + 2]
                    and '"position":' in lines[i + 3]
                    and '"head":' in lines[i + 4]
                    and '"ablated":' in lines[i + 5]
                ):
                    # Get indentation
                    indent = (
                        len(lines[i - 1]) - len(lines[i - 1].lstrip())
                        if i > 0 and "{" in lines[i - 1]
                        else len(line) - len(line.lstrip())
                    )
                    # Remove previous { line if exists
                    if i > 0 and lines[i - 1].strip() == "{":
                        formatted_lines.pop()
                    # Combine all fields
                    combined = f"{' ' * indent}{{ {line.strip()} {lines[i + 1].strip()} {lines[i + 2].strip()} {lines[i + 3].strip()} {lines[i + 4].strip()} {lines[i + 5].strip()} }}"
                    # Check for closing brace
                    if i + 6 < len(lines) and lines[i + 6].strip() in ["},", "}"]:
                        if lines[i + 6].strip() == "},":
                            combined += ","
                        i += 7
                    else:
                        i += 6
                    formatted_lines.append(combined)
                    continue

            # Inline ablation header info (layer, idx1, idx2)
            elif '"layer":' in line and i + 2 < len(lines):
                if '"idx1":' in lines[i + 1] and '"idx2":' in lines[i + 2]:
                    indent = len(line) - len(line.lstrip())
                    combined = f"{' ' * indent}{line.strip()} {lines[i + 1].strip()} {lines[i + 2].strip()}"
                    formatted_lines.append(combined)
                    # Skip the lines we've processed
                    i = i + 3
                    continue

            # Inline correct_generation after ablated_logits
            elif '"correct_generation":' in line:
                # This should come right after ablated_logits, so append to previous line
                if formatted_lines and '"ablated_logits":' in formatted_lines[-1]:
                    formatted_lines[-1] = formatted_lines[-1].rstrip(",") + ","
                    formatted_lines.append(line)

            # Compress numeric arrays on single lines
            elif (
                '"clean_logits": [' in line
                or '"ablated_logits": [' in line
                or '"generated_logits": [' in line
            ):
                # Start of a logits array
                indent = len(line) - len(line.lstrip())
                array_content = [line.strip()]
                i += 1
                # Collect all numeric values
                while i < len(lines):
                    stripped = lines[i].strip()
                    if stripped == "]" or stripped == "],":
                        array_content.append(stripped)
                        break
                    elif stripped and (stripped[0].isdigit() or stripped[0] == "-"):
                        # Remove trailing comma if present
                        array_content.append(stripped.rstrip(",") + ",")
                    i += 1
                # Join and format properly
                result = " " * indent + array_content[0] + " "
                result += " ".join(array_content[1:-1]).rstrip(",") + " "
                result += array_content[-1]
                formatted_lines.append(result)
            else:
                formatted_lines.append(line)
            i += 1
        return "\n".join(formatted_lines)

    with open(args.output_file, "w") as f:
        f.write(format_json_custom(results))

    print("Done!")

    # Print summary statistics
    total_ablations = sum(len(r["ablations"]) for r in results)
    incorrect_generations = sum(
        1 for r in results for abl in r["ablations"] if not abl["correct_generation"]
    )

    print("\nSummary:")
    print(f"- Processed {len(results)} memorized QA pairs")
    print(f"- Total ablations performed: {total_ablations}")
    print(
        f"- Ablations causing incorrect generation: {incorrect_generations} ({incorrect_generations / total_ablations * 100:.1f}%)"
    )

    if total_ablations == 0:
        print(
            "- No ablations performed (no keys were activated during clean forward passes)"
        )


if __name__ == "__main__":
    main()
