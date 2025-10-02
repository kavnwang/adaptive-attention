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
        scores, indices, probs1, probs2 = original_forward(query, knn, return_individual_probs=True)

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
                selected_scores = scores[pos_idx]   # shape: (knn,)

                # Decode combined indices to get idx1 and idx2
                if selected_indices.dim() == 0:  # knn=1 case
                    selected_indices = selected_indices.unsqueeze(0)
                    selected_scores = selected_scores.unsqueeze(0)

                for k in range(len(selected_indices)):
                    combined_idx = selected_indices[k].item()
                    idx1 = combined_idx // n_keys
                    idx2 = combined_idx % n_keys
                    score = selected_scores[k].item()

                    activations_dict[layer_idx].append({
                        "idx1": idx1,
                        "idx2": idx2,
                        "score": score,
                        "position": len(activations_dict[layer_idx]) // heads,
                        "head": head_idx,
                    })

        # Return the original result format
        if return_individual_probs:
            return scores, indices, probs1, probs2
        else:
            return scores, indices

    # Replace the method
    memory_retrieval.forward = tracked_forward
    return original_forward
        )  # (bs, heads, k_dim)
        half = layer.memory.k_dim // 2

        # keys : (heads, 2, n_keys, half)
        keys = layer.memory.keys.view(layer.memory.heads, 2, -1, half)
        keys1 = keys[:, 0, :, :]  # (heads, n_keys, half)
        keys2 = keys[:, 1, :, :]  # (heads, n_keys, half)
        n_keys = len(keys[0][0])

        # split query for product quantization
        q1 = query_reshaped[:, :, :half]  # (bs, heads, half)
        q2 = query_reshaped[:, :, half:]  # (bs, heads, half)

        # compute raw scores (before softmax)
        raw_scores1 = torch.einsum("blh, lkh->blk", q1, keys1)  # (bs, heads, n_keys)
        raw_scores2 = torch.einsum("blh, lkh->blk", q2, keys2)  # (bs, heads, n_keys)

        # Call original method to get the selected indices
        scores, indices = original_get_indices(query, knn)

        # Convert combined indices back to idx1, idx2
        indices1 = indices // n_keys
        indices2 = indices % n_keys

        # For each position in the batch, track the top key
        # We're interested in the last position (current generation step)
        if bs > 0:
            # Get data for last position across all heads
            last_pos_scores = scores[-layer.memory.heads :]  # (heads, knn)
            last_pos_indices1 = indices1[-layer.memory.heads :]  # (heads, knn)
            last_pos_indices2 = indices2[-layer.memory.heads :]  # (heads, knn)

            # Average scores across heads to find top key
            if layer.memory.heads > 1:
                avg_scores = last_pos_scores.mean(dim=0)  # (knn,)
                top_k_idx = avg_scores.argmax().item()

                # Find which head contributed most at this position
                head_scores_at_k = last_pos_scores[:, top_k_idx]  # (heads,)
                best_head = head_scores_at_k.argmax().item()

                idx1 = last_pos_indices1[best_head, top_k_idx].item()
                idx2 = last_pos_indices2[best_head, top_k_idx].item()

                # Get the raw score for this key (sum of both halves)
                raw_score = (
                    raw_scores1[-1, best_head, idx1] + raw_scores2[-1, best_head, idx2]
                ).item()
            else:
                # Single head case
                if last_pos_scores.dim() == 0:  # knn=1
                    idx1 = last_pos_indices1.item()
                    idx2 = last_pos_indices2.item()
                else:
                    top_idx = last_pos_scores.argmax().item()
                    idx1 = last_pos_indices1[top_idx].item()
                    idx2 = last_pos_indices2[top_idx].item()

                # Get the raw score for this key
                raw_score = (raw_scores1[-1, 0, idx1] + raw_scores2[-1, 0, idx2]).item()

            # Track the activation
            if layer_idx not in activations_dict:
                activations_dict[layer_idx] = []

            activations_dict[layer_idx].append(
                {
                    "idx1": idx1,
                    "idx2": idx2,
                    "score": raw_score,  # Raw score before softmax
                    "position": len(activations_dict[layer_idx]),
                }
            )

        return scores, indices

    # Replace the method
    layer.memory.get_indices = tracked_get_indices
    return original_get_indices


def clean_forward_pass(model, tokenizer, question, answer, device="cuda"):
    """
    Do a forward pass on clean prompt to generate answer.
    Returns key activations and logits for each answer token position.
    """
    # Tokenize inputs
    inputs = tokenizer(question, return_tensors="pt").to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
    answer_token_ids = answer_tokens["input_ids"][0].to(device)

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
                layer.mlp, i, activations  # Pass the MemoryLayer, not the whole layer
            )
            original_methods.append((layer.mlp, original_method))

    print(f"Found memory layers at indices: {found_memory_layers}")

    # Collect logits for each answer position
    all_logits = []
    current_ids = inputs["input_ids"]

    with torch.no_grad():
        for i in range(len(answer_token_ids)):
            outputs = model(input_ids=current_ids)
            next_logits = outputs.logits[0, -1, :]  # Logits for next token
            all_logits.append(
                next_logits[answer_token_ids[i]].item()
            )  # Get logit for correct token

            # Add the true answer token to continue generation
            current_ids = torch.cat(
                [current_ids, answer_token_ids[i : i + 1].unsqueeze(0)], dim=1
            )

    # Restore original methods
    for layer, original_method in original_methods:
        layer.memory.get_indices = original_method

    # Format activations by position
    clean_activations = {}
    for layer_idx, layer_acts in activations.items():
        clean_activations[layer_idx] = layer_acts[
            : len(answer_token_ids)
        ]  # Only keep answer positions

    return {
        "clean_activations": clean_activations,
        "clean_logits": all_logits,
        "answer_tokens": answer_token_ids.tolist(),
    }


def patch_memory_layer_for_ablation(
    layer, layer_idx, target_idx1, target_idx2, activations_dict, zero_ablation=False
):
    """Patch a memory layer's get_indices method to ablate a specific key.

    Args:
        layer: The memory layer to patch
        layer_idx: Index of the layer
        target_idx1, target_idx2: The key indices to ablate
        activations_dict: Dictionary to store activations
        zero_ablation: If True and knn=1, completely skip key selection when ablated key is top-1
    """
    original_get_indices = layer.memory.get_indices
    original_knn = layer.memory.knn

    def ablated_get_indices(query, knn):
        # Special handling for zero_ablation with knn=1
        if zero_ablation and knn == 1:
            # First, get the top key without modification
            scores_orig, indices_orig = original_get_indices(query, 1)

            # We need to compute raw scores to track them
            assert query.dim() == 2 and query.size(1) == layer.memory.k_dim
            bs = len(query) // layer.memory.heads
            query_reshaped = query.view(-1, layer.memory.heads, layer.memory.k_dim)
            half = layer.memory.k_dim // 2

            # keys : (heads, 2, n_keys, half)
            keys = layer.memory.keys.view(layer.memory.heads, 2, -1, half)
            keys1 = keys[:, 0, :, :]
            keys2 = keys[:, 1, :, :]
            n_keys = len(keys[0][0])

            # split query for product quantization
            q1 = query_reshaped[:, :, :half]
            q2 = query_reshaped[:, :, half:]

            # compute raw scores
            raw_scores1 = torch.einsum("blh, lkh->blk", q1, keys1)
            raw_scores2 = torch.einsum("blh, lkh->blk", q2, keys2)

            # Convert to idx1, idx2
            indices1_orig = indices_orig // n_keys
            indices2_orig = indices_orig % n_keys

            # Check if target key would be selected
            mask = (indices1_orig == target_idx1) & (indices2_orig == target_idx2)

            # For positions where target key would be selected, return zero scores
            scores = scores_orig.clone()
            indices = indices_orig.clone()
            scores[mask] = 0.0

            # Track activations for the last position
            if bs > 0 and layer_idx not in activations_dict:
                activations_dict[layer_idx] = []

            if bs > 0:
                # Get data for last position across all heads
                last_pos_mask = mask[-layer.memory.heads:]

                for head_idx in range(layer.memory.heads):
                    if last_pos_mask[head_idx]:
                        # This head would have selected the ablated key - record zero selection
                        activations_dict[layer_idx].append({
                            "idx1": -1,
                            "idx2": -1,
                            "score": 0.0,
                            "position": len(activations_dict[layer_idx]) // layer.memory.heads,
                            "head": head_idx,
                            "k_position": 0,
                            "softmax_score": 0.0,
                            "note": "key_ablated_zero"
                        })
                    else:
                        # Record the actual selected key
                        idx1 = indices1_orig[-layer.memory.heads + head_idx].item()
                        idx2 = indices2_orig[-layer.memory.heads + head_idx].item()
                        raw_score = (raw_scores1[-1, head_idx, idx1] + raw_scores2[-1, head_idx, idx2]).item()

                        activations_dict[layer_idx].append({
                            "idx1": idx1,
                            "idx2": idx2,
                            "score": raw_score,
                            "position": len(activations_dict[layer_idx]) // layer.memory.heads,
                            "head": head_idx,
                            "k_position": 0,
                            "softmax_score": scores_orig[-layer.memory.heads + head_idx].item()
                        })

            return scores, indices

        # Original behavior for standard ablation
        # Ensure knn is at least 2 for ablation
        effective_knn = max(knn, 2)

        # Temporarily set the layer's knn to effective_knn
        layer.memory.knn = effective_knn

        # We need to compute raw scores ourselves
        assert query.dim() == 2 and query.size(1) == layer.memory.k_dim
        bs = len(query) // layer.memory.heads
        query_reshaped = query.view(
            -1, layer.memory.heads, layer.memory.k_dim
        )  # (bs, heads, k_dim)
        half = layer.memory.k_dim // 2

        # keys : (heads, 2, n_keys, half)
        keys = layer.memory.keys.view(layer.memory.heads, 2, -1, half)
        keys1 = keys[:, 0, :, :]  # (heads, n_keys, half)
        keys2 = keys[:, 1, :, :]  # (heads, n_keys, half)
        n_keys = len(keys[0][0])

        # split query for product quantization
        q1 = query_reshaped[:, :, :half]  # (bs, heads, half)
        q2 = query_reshaped[:, :, half:]  # (bs, heads, half)

        # compute raw scores (before softmax)
        raw_scores1 = torch.einsum("blh, lkh->blk", q1, keys1)  # (bs, heads, n_keys)
        raw_scores2 = torch.einsum("blh, lkh->blk", q2, keys2)  # (bs, heads, n_keys)

        # Call original method with effective_knn
        scores, indices = original_get_indices(query, effective_knn)

        # Restore original knn
        layer.memory.knn = original_knn

        # Convert combined indices back to idx1, idx2
        indices1 = indices // n_keys
        indices2 = indices % n_keys

        # Find where the target key appears and set its score to 0
        mask = (indices1 == target_idx1) & (indices2 == target_idx2)
        scores = scores.clone()
        scores[mask] = 0.0

        # Renormalize scores
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-10)

        # If original knn was smaller, select top-k
        if knn < effective_knn:
            topk_scores, topk_indices = scores.topk(knn, dim=-1)
            selected_indices = indices.gather(-1, topk_indices)
            # Renormalize the selected scores
            scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-10)
            indices = selected_indices

            # Update indices1 and indices2 for tracking
            indices1 = indices // n_keys
            indices2 = indices % n_keys

        # Track activations for the last position
        if bs > 0 and layer_idx not in activations_dict:
            activations_dict[layer_idx] = []

        if bs > 0:
            # Get data for last position across all heads
            last_pos_scores = scores[-layer.memory.heads :]  # (heads, knn)
            last_pos_indices1 = indices1[-layer.memory.heads :]  # (heads, knn)
            last_pos_indices2 = indices2[-layer.memory.heads :]  # (heads, knn)

            # For ablation tracking, record ALL selected keys and their scores
            for head_idx in range(layer.memory.heads):
                for k_idx in range(scores.shape[-1]):
                    idx1 = last_pos_indices1[head_idx, k_idx].item()
                    idx2 = last_pos_indices2[head_idx, k_idx].item()
                    score = last_pos_scores[head_idx, k_idx].item()

                    # Get raw score for this key
                    raw_score = (
                        raw_scores1[-1, head_idx, idx1]
                        + raw_scores2[-1, head_idx, idx2]
                    ).item()

                    activations_dict[layer_idx].append(
                        {
                            "idx1": idx1,
                            "idx2": idx2,
                            "score": raw_score,  # Raw score before softmax
                            "position": len(activations_dict[layer_idx])
                            // (layer.memory.heads * scores.shape[-1]),
                            "head": head_idx,
                            "k_position": k_idx,
                            "softmax_score": score,  # Score after ablation and renormalization
                        }
                    )

        return scores, indices

    # Replace the method
    layer.memory.get_indices = ablated_get_indices
    return original_get_indices


def ablate_key_forward_pass(
    model, tokenizer, question, answer, ablate_keys, device="cuda", zero_ablation=False
):
    """
    Do a forward pass with specific keys ablated across all positions.
    ablate_keys: dict mapping layer_idx to (idx1, idx2) tuple
    zero_ablation: If True and knn=1, completely skip key selection when ablated key is top-1
    Returns key activations, logits, and whether answer was generated correctly.
    """
    # Tokenize inputs
    inputs = tokenizer(question, return_tensors="pt").to(device)
    answer_tokens = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
    answer_token_ids = answer_tokens["input_ids"][0].to(device)

    # Dictionary to store activations
    activations = {}

    # Patch memory layers for ablation
    original_methods = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "memory") and hasattr(layer.memory, "get_indices"):
            if i in ablate_keys:
                idx1, idx2 = ablate_keys[i]
                original_method = patch_memory_layer_for_ablation(
                    layer, i, idx1, idx2, activations, zero_ablation
                )
                original_methods.append((layer, original_method))

    # Collect logits for each answer position
    all_logits = []
    generated_tokens = []
    generated_logits = []  # Logits for generated tokens
    current_ids = inputs["input_ids"]

    with torch.no_grad():
        for i in range(len(answer_token_ids)):
            outputs = model(input_ids=current_ids)
            next_logits = outputs.logits[0, -1, :]  # Logits for next token
            all_logits.append(
                next_logits[answer_token_ids[i]].item()
            )  # Get logit for correct token

            # Get the predicted token
            predicted_token = next_logits.argmax().item()
            generated_tokens.append(predicted_token)
            generated_logits.append(
                next_logits[predicted_token].item()
            )  # Logit for generated token

            # Add the true answer token to continue generation
            current_ids = torch.cat(
                [current_ids, answer_token_ids[i : i + 1].unsqueeze(0)], dim=1
            )

    # Restore original methods
    for layer, original_method in original_methods:
        layer.memory.get_indices = original_method

    # Check if generation is correct
    correct_generation = (
        (torch.tensor(generated_tokens) == answer_token_ids.cpu()).all().item()
    )

    # Decode generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Format activations - only keep the top key per position for simplicity
    ablated_activations = {}
    for layer_idx, layer_acts in activations.items():
        # Group by position and select top key
        position_acts = {}
        for act in layer_acts:
            pos = act["position"]
            if pos not in position_acts:
                position_acts[pos] = []
            position_acts[pos].append(act)

        # For each position, find the key with highest average score across heads
        formatted_acts = []
        for pos in sorted(position_acts.keys()):
            # Group by key
            key_scores = {}
            for act in position_acts[pos]:
                key = (act["idx1"], act["idx2"])
                if key not in key_scores:
                    key_scores[key] = []
                key_scores[key].append(act["score"])

            # Find key with highest average score
            best_key = None
            best_score = float("-inf")
            for key, scores in key_scores.items():
                avg_score = sum(scores) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_key = key

            if best_key:
                formatted_acts.append(
                    {
                        "idx1": best_key[0],
                        "idx2": best_key[1],
                        "score": best_score,
                        "position": pos,
                    }
                )

        ablated_activations[layer_idx] = formatted_acts[: len(answer_token_ids)]

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
        "--input_file",
        type=str,
        required=True,
        help="Input JSON file with QA pairs and keys",
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

    args = parser.parse_args()

    # Zero ablation is now the default
    zero_ablation = not args.no_zero_ablation

    if zero_ablation:
        print("Zero ablation mode enabled (default): will completely skip key selection when ablated key is top-1")
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

    print("Loading input data...")
    with open(args.input_file, "r") as f:
        data = json.load(f)

    if args.max_samples:
        data = data[: args.max_samples]

    results = []

    print(f"Processing {len(data)} QA pairs...")
    for qa_idx, qa_item in enumerate(tqdm(data)):
        question = qa_item["question"]
        answer = qa_item["answer"]

        # Run clean forward pass
        clean_result = clean_forward_pass(
            model, tokenizer, question, answer, args.device
        )

        # Prepare result for this QA pair
        qa_result = {
            "question": question,
            "answer": answer,
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
                model, tokenizer, question, answer, ablate_keys, args.device, zero_ablation
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

            # Detect activation entries with idx1, idx2, score, position
            if '"idx1":' in line and i + 3 < len(lines):
                if (
                    '"idx2":' in lines[i + 1]
                    and '"score":' in lines[i + 2]
                    and '"position":' in lines[i + 3]
                ):
                    # Check if there's a { on the line above
                    if i > 0 and "{" in lines[i - 1]:
                        # Remove the previous line with just {
                        formatted_lines.pop()
                        # Get the base indentation from the { line
                        indent = len(lines[i - 1]) - len(lines[i - 1].lstrip())
                    else:
                        # Get the base indentation
                        indent = len(line) - len(line.lstrip())

                    # Combine into a single line
                    combined = (
                        " " * indent
                        + "{ "
                        + line.strip()
                        + " "
                        + lines[i + 1].strip()
                        + " "
                        + lines[i + 2].strip()
                        + " "
                        + lines[i + 3].strip()
                        + " }"
                    )

                    # Check if there's a closing brace after
                    if i + 4 < len(lines) and lines[i + 4].strip() == "},":
                        combined += ","
                        i += 5
                    elif i + 4 < len(lines) and lines[i + 4].strip() == "}":
                        i += 5
                    else:
                        i += 4

                    formatted_lines.append(combined)
                    continue

            # Detect logits arrays
            elif (
                '"clean_logits": [' in line
                or '"ablated_logits": [' in line
                or '"generated_logits": [' in line
            ):
                # Start of a logits array
                array_lines = [line]
                i += 1
                while i < len(lines) and "]" not in lines[i]:
                    array_lines.append(lines[i].strip())
                    i += 1
                if i < len(lines):
                    array_lines.append(lines[i].strip())

                # Combine into single line
                indent = len(line) - len(line.lstrip())
                logits_str = " " * indent + array_lines[0].strip()
                for j in range(1, len(array_lines)):
                    logits_str += " " + array_lines[j].strip().rstrip(",") + ","
                logits_str = logits_str.rstrip(",") + "],"
                formatted_lines.append(logits_str)
                i += 1
                continue

            formatted_lines.append(line)
            i += 1

        return "\n".join(formatted_lines)

    with open(output_path, "w") as f:
        f.write(format_json_custom(results))

    print("Done!")

    # Print summary statistics
    total_ablations = sum(len(r["ablations"]) for r in results)
    correct_ablations = sum(
        sum(1 for a in r["ablations"] if a["correct_generation"]) for r in results
    )

    print("\nSummary:")
    print(f"- Processed {len(results)} QA pairs")
    print(f"- Total ablations performed: {total_ablations}")
    if total_ablations > 0:
        print(
            f"- Correct generations after ablation: {correct_ablations}/{total_ablations} ({correct_ablations / total_ablations * 100:.1f}%)"
        )
    else:
        print("- No ablations performed (no keys were activated during clean forward passes)")


if __name__ == "__main__":
    main()
