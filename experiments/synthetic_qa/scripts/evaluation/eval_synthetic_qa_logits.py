#!/usr/bin/env python3
"""
Evaluate synthetic QA using logit-based scoring instead of generation.
This approach:
1. Computes the log probability of the correct answer given the question
2. Checks if the correct answer token has the highest probability
3. Avoids issues with generation producing number sequences
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import defaultdict
import sys
import os


def load_model_and_tokenizer(model_path, tokenizer_path, device="cuda"):
    """Load the model and tokenizer."""
    print(f"Loading model from {model_path}")

    # Add bento path for custom model
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "3rdparty/bento"))

    # Import the custom model class - try memory_mlp_router first, then memory
    try:
        from bento.models.memory_mlp_router.modeling_memory import MemoryForCausalLM

        model = MemoryForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
    except ImportError as e:
        print(f"Failed to import MemoryForCausalLM from memory_mlp_router: {e}")
        try:
            from bento.models.memory.modeling_memory import MemoryForCausalLM

            model = MemoryForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
            )
        except ImportError as e2:
            print(f"Failed to import MemoryForCausalLM from memory: {e2}")
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
            )

    model.eval()

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def find_answer_positions(input_ids, answer_ids):
    """Find where answer tokens appear in the input sequence."""
    seq_len = input_ids.shape[0]
    answer_len = answer_ids.shape[0]

    # Ensure both tensors are on the same device
    if input_ids.device != answer_ids.device:
        answer_ids = answer_ids.to(input_ids.device)

    for i in range(seq_len - answer_len + 1):
        if torch.equal(input_ids[i : i + answer_len], answer_ids):
            return list(range(i, i + answer_len))
    return None


def evaluate_qa_logits(model, tokenizer, qa_text, device="cuda"):
    """
    Evaluate a key-value pair using logit-based scoring.
    Returns metrics about how well the model predicts the value.
    """
    # Parse key:value
    parts = qa_text.split(":")
    if len(parts) != 2:
        return None

    key = parts[0].strip() + ":"  # Key with colon
    true_value = parts[1].strip()

    # Tokenize full Q&A as it appears in training data
    full_encoding = tokenizer(qa_text, return_tensors="pt", add_special_tokens=True)
    input_ids = full_encoding["input_ids"].to(device)

    # Tokenize value alone (no special tokens)
    value_encoding = tokenizer(
        true_value, return_tensors="pt", add_special_tokens=False
    )
    value_ids = value_encoding["input_ids"][0]

    # Find value position in full sequence
    value_positions = find_answer_positions(input_ids[0], value_ids)

    if value_positions is None:
        return None

    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Evaluate value prediction
    results = {
        "key": key,
        "true_value": true_value,
        "value_positions": value_positions,
        "value_tokens": value_ids.tolist(),
    }

    # For each value token position, check model's prediction
    value_log_probs = []
    value_ranks = []
    top_5_predictions = []

    for i, pos in enumerate(value_positions):
        if pos == 0:  # Skip if value starts at position 0
            continue

        # Get logits for predicting this position (from previous position)
        pred_logits = logits[pos - 1]  # (vocab_size,)
        probs = torch.softmax(pred_logits, dim=-1)

        # Get log probability of correct token
        correct_token_id = input_ids[0, pos].item()
        log_prob = torch.log(probs[correct_token_id]).item()
        value_log_probs.append(log_prob)

        # Get rank of correct token
        sorted_indices = torch.argsort(probs, descending=True)
        rank = (sorted_indices == correct_token_id).nonzero(as_tuple=True)[0].item() + 1
        value_ranks.append(rank)

        # Get top 5 predictions
        top_5_ids = sorted_indices[:5]
        top_5_tokens = [tokenizer.decode([tid]) for tid in top_5_ids]
        top_5_probs = [probs[tid].item() for tid in top_5_ids]
        top_5_predictions.append(
            {
                "position": pos,
                "correct_token": tokenizer.decode([correct_token_id]),
                "predictions": list(zip(top_5_tokens, top_5_probs)),
            }
        )

    # Aggregate metrics
    if value_log_probs:
        results["avg_log_prob"] = np.mean(value_log_probs)
        results["total_log_prob"] = np.sum(value_log_probs)
        results["perplexity"] = np.exp(-results["avg_log_prob"])
        results["avg_rank"] = np.mean(value_ranks)
        results["all_rank_1"] = all(r == 1 for r in value_ranks)
        results["any_rank_1"] = any(r == 1 for r in value_ranks)
        results["value_ranks"] = value_ranks
        results["top_5_predictions"] = top_5_predictions[
            :3
        ]  # Save first 3 for inspection

        # Check if the model would generate the correct value
        # For accuracy, we check if all value tokens are rank 1
        results["correct"] = results["all_rank_1"]

    # Also check what model would generate after the key
    # Model was trained to predict value directly after ':'
    key_encoding = tokenizer(key, return_tensors="pt", add_special_tokens=True)
    key_ids = key_encoding["input_ids"].to(device)

    with torch.no_grad():
        k_outputs = model(key_ids, return_dict=True)
        last_logits = k_outputs.logits[0, -1]  # Logits after key:
        top_5_next = torch.topk(last_logits, k=5)

        results["after_key_top_5"] = [
            (tokenizer.decode([idx]), prob.item())
            for idx, prob in zip(
                top_5_next.indices, torch.softmax(top_5_next.values, dim=-1)
            )
        ]

    return results


def main():
    parser = argparse.ArgumentParser(description="Logit-based synthetic QA evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to tokenizer"
    )
    parser.add_argument(
        "--data_path", type=str, default="synthetic_qa_data", help="Path to dataset"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to use"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save results"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("SYNTHETIC QA EVALUATION (LOGIT-BASED)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.data_path}")
    print(f"Split: {args.split}")
    print(f"Num samples: {args.num_samples}")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.device
    )

    # Load dataset
    print(f"\nLoading dataset from {args.data_path}")
    dataset = load_dataset(args.data_path, split=args.split)

    # Sample subset if needed
    total_size = len(dataset)
    if args.num_samples < total_size:
        indices = np.random.choice(total_size, size=args.num_samples, replace=False)
        dataset = dataset.select(indices)
        print(f"Sampled {args.num_samples} examples from {total_size} total")

    # Evaluate
    print(f"\nEvaluating {len(dataset)} examples...")
    all_results = []

    for i in tqdm(range(len(dataset))):
        qa_text = dataset[i]["text"]
        result = evaluate_qa_logits(model, tokenizer, qa_text, args.device)

        if result:
            if "frequency" in dataset.column_names:
                result["frequency"] = dataset[i]["frequency"]
            all_results.append(result)

    # Calculate aggregate metrics
    valid_results = [r for r in all_results if "avg_log_prob" in r]

    if valid_results:
        # Calculate accuracy
        correct_count = sum(1 for r in valid_results if r.get("correct", False))
        accuracy = (correct_count / len(valid_results)) * 100

        metrics = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(valid_results),
            "avg_perplexity": np.mean([r["perplexity"] for r in valid_results]),
            "avg_rank": np.mean([r["avg_rank"] for r in valid_results]),
            "pct_all_rank_1": np.mean([r["all_rank_1"] for r in valid_results]) * 100,
            "pct_any_rank_1": np.mean([r["any_rank_1"] for r in valid_results]) * 100,
            "avg_log_prob": np.mean([r["avg_log_prob"] for r in valid_results]),
        }

        # Group by frequency if available
        if "frequency" in valid_results[0]:
            freq_results = defaultdict(list)
            for r in valid_results:
                freq_results[r["frequency"]].append(r)

            freq_metrics = {}
            for freq, results in freq_results.items():
                freq_correct = sum(1 for r in results if r.get("correct", False))
                freq_metrics[f"freq_{freq}"] = {
                    "count": len(results),
                    "accuracy": (freq_correct / len(results)) * 100 if results else 0,
                    "correct_count": freq_correct,
                    "avg_perplexity": np.mean([r["perplexity"] for r in results]),
                    "avg_rank": np.mean([r["avg_rank"] for r in results]),
                    "pct_all_rank_1": np.mean([r["all_rank_1"] for r in results]) * 100,
                }
            metrics["by_frequency"] = freq_metrics
    else:
        metrics = {}

    # Save results
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    final_results = {
        "config": {
            "model_path": args.model_path,
            "num_samples": len(all_results),
            "seed": args.seed,
        },
        "metrics": metrics,
        "examples": all_results[:10],  # Save first 10 for inspection
    }

    with open(args.output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    if metrics:
        print("Overall Metrics:")
        print(
            f"  Accuracy: {metrics['accuracy']:.1f}% ({metrics['correct_count']}/{metrics['total_count']})"
        )
        print(f"  Average Perplexity: {metrics['avg_perplexity']:.2f}")
        print(f"  Average Rank: {metrics['avg_rank']:.2f}")
        print(f"  % All Tokens Rank 1: {metrics['pct_all_rank_1']:.1f}%")
        print(f"  % Any Token Rank 1: {metrics['pct_any_rank_1']:.1f}%")
        print(f"  Average Log Prob: {metrics['avg_log_prob']:.3f}")

        if "by_frequency" in metrics:
            print("\nMetrics by Frequency:")
            for freq_key, freq_data in sorted(metrics["by_frequency"].items()):
                print(
                    f"  {freq_key}: accuracy={freq_data['accuracy']:.1f}% ({freq_data['correct_count']}/{freq_data['count']}), "
                    f"perplexity={freq_data['avg_perplexity']:.2f}, "
                    f"rank={freq_data['avg_rank']:.2f}, "
                    f"all_rank_1={freq_data['pct_all_rank_1']:.1f}%"
                )

    print(f"\nResults saved to: {args.output_path}")

    # Show a few examples
    if all_results:
        print("\nExample predictions:")
        for i in range(min(3, len(all_results))):
            r = all_results[i]
            print(f"\nKey: {r['key']}")
            print(f"Value: {r['true_value']}")
            if "after_key_top_5" in r:
                print("Top predictions after key:")
                for token, prob in r["after_key_top_5"][:3]:
                    print(f"  '{token}': {prob:.3f}")


if __name__ == "__main__":
    main()
