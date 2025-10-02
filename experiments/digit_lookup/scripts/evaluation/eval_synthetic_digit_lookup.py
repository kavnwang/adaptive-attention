#!/usr/bin/env python3
"""
Evaluation script for synthetic digit lookup dataset.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict


def load_model_and_tokenizer(model_path, device="cuda"):
    """Load the model and tokenizer."""
    print(f"Loading model from {model_path}")

    # Add bento path for custom model
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "3rdparty/bento"))

    # Import the custom model class
    try:
        from bento.models.memory.modeling_memory import MemoryForCausalLM

        # Load model using the custom class
        model = MemoryForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
    except ImportError as e:
        print(f"Failed to import MemoryForCausalLM: {e}")
        # Fallback to standard loading
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

    model.eval()

    # Load tokenizer (always use the same tokenizer)
    tokenizer_path = "fla-hub/transformer-1.3B-100B"
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def extract_digit_lookup_parts(text):
    """Extract key and value from the text."""
    # Expected format: "key:value "
    if ":" in text:
        parts = text.split(":", 1)
        key = parts[0].strip()
        value = parts[1].strip() if len(parts) > 1 else ""
        return key + ":", value
    return text, ""


def compute_expected_value(key):
    """
    Apply the transformation rule to compute expected value.
    Value[i] = Key[Key[i]-1] where indexing is 1-based for digit values.
    """
    value = []
    n = len(key)

    for i in range(n):
        # Get the digit at position i in the key
        digit_at_i = int(key[i])

        # Use that digit to index into the key (subtract 1 for 0-based indexing)
        # digit_at_i is in range [1, n], so digit_at_i - 1 is in range [0, n-1]
        index = digit_at_i - 1

        # Get the digit at that index in the key
        value_digit = key[index]

        # Add to our value
        value.append(value_digit)

    return "".join(value)


def evaluate_batch(model, tokenizer, batch_texts, device="cuda"):
    """Evaluate a batch of digit lookup examples."""
    results = []

    for text in batch_texts:
        question, true_answer = extract_digit_lookup_parts(text)

        # Tokenize the question (key with colon)
        inputs = tokenizer(question, return_tensors="pt", padding=False).to(device)

        # Since each digit is 2 tokens (space + digit) when tokenized independently,
        # but we don't want the initial space when continuing from ":",
        # we need to subtract 1 from the answer length
        answer_tokens = tokenizer(
            true_answer, add_special_tokens=False, return_tensors="pt"
        )
        answer_length = answer_tokens["input_ids"].shape[1] - 1

        # Generate answer with exact length
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=answer_length,
                min_new_tokens=answer_length,
                temperature=0.0,  # Greedy decoding
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode the generated text
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (after the key and colon)
        if question in generated:
            pred_answer = generated[len(question) :].strip()
        else:
            pred_answer = generated.strip()

        # Extract the key for computing expected value
        key_only = question.rstrip(":")
        expected_value = compute_expected_value(key_only)

        results.append(
            {
                "key": key_only,
                "true_value": true_answer,
                "expected_value": expected_value,
                "pred_value": pred_answer,
                "exact_match": pred_answer == true_answer,
                "matches_expected": true_answer == expected_value,
                "text": text,
            }
        )

    return results


def analyze_position_accuracy(results):
    """Analyze accuracy by position in the value."""
    if not results:
        return {}

    # Assuming all values have the same length
    value_length = len(results[0]["true_value"])
    position_correct = defaultdict(int)
    position_total = defaultdict(int)

    for result in results:
        true_val = result["true_value"]
        pred_val = result["pred_value"]

        # Pad prediction if shorter
        if len(pred_val) < len(true_val):
            pred_val = pred_val + " " * (len(true_val) - len(pred_val))

        for i in range(min(len(true_val), len(pred_val))):
            position_total[i] += 1
            if i < len(pred_val) and true_val[i] == pred_val[i]:
                position_correct[i] += 1

    position_accuracy = {}
    for pos in range(value_length):
        if position_total[pos] > 0:
            position_accuracy[f"pos_{pos}"] = (
                position_correct[pos] / position_total[pos]
            )

    return position_accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic digit lookup dataset"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="synthetic_digit_lookup_data_20000_special_tests",
        help="Path to dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate (-1 for all)",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save results"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("SYNTHETIC DIGIT LOOKUP EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.data_path}")
    print(f"Num samples: {args.num_samples if args.num_samples > 0 else 'all'}")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    # Results dictionary
    all_results = {"known_pairs": [], "novel_pairs": []}

    # Evaluate on both test sets
    for test_type in ["test_known_pairs", "test_novel_pairs"]:
        print(f"\n{'=' * 60}")
        print(f"Evaluating {test_type}")
        print(f"{'=' * 60}")

        # Load dataset
        dataset_path = f"{args.data_path}/{test_type}"
        print(f"Loading dataset from {dataset_path}")
        dataset = load_dataset(
            dataset_path, split="train"
        )  # Parquet files default to 'train' split

        # Sample subset if needed
        total_size = len(dataset)
        if args.num_samples > 0 and args.num_samples < total_size:
            indices = np.random.choice(total_size, size=args.num_samples, replace=False)
            dataset = dataset.select(indices)
            print(f"Sampled {args.num_samples} examples from {total_size} total")
        else:
            print(f"Using all {total_size} examples")

        # Evaluate
        print(f"\nEvaluating {len(dataset)} examples...")
        test_results = []

        for i in tqdm(range(0, len(dataset), args.batch_size)):
            batch = dataset[i : i + args.batch_size]
            batch_texts = (
                batch["text"] if isinstance(batch["text"], list) else [batch["text"]]
            )
            results = evaluate_batch(model, tokenizer, batch_texts, args.device)
            test_results.extend(results)

        # Store results
        result_key = "known_pairs" if "known" in test_type else "novel_pairs"
        all_results[result_key] = test_results

        # Calculate metrics
        exact_matches = [r["exact_match"] for r in test_results]
        accuracy = np.mean(exact_matches)

        # Position-wise accuracy
        position_accuracy = analyze_position_accuracy(test_results)

        # Print summary
        print(f"\n{test_type} Results:")
        print(
            f"Accuracy: {accuracy:.4f} ({int(accuracy * len(test_results))}/{len(test_results)})"
        )

        if position_accuracy:
            print("\nPosition-wise accuracy:")
            for pos, acc in sorted(position_accuracy.items()):
                print(f"  {pos}: {acc:.4f}")

        # Show a few examples
        print(f"\nExample predictions from {test_type}:")
        for i in range(min(5, len(test_results))):
            r = test_results[i]
            print(f"\nKey: {r['key']}")
            print(f"True: {r['true_value']}")
            print(f"Pred: {r['pred_value']}")
            print(f"Match: {r['exact_match']}")

    # Prepare final results
    known_accuracy = np.mean([r["exact_match"] for r in all_results["known_pairs"]])
    novel_accuracy = np.mean([r["exact_match"] for r in all_results["novel_pairs"]])

    final_results = {
        "config": {
            "model_path": args.model_path,
            "num_known_samples": len(all_results["known_pairs"]),
            "num_novel_samples": len(all_results["novel_pairs"]),
            "seed": args.seed,
        },
        "results": {
            "known_pairs_accuracy": float(known_accuracy),
            "novel_pairs_accuracy": float(novel_accuracy),
            "generalization_gap": float(known_accuracy - novel_accuracy),
        },
        "position_analysis": {
            "known_pairs": analyze_position_accuracy(all_results["known_pairs"]),
            "novel_pairs": analyze_position_accuracy(all_results["novel_pairs"]),
        },
        "sample_predictions": {
            "known_pairs": all_results["known_pairs"][:10],
            "novel_pairs": all_results["novel_pairs"][:10],
        },
    }

    # Save results
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("OVERALL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Known Pairs Accuracy: {known_accuracy:.4f}")
    print(f"Novel Pairs Accuracy: {novel_accuracy:.4f}")
    print(f"Generalization Gap: {known_accuracy - novel_accuracy:.4f}")
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()
