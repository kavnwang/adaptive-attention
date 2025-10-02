#!/usr/bin/env python3
"""
Simple direct evaluation script for synthetic QA without lm-eval harness.
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


def load_model_and_tokenizer(model_path, tokenizer_path, device="cuda"):
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

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def extract_qa_parts(text):
    """Extract question and answer from the text."""
    # Expected format: "Q: Where is <entity>? <answer>"
    if "?" in text:
        parts = text.split("?", 1)
        # Don't add space - model expects no space after '?'
        question = parts[0] + "?"  # No trailing space
        answer = parts[1].strip() if len(parts) > 1 else ""
        # Clean answer
        answer = answer.lstrip("-: ")
        return question, answer
    return text, ""


def evaluate_batch(model, tokenizer, batch_texts, device="cuda"):
    """Evaluate a batch of QA examples."""
    results = []

    for text in batch_texts:
        question, true_answer = extract_qa_parts(text)

        # Tokenize the question
        inputs = tokenizer(question, return_tensors="pt", padding=False).to(device)

        # Generate answer
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.0,  # Greedy decoding
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode the generated text
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (after the question)
        if question in generated:
            pred_answer = generated[len(question) :].strip()
        else:
            pred_answer = generated.strip()

        # Clean up the predicted answer
        pred_answer = pred_answer.split("\n")[0].strip()  # Take first line only

        results.append(
            {
                "question": question.strip(),
                "true_answer": true_answer,
                "pred_answer": pred_answer,
                "exact_match": pred_answer.lower() == true_answer.lower(),
                "text": text,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Simple synthetic QA evaluation")
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
        "--batch_size", type=int, default=1, help="Batch size (1 for now)"
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
    print("SYNTHETIC QA EVALUATION (SIMPLE)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
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

    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset[i : i + args.batch_size]
        batch_texts = (
            batch["text"] if isinstance(batch["text"], list) else [batch["text"]]
        )
        results = evaluate_batch(model, tokenizer, batch_texts, args.device)
        all_results.extend(results)

    # Calculate metrics
    exact_matches = [r["exact_match"] for r in all_results]
    accuracy = np.mean(exact_matches)

    # Group by frequency if available
    if "frequency" in dataset.column_names:
        freq_results = defaultdict(list)
        for i, result in enumerate(all_results):
            freq = dataset[i]["frequency"]
            freq_results[freq].append(result["exact_match"])

        freq_accuracies = {}
        for freq, matches in freq_results.items():
            freq_accuracies[f"freq_{freq}"] = np.mean(matches)
    else:
        freq_accuracies = {}

    # Prepare final results
    final_results = {
        "config": {
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path,
            "num_samples": len(all_results),
            "seed": args.seed,
        },
        "results": {
            "overall_accuracy": float(accuracy),
            "exact_match": float(accuracy),
            **freq_accuracies,
        },
        "detailed_results": all_results[:10],  # Save first 10 for inspection
    }

    # Save results
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(
        f"Overall Accuracy: {accuracy:.4f} ({int(accuracy * len(all_results))}/{len(all_results)})"
    )

    if freq_accuracies:
        print("\nAccuracy by Frequency:")
        for freq_key, acc in sorted(freq_accuracies.items()):
            print(f"  {freq_key}: {acc:.4f}")

    print(f"\nResults saved to: {args.output_path}")

    # Show a few examples
    print("\nExample predictions:")
    for i in range(min(5, len(all_results))):
        r = all_results[i]
        print(f"\nQ: {r['question']}")
        print(f"True: {r['true_answer']}")
        print(f"Pred: {r['pred_answer']}")
        print(f"Match: {r['exact_match']}")


if __name__ == "__main__":
    main()
