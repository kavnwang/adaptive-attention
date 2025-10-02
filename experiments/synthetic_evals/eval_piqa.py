#!/usr/bin/env python3
"""
Evaluation script for PIQA (Physical Interaction QA) dataset.

This script evaluates transformer models on the PIQA test set by computing
the likelihood of each answer choice and selecting the one with higher probability.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from typing import Dict, List, Tuple

import bento  # noqa - register custom model types


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load the model and tokenizer."""
    print(f"Loading model from {model_path}")
    
    # Load model using AutoModelForCausalLM (works with transformer models)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    # Load tokenizer (use the standard tokenizer for transformer models)
    tokenizer_path = "fla-hub/transformer-1.3B-100B"
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def compute_choice_logprobs(
    model, tokenizer, question: str, choice: str, device: str = "cuda"
) -> float:
    """
    Compute the log probability of a choice given a question.
    
    Format: "Question: {question}\nAnswer: {choice}"
    """
    # Format the prompt
    prompt = f"Question: {question}\nAnswer: {choice}"
    
    # Tokenize the full sequence
    inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
    input_ids = inputs["input_ids"]
    
    # Find where the answer starts (after "Answer: ")
    prompt_without_choice = f"Question: {question}\nAnswer: "
    prompt_tokens = tokenizer(prompt_without_choice, return_tensors="pt", padding=False)
    prompt_length = prompt_tokens["input_ids"].shape[1]
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
    
    # Compute log probabilities for the answer tokens
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, prompt_length-1:-1, :].contiguous()
    shift_labels = input_ids[:, prompt_length:].contiguous()
    
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probabilities of the actual tokens
    gathered_log_probs = torch.gather(
        log_probs, 
        dim=-1, 
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Sum log probabilities for the entire answer
    total_log_prob = gathered_log_probs.sum().item()
    
    return total_log_prob


def evaluate_piqa_example(
    model, tokenizer, example: Dict, device: str = "cuda"
) -> Dict:
    """Evaluate a single PIQA example."""
    goal = example["goal"]
    sol1 = example["sol1"]
    sol2 = example["sol2"]
    label = example["label"]
    
    # Compute log probabilities for both choices
    logprob1 = compute_choice_logprobs(model, tokenizer, goal, sol1, device)
    logprob2 = compute_choice_logprobs(model, tokenizer, goal, sol2, device)
    
    # Predict the choice with higher log probability
    predicted_label = 0 if logprob1 > logprob2 else 1
    correct = predicted_label == label
    
    # Compute normalized probabilities
    probs = F.softmax(torch.tensor([logprob1, logprob2]), dim=0).tolist()
    
    return {
        "goal": goal,
        "sol1": sol1,
        "sol2": sol2,
        "true_label": label,
        "predicted_label": predicted_label,
        "correct": correct,
        "logprob_sol1": logprob1,
        "logprob_sol2": logprob2,
        "prob_sol1": probs[0],
        "prob_sol2": probs[1],
        "confidence": max(probs),
    }


def evaluate_dataset(
    model, tokenizer, dataset, num_samples: int = -1, device: str = "cuda"
) -> List[Dict]:
    """Evaluate the model on the PIQA dataset."""
    results = []
    
    # Determine number of samples to evaluate
    total_samples = len(dataset)
    if num_samples > 0:
        num_samples = min(num_samples, total_samples)
    else:
        num_samples = total_samples
    
    print(f"Evaluating {num_samples} examples...")
    
    # Evaluate each example
    for i in tqdm(range(num_samples)):
        example = dataset[i]
        result = evaluate_piqa_example(model, tokenizer, example, device)
        results.append(result)
    
    return results


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute evaluation metrics from results."""
    correct_predictions = [r["correct"] for r in results]
    accuracy = np.mean(correct_predictions)
    
    # Compute average confidence
    confidences = [r["confidence"] for r in results]
    avg_confidence = np.mean(confidences)
    
    # Compute accuracy by confidence bins
    confidence_bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    accuracy_by_confidence = {}
    
    for low, high in confidence_bins:
        bin_results = [r for r in results if low <= r["confidence"] < high]
        if bin_results:
            bin_accuracy = np.mean([r["correct"] for r in bin_results])
            accuracy_by_confidence[f"{low:.1f}-{high:.1f}"] = {
                "accuracy": float(bin_accuracy),
                "count": len(bin_results),
            }
    
    return {
        "accuracy": float(accuracy),
        "num_correct": int(sum(correct_predictions)),
        "num_total": len(results),
        "avg_confidence": float(avg_confidence),
        "accuracy_by_confidence": accuracy_by_confidence,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on PIQA dataset")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save evaluation results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate (-1 for all)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 60)
    print("PIQA EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Num samples: {args.num_samples if args.num_samples > 0 else 'all'}")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    
    # Load PIQA dataset (validation split is used as test set)
    print("\nLoading PIQA validation set...")
    dataset = load_dataset("piqa", split="train")
    print(f"Dataset size: {len(dataset)} examples")
    
    # Evaluate
    results = evaluate_dataset(model, tokenizer, dataset, args.num_samples, args.device)
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['num_correct']}/{metrics['num_total']})")
    print(f"Average confidence: {metrics['avg_confidence']:.4f}")
    
    if metrics["accuracy_by_confidence"]:
        print("\nAccuracy by confidence level:")
        for bin_range, bin_stats in sorted(metrics["accuracy_by_confidence"].items()):
            print(
                f"  {bin_range}: {bin_stats['accuracy']:.4f} "
                f"({bin_stats['count']} examples)"
            )
    
    # Show some example predictions
    print("\nExample predictions:")
    for i in range(min(5, len(results))):
        r = results[i]
        print(f"\nExample {i+1}:")
        print(f"  Question: {r['goal'][:80]}...")
        print(f"  Solution 1: {r['sol1'][:80]}...")
        print(f"  Solution 2: {r['sol2'][:80]}...")
        print(f"  True answer: Solution {r['true_label'] + 1}")
        print(f"  Predicted: Solution {r['predicted_label'] + 1}")
        print(f"  Correct: {'✓' if r['correct'] else '✗'}")
        print(f"  Confidence: {r['confidence']:.3f}")
    
    # Prepare final results
    final_results = {
        "config": {
            "model_path": args.model_path,
            "num_samples": len(results),
            "seed": args.seed,
            "device": args.device,
        },
        "metrics": metrics,
        "sample_predictions": results[:100],  # Save first 100 predictions
    }
    
    # Save results
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()