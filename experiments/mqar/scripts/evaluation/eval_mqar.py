#!/usr/bin/env python3
"""
Evaluation script for MQAR (Multi-Query Associative Recall) synthetic dataset.

This evaluates models on their ability to recall values associated with keys.
In MQAR, each example has:
1. Storage phase: K1 V1 K2 V2 ... Kn Vn
2. Query phase: Keys in power-law order, model must predict corresponding values
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
from typing import List, Dict, Tuple

import bento  # noqa - register custom model types


def load_model_and_tokenizer(model_path, device="cuda"):
    """Load the model and tokenizer."""
    print(f"Loading model from {model_path}")
    
    # Load model using AutoModelForCausalLM (works with bento models)
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


def extract_mqar_structure(tokens: List[int], num_kv_pairs: int) -> Dict:
    """Extract the MQAR structure from a tokenized example."""
    if len(tokens) < 4 * num_kv_pairs:
        return None
    
    # Extract storage phase (first 2*num_kv_pairs tokens)
    storage_phase = tokens[:2*num_kv_pairs]
    keys = []
    values = []
    for i in range(0, 2*num_kv_pairs, 2):
        keys.append(storage_phase[i])
        values.append(storage_phase[i+1])
    
    # Extract query phase (next 2*num_kv_pairs tokens)
    query_phase = tokens[2*num_kv_pairs:4*num_kv_pairs]
    query_keys = []
    query_values = []
    for i in range(0, 2*num_kv_pairs, 2):
        query_keys.append(query_phase[i])
        if i+1 < len(query_phase):
            query_values.append(query_phase[i+1])
    
    return {
        "keys": keys,
        "values": values,
        "query_keys": query_keys,
        "query_values": query_values,
        "kv_dict": dict(zip(keys, values))
    }


def evaluate_mqar_example(
    model, 
    tokenizer, 
    text: str, 
    num_kv_pairs: int,
    device="cuda"
) -> Dict:
    """Evaluate a single MQAR example."""
    
    # Tokenize the full text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Extract MQAR structure
    structure = extract_mqar_structure(tokens, num_kv_pairs)
    if structure is None:
        return {"error": "Could not parse MQAR structure"}
    
    results = []
    
    # For each query position, predict the value
    for query_idx in range(num_kv_pairs):
        # Context includes storage phase + queries up to this point
        context_len = 2*num_kv_pairs + 2*query_idx + 1  # +1 to include current query key
        context_tokens = tokens[:context_len]
        
        # Convert to tensor
        inputs = torch.tensor([context_tokens], device=device)
        
        # Get model predictions for next token
        with torch.no_grad():
            outputs = model(inputs)
            logits = outputs.logits[0, -1, :]  # Get logits for last position
            predicted_token = torch.argmax(logits).item()
        
        # Get true value for this query
        query_key = structure["query_keys"][query_idx]
        true_value = structure["kv_dict"].get(query_key, -1)
        
        results.append({
            "position": query_idx,
            "query_key": query_key,
            "true_value": true_value,
            "predicted_value": predicted_token,
            "correct": predicted_token == true_value
        })
    
    return {
        "results": results,
        "accuracy": np.mean([r["correct"] for r in results]),
        "num_correct": sum(r["correct"] for r in results),
        "num_total": len(results)
    }


def main():
    parser = argparse.ArgumentParser(description="MQAR evaluation script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--data_path", type=str, default="mqar_b16_kv1_seq8192_500samples", 
        help="Path to MQAR dataset"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to use"
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--num_kv_pairs", type=int, default=1, help="Number of KV pairs in each example"
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
    print("MQAR EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.data_path}")
    print(f"Split: {args.split}")
    print(f"Num samples: {args.num_samples}")
    print(f"Num KV pairs: {args.num_kv_pairs}")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

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
    position_accuracies = defaultdict(list)

    for i in tqdm(range(len(dataset))):
        text = dataset[i]["text"]
        
        # Evaluate this example
        result = evaluate_mqar_example(
            model, tokenizer, text, args.num_kv_pairs, args.device
        )
        
        if "error" not in result:
            all_results.append(result)
            
            # Track accuracy by query position
            for r in result["results"]:
                position_accuracies[r["position"]].append(r["correct"])

    # Calculate overall metrics
    if all_results:
        overall_accuracy = np.mean([r["accuracy"] for r in all_results])
        total_correct = sum(r["num_correct"] for r in all_results)
        total_queries = sum(r["num_total"] for r in all_results)
        
        # Calculate position-wise accuracies
        position_acc_dict = {}
        for pos, correct_list in position_accuracies.items():
            position_acc_dict[f"position_{pos}"] = float(np.mean(correct_list))
    else:
        overall_accuracy = 0.0
        total_correct = 0
        total_queries = 0
        position_acc_dict = {}

    # Prepare final results
    final_results = {
        "config": {
            "model_path": args.model_path,
            "num_samples": len(all_results),
            "num_kv_pairs": args.num_kv_pairs,
            "seed": args.seed,
        },
        "results": {
            "overall_accuracy": float(overall_accuracy),
            "total_correct": total_correct,
            "total_queries": total_queries,
            **position_acc_dict,
        },
        "detailed_results": all_results[:5],  # Save first 5 for inspection
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
        f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_queries})"
    )
    
    if position_acc_dict:
        print("\nAccuracy by Query Position:")
        for pos_key in sorted(position_acc_dict.keys()):
            print(f"  {pos_key}: {position_acc_dict[pos_key]:.4f}")

    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()