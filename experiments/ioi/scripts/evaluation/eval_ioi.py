#!/usr/bin/env python3
"""
Evaluation script for Indirect Object Identification (IOI) task.
Based on the mechanistic interpretability probe from Olsson et al.
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

import bento  # noqa - register custom model types


def load_model_and_tokenizer(model_path, device="cuda"):
    """Load the model and tokenizer."""
    print(f"Loading model from {model_path}")
    
    # Load model using AutoModelForCausalLM (will handle bento models automatically)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    # Load tokenizer - try model path first, then fallback to standard
    print(f"Loading tokenizer from {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        # Fallback to a standard tokenizer if model doesn't have one
        tokenizer_path = "fla-hub/transformer-1.3B-100B"
        print(f"Failed to load tokenizer from model, using {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def construct_few_shot_prompt(demonstration_examples, test_sentence, separator="\n\n"):
    """
    Construct a few-shot prompt with demonstration examples.
    
    Args:
        demonstration_examples: List of (sentence, indirect_object) tuples
        test_sentence: The test sentence to predict
        separator: String to separate examples
        
    Returns:
        The constructed few-shot prompt
    """
    prompt_parts = []
    
    # Add instructions
    prompt_parts.append("Task: Identify who receives the object in each sentence (the indirect object).")
    prompt_parts.append("")
    
    # Add demonstration examples with more explicit format
    for sentence, indirect_object in demonstration_examples:
        # Extract the key part to make the pattern clearer
        if " to " in sentence:
            parts = sentence.split(" to ")
            giver_part = parts[0].split()[-1] if parts[0].split() else ""
            receiver = indirect_object
            example_text = f"Sentence: {sentence}\nWho receives the object? {indirect_object}"
        else:
            example_text = f"Sentence: {sentence}\nWho receives the object? {indirect_object}"
        prompt_parts.append(example_text)
    
    # Add test example
    test_text = f"Sentence: {test_sentence}\nWho receives the object?"
    prompt_parts.append(test_text)
    
    return separator.join(prompt_parts)


def extract_ioi_components(example):
    """Extract the indirect object and other components from IOI example."""
    # Get the sentence from the 'ioi_sentences' field
    sentence = example.get('ioi_sentences', '')
    
    # Extract the indirect object from the sentence
    # IOI sentences have patterns like "X gave it to Y" or "X gave a Z to Y"
    indirect_object = None
    
    if " to " in sentence:
        # Split at " to " and get what comes after
        parts = sentence.split(" to ")
        if len(parts) >= 2:
            after_to = parts[-1]  # Get last part in case " to " appears multiple times
            # Extract the first word (the indirect object name)
            words = after_to.split()
            if words:
                indirect_object = words[0].rstrip('.,!?;:')  # Remove punctuation
    
    return sentence, indirect_object


def predict_indirect_object_few_shot(model, tokenizer, prompt, device="cuda"):
    """
    Predict the indirect object using a few-shot prompt.
    The prompt should end with "Who receives the object?" and we extract the generated answer.
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
    
    # Generate the answer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,  # Allow slightly more tokens for few-shot
            temperature=0.0,    # Greedy decoding
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated text
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the last "Who receives the object?")
    search_phrase = "Who receives the object?"
    if search_phrase in generated:
        # Find the last occurrence (which should be our test case)
        answer_start = generated.rfind(search_phrase) + len(search_phrase)
        prediction = generated[answer_start:].strip()
    else:
        # Fallback: just get the generated part
        prompt_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        prediction = generated[prompt_length:].strip()
    
    # Clean up - take only the first word (name)
    prediction = prediction.split()[0] if prediction else ""
    # Remove punctuation and any trailing period from the sentence format
    prediction = prediction.rstrip('.,!?;:\n')
    
    return prediction


def predict_indirect_object(model, tokenizer, sentence, device="cuda"):
    """
    Predict the indirect object in a sentence.
    For IOI task, we typically want to predict the name after "to" in sentences like:
    "Mary gave the ball to John"
    """
    # Find the position of "to" in the sentence
    if " to " not in sentence:
        return None
    
    # Split at "to" and prepare the prompt
    prefix = sentence.split(" to ")[0] + " to"
    
    # Tokenize the prefix
    inputs = tokenizer(prefix, return_tensors="pt", padding=False).to(device)
    
    # Generate the next token(s) - typically a name
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,  # Names are usually 1-2 tokens
            temperature=0.0,   # Greedy decoding
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated text
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prefix)
    if prefix in generated:
        prediction = generated[len(prefix):].strip()
    else:
        prediction = generated.strip()
    
    # Clean up - take only the first word (name)
    prediction = prediction.split()[0] if prediction else ""
    # Remove punctuation
    prediction = prediction.rstrip('.,!?;:')
    
    return prediction


def evaluate_batch(model, tokenizer, batch_examples, device="cuda", num_shots=0, shot_separator="\n\n", all_examples=None):
    """Evaluate a batch of IOI examples."""
    results = []
    
    for example in batch_examples:
        sentence, true_indirect_object = extract_ioi_components(example)
        
        if not sentence or true_indirect_object is None:
            continue
        
        if num_shots > 0:
            # Few-shot evaluation
            if all_examples is None:
                raise ValueError("all_examples must be provided for few-shot evaluation")
            
            # Sample demonstration examples (excluding the current test example)
            demonstration_pool = []
            for demo_ex in all_examples:
                demo_sentence, demo_io = extract_ioi_components(demo_ex)
                if demo_sentence and demo_io and demo_sentence != sentence:
                    demonstration_pool.append((demo_sentence, demo_io))
            
            # Randomly sample num_shots examples
            if len(demonstration_pool) >= num_shots:
                import random
                demonstration_examples = random.sample(demonstration_pool, num_shots)
            else:
                # If we don't have enough examples, use all available
                demonstration_examples = demonstration_pool
            
            # Construct few-shot prompt
            prompt = construct_few_shot_prompt(demonstration_examples, sentence, shot_separator)
            
            # Get model's prediction using few-shot prompt
            predicted_io = predict_indirect_object_few_shot(model, tokenizer, prompt, device)
        else:
            # Zero-shot evaluation (original behavior)
            predicted_io = predict_indirect_object(model, tokenizer, sentence, device)
        
        if predicted_io is None:
            continue
        
        # Check if prediction matches
        correct = predicted_io.lower() == true_indirect_object.lower()
        
        results.append({
            "sentence": sentence,
            "true_indirect_object": true_indirect_object,
            "predicted_indirect_object": predicted_io,
            "correct": correct,
        })
    
    return results


def analyze_results_by_pattern(results):
    """Analyze results by different IOI patterns if available."""
    pattern_results = defaultdict(list)
    
    for result in results:
        sentence = result["sentence"].lower()
        
        # Categorize by verb used
        if " gave " in sentence:
            pattern_results["gave"].append(result["correct"])
        elif " handed " in sentence:
            pattern_results["handed"].append(result["correct"])
        elif " passed " in sentence:
            pattern_results["passed"].append(result["correct"])
        else:
            pattern_results["other"].append(result["correct"])
    
    pattern_accuracies = {}
    for pattern, corrects in pattern_results.items():
        if corrects:
            pattern_accuracies[f"pattern_{pattern}"] = np.mean(corrects)
    
    return pattern_accuracies


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on IOI task")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="fahamu/ioi",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=200,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save results"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num_shots", 
        type=int, 
        default=0,
        help="Number of few-shot examples (0 for zero-shot)"
    )
    parser.add_argument(
        "--shot_separator", 
        type=str, 
        default="\n\n",
        help="Separator between few-shot examples"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("INDIRECT OBJECT IDENTIFICATION (IOI) EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Num samples: {args.num_samples}")
    print(f"Evaluation mode: {'Few-shot (' + str(args.num_shots) + ' shots)' if args.num_shots > 0 else 'Zero-shot'}")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset_name}")
    # The IOI dataset only has a 'train' split
    dataset = load_dataset(args.dataset_name, split="train", streaming=True)
    
    # Take only the specified number of samples
    print(f"\nEvaluating {args.num_samples} examples...")
    all_results = []
    
    if args.num_shots > 0:
        # For few-shot evaluation, we need to collect all examples first
        print(f"Few-shot evaluation with {args.num_shots} demonstration examples per test")
        print("Loading dataset examples for few-shot sampling...")
        
        # Collect examples for few-shot pool (we'll collect more than needed)
        few_shot_pool_size = args.num_samples + args.num_shots * 10  # Extra examples for demonstrations
        dataset_examples = []
        
        for i, example in enumerate(dataset):
            dataset_examples.append(example)
            if i >= few_shot_pool_size:
                break
        
        print(f"Collected {len(dataset_examples)} examples for few-shot pool")
        
        # Now evaluate with few-shot
        examples_processed = 0
        pbar = tqdm(total=args.num_samples)
        
        batch_examples = []
        for i, example in enumerate(dataset_examples[:args.num_samples]):
            batch_examples.append(example)
            
            if len(batch_examples) >= args.batch_size:
                results = evaluate_batch(
                    model, tokenizer, batch_examples, args.device,
                    num_shots=args.num_shots, 
                    shot_separator=args.shot_separator,
                    all_examples=dataset_examples
                )
                all_results.extend(results)
                examples_processed += len(batch_examples)
                pbar.update(len(batch_examples))
                batch_examples = []
        
        # Process remaining examples
        if batch_examples:
            results = evaluate_batch(
                model, tokenizer, batch_examples, args.device,
                num_shots=args.num_shots,
                shot_separator=args.shot_separator,
                all_examples=dataset_examples
            )
            all_results.extend(results)
            pbar.update(len(batch_examples))
        
        pbar.close()
    else:
        # Original zero-shot evaluation
        examples_processed = 0
        pbar = tqdm(total=args.num_samples)
        
        batch_examples = []
        for example in dataset:
            batch_examples.append(example)
            
            if len(batch_examples) >= args.batch_size:
                results = evaluate_batch(model, tokenizer, batch_examples, args.device)
                all_results.extend(results)
                examples_processed += len(batch_examples)
                pbar.update(len(batch_examples))
                batch_examples = []
                
                if examples_processed >= args.num_samples:
                    break
        
        # Process remaining examples
        if batch_examples and examples_processed < args.num_samples:
            results = evaluate_batch(model, tokenizer, batch_examples, args.device)
            all_results.extend(results)
            pbar.update(len(batch_examples))
        
        pbar.close()
    
    # Calculate metrics
    if all_results:
        correct_predictions = [r["correct"] for r in all_results]
        accuracy = np.mean(correct_predictions)
        
        # Analyze by pattern
        pattern_accuracies = analyze_results_by_pattern(all_results)
    else:
        accuracy = 0.0
        pattern_accuracies = {}
    
    # Prepare final results
    final_results = {
        "config": {
            "model_path": args.model_path,
            "dataset_name": args.dataset_name,
            "num_samples": len(all_results),
            "seed": args.seed,
            "num_shots": args.num_shots,
            "shot_separator": args.shot_separator,
            "evaluation_mode": "few-shot" if args.num_shots > 0 else "zero-shot",
        },
        "results": {
            "overall_accuracy": float(accuracy),
            "num_correct": int(sum([r["correct"] for r in all_results])),
            "num_total": len(all_results),
            **pattern_accuracies,
        },
        "sample_predictions": all_results[:20],  # Save first 20 for inspection
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
    print(f"Overall Accuracy: {accuracy:.4f} ({int(accuracy * len(all_results))}/{len(all_results)})")
    
    if pattern_accuracies:
        print("\nAccuracy by Pattern:")
        for pattern_key, acc in sorted(pattern_accuracies.items()):
            print(f"  {pattern_key}: {acc:.4f}")
    
    print(f"\nResults saved to: {args.output_path}")
    
    # Show a few examples
    print("\nExample predictions:")
    for i in range(min(5, len(all_results))):
        r = all_results[i]
        print(f"\nSentence: {r['sentence']}")
        print(f"True IO: {r['true_indirect_object']}")
        print(f"Predicted IO: {r['predicted_indirect_object']}")
        print(f"Correct: {r['correct']}")


if __name__ == "__main__":
    main()