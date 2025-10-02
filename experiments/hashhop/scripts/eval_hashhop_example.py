#!/usr/bin/env python3
"""
Example script for evaluating a model on the hash-hop dataset
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_hashhop(model_name, dataset_path="hashhop_eval_340m/medium_2hop.json"):
    """
    Example evaluation function for hash-hop
    """
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    print(f"Evaluating on {len(dataset)} samples from {dataset_path}")

    correct = 0
    total_queries = 0

    for i, sample in enumerate(dataset):
        print(f"\nSample {i + 1}/{len(dataset)}")

        # Prepare input - just the prompt
        input_text = sample["prompt"] + "\n\nCOMPLETION:\n"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,  # Low temperature for deterministic outputs
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        # Parse expected answers from completion
        expected_answers = {}
        for line in sample["completion"].split("\n"):
            if " = '" in line and line.endswith("'"):
                key, value = line.split(" = '")
                value = value.rstrip("'")
                expected_answers[key] = value

        # Check accuracy
        sample_correct = 0
        for key, expected_value in expected_answers.items():
            if f"{key} = '{expected_value}'" in generated:
                sample_correct += 1
                correct += 1
            total_queries += 1

        print(f"  Correct: {sample_correct}/{len(expected_answers)}")

        if i == 0:  # Show first example
            print(f"  Expected: {list(expected_answers.items())[:3]}")
            print(f"  Generated preview: {generated[:200]}...")

    accuracy = correct / total_queries if total_queries > 0 else 0
    print(f"\nOverall Accuracy: {correct}/{total_queries} = {accuracy:.2%}")

    return {"accuracy": accuracy, "correct": correct, "total": total_queries}


if __name__ == "__main__":
    # Example usage - replace with your model
    # results = evaluate_hashhop("your-model-name-here", "hashhop_eval_340m/easy_1hop.json")
    print("Example evaluation script created.")
    print("To use: evaluate_hashhop('model-name', 'dataset-path')")
