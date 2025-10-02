#!/usr/bin/env python3
"""Run ablation analysis for all models (200, 2K, 10K, 50K)"""

import subprocess
import os
import sys

# Model configurations
# Note: Only 50K model exists, so commenting out missing models
models = [
    # {
    #     "name": "200",
    #     "model_path": "../../exp/memory_2layer_synthetic_qa_200",
    #     "tokenizer_path": "../../exp/memory_2layer_synthetic_qa_200",
    #     "input_file": "../old_results/memory_key_activations/memory_key_activations_200.json",
    #     "output_file": "../results/ablation_results/200/ablation_results_200_new.json",
    # },
    # {
    #     "name": "2K",
    #     "model_path": "../../exp/memory_2layer_synthetic_qa_2K",
    #     "tokenizer_path": "../../exp/memory_2layer_synthetic_qa_2K",
    #     "input_file": "../old_results/memory_key_activations/memory_key_activations_2K.json",
    #     "output_file": "../results/ablation_results/2K/ablation_results_2K_new.json",
    # },
    # {
    #     "name": "10K",
    #     "model_path": "../../exp/memory_2layer_synthetic_qa_10K",
    #     "tokenizer_path": "../../exp/memory_2layer_synthetic_qa_10K",
    #     "input_file": "../old_results/memory_key_activations/memory_key_activations_10K.json",
    #     "output_file": "../results/ablation_results/10K/ablation_results_10K_new.json",
    # },
    {
        "name": "50K",
        "model_path": "../../exp/memory_2layer_synthetic_qa_50K",
        "tokenizer_path": "../../exp/memory_2layer_synthetic_qa_50K",
        "input_file": "../old_results/memory_key_activations/memory_key_activations_50K.json",
        "output_file": "../results/ablation_results/50K/ablation_results_50K_new.json",
    },
]

# Parse command line arguments
# Zero ablation is now on by default
zero_ablation = "--no_zero_ablation" not in sys.argv

if zero_ablation:
    print("Running with zero ablation mode enabled (default)")
else:
    print("Running with standard ablation mode (zero ablation disabled)")

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(
    os.path.dirname(script_dir)
)  # Go up 2 levels to LLMonade_interp

# Convert relative paths to absolute paths
for model in models:
    model["model_path"] = os.path.join(
        base_dir, model["model_path"].replace("../../", "")
    )
    model["tokenizer_path"] = os.path.join(
        base_dir, model["tokenizer_path"].replace("../../", "")
    )
    model["input_file"] = os.path.join(script_dir, model["input_file"])
    model["output_file"] = os.path.join(script_dir, model["output_file"])

# Run ablation for each model
for model in models:
    print(f"\n{'=' * 60}")
    print(f"Running ablation analysis for {model['name']} model...")
    print(f"{'=' * 60}")

    cmd = [
        "python",
        os.path.join(script_dir, "ablate_memory_keys.py"),
        "--model_path",
        model["model_path"],
        "--tokenizer_path",
        model["tokenizer_path"],
        "--input_file",
        model["input_file"],
        "--output_file",
        model["output_file"],
        "--device",
        "cuda",
    ]

    if not zero_ablation:
        cmd.append("--no_zero_ablation")

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Ablation analysis for {model['name']} model completed!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running ablation for {model['name']} model: {e}")
        continue

print("\n" + "=" * 60)
print("All ablation analyses completed!")
print("=" * 60)
