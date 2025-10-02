#!/usr/bin/env python3
"""
Run hash-hop evaluation for the memory_mlp_router_340M_24K model
"""

import subprocess
import sys
import json
from pathlib import Path

# Model configuration for memory_mlp_router_340M_24K
MODEL_PATH = "../../exp/memory_mlp_router_340M_24K"
TOKENIZER_PATH = MODEL_PATH  # Usually same as model path
DEVICE = "cuda"


def run_evaluation(dataset_name, output_name=None):
    """Run evaluation on a specific dataset"""
    if output_name is None:
        output_name = dataset_name

    dataset_path = f"../datasets/hashhop_eval_340m/{dataset_name}.json"
    output_dir = f"../results/{output_name}"

    cmd = [
        sys.executable,
        "analyze_hashhop_routing.py",
        "--model_path",
        MODEL_PATH,
        "--tokenizer_path",
        TOKENIZER_PATH,
        "--dataset_path",
        dataset_path,
        "--output_dir",
        output_dir,
        "--device",
        DEVICE,
    ]

    print(f"\nRunning evaluation on {dataset_name}...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {dataset_name}:")
            print(result.stderr)
            return None
        else:
            print(f"Completed {dataset_name}")
            # Try to extract accuracy from output
            for line in result.stdout.split("\n"):
                if "Overall accuracy:" in line:
                    print(f"  {line.strip()}")
            return output_dir
    except Exception as e:
        print(f"Exception running {dataset_name}: {e}")
        return None


def main():
    print(f"Evaluating Hash-Hop on model: {MODEL_PATH}")

    # Create main results directory
    Path("hashhop_results_340m").mkdir(exist_ok=True)

    # List of datasets to evaluate
    datasets = [
        "easy_1hop",
        "medium_2hop",
        "hard_3hop",
        "challenge_4hop",
        "medium_2hop_cot",
        "hard_3hop_cot",
    ]

    # Run evaluations
    completed = []
    for dataset in datasets:
        output_dir = run_evaluation(dataset)
        if output_dir:
            completed.append((dataset, output_dir))

    # Create summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    summary = {}
    for dataset, output_dir in completed:
        results_file = Path(output_dir) / "hashhop_results.json"
        if results_file.exists():
            with open(results_file, "r") as f:
                data = json.load(f)
                metrics = data["overall_metrics"]
                summary[dataset] = metrics
                print(
                    f"{dataset:20s}: {metrics['accuracy']:6.2%} ({metrics['total_correct']}/{metrics['total_queries']})"
                )

    # Save summary
    with open("hashhop_results_340m/evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nResults saved to: hashhop_results_340m/")
    print("Visualizations available in each dataset subdirectory")


if __name__ == "__main__":
    main()
