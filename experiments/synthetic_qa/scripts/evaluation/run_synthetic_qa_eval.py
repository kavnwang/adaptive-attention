#!/usr/bin/env python3
"""
Direct evaluation script for synthetic QA using lm-evaluation-harness
"""

import sys
import subprocess
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run synthetic QA evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save results"
    )
    parser.add_argument(
        "--task", type=str, default="synthetic_qa_recall", help="Task to run"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")

    args = parser.parse_args()

    # Run evaluation using lm_eval command line
    cmd = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={args.model_path},dtype=bfloat16",
        "--tasks",
        args.task,
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
        "--output_path",
        args.output_path,
        "--log_samples",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("✅ Evaluation completed successfully!")
        print(f"Results saved to: {args.output_path}")

        # Load and display results
        if Path(args.output_path).exists():
            with open(args.output_path, "r") as f:
                results = json.load(f)
                if "results" in results:
                    print("\n📊 Results:")
                    for task, metrics in results["results"].items():
                        print(f"\n{task}:")
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                print(f"  {metric}: {value:.4f}")
    else:
        print(f"❌ Evaluation failed with return code: {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
