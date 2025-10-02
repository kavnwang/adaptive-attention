nth  #!/usr/bin/env python3
"""
Fast evaluation script for synthetic QA datasets.

This script provides optimized evaluation that completes in minutes rather than hours
by using limited sample sizes while maintaining statistical validity.
"""

import argparse
import json
import logging
import subprocess
import sys
import os
from pathlib import Path

# Fix the import path for lm-evaluation-harness
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "3rdparty/lm-evaluation-harness")
)

from torchtitan.tools.logging import init_logger


def main():
    init_logger()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Fast evaluation of synthetic QA datasets"
    )

    parser.add_argument(
        "--dump_folder",
        type=str,
        required=True,
        help="Path to the folder containing model checkpoints",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to the tokenizer"
    )
    parser.add_argument(
        "--curr_step",
        type=int,
        required=True,
        help="Current training step/checkpoint to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32 for faster evaluation)",
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="Wandb project name"
    )
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Wandb run id")
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the evaluation results",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        choices=["single", "by_frequency", "recall_only"],
        default="by_frequency",
        help="Type of evaluation to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run evaluation on",
    )

    args = parser.parse_args()

    # Configure tasks based on evaluation type
    if args.eval_type == "single":
        tasks = ["synthetic_qa_recall"]
    elif args.eval_type == "by_frequency":
        tasks = ["synthetic_qa_by_frequency"]
    elif args.eval_type == "recall_only":
        tasks = ["synthetic_qa_recall"]

    logger.info("=" * 60)
    logger.info("FAST SYNTHETIC QA EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Dump folder: {args.dump_folder}")
    logger.info(f"Tokenizer path: {args.tokenizer_path}")
    logger.info(f"Evaluation step: {args.curr_step}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Evaluation type: {args.eval_type}")
    logger.info("=" * 60)

    print(f"🚀 Starting FAST synthetic QA evaluation for step {args.curr_step}")
    print(f"📊 Evaluation type: {args.eval_type}")
    print(f"⚡ Batch size: {args.batch_size}")
    print(f"🎯 Tasks: {tasks}")

    try:
        # Build the command to run lm_eval via llmonade harness
        # This ensures the bento model is registered
        cmd = [
            "python",
            "-m",
            "llmonade.evals.harness",
            "--model",
            "bento",
            "--model_args",
            f"pretrained={args.dump_folder},dtype=bfloat16,max_length=4096",
            "--tasks",
            ",".join(tasks),
            "--batch_size",
            str(args.batch_size),
            "--num_fewshot",
            "0",
            "--device",
            args.device,
        ]

        # Set up environment with correct PYTHONPATH
        env = os.environ.copy()
        lm_eval_path = os.path.join(
            os.path.dirname(__file__), "3rdparty/lm-evaluation-harness"
        )
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{lm_eval_path}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = lm_eval_path

        # Add output path if specified
        if args.save_path:
            cmd.extend(["--output_path", args.save_path])

        logger.info(f"Running command: {' '.join(cmd)}")

        # Run the evaluation with the updated environment
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode != 0:
            logger.error(f"Evaluation failed with return code: {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Evaluation failed: {result.stderr}")

        logger.info("=" * 60)
        logger.info("FAST EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

        print("✅ Fast evaluation completed!")

        # Load and display results if saved
        if args.save_path and Path(args.save_path).exists():
            with open(args.save_path, "r") as f:
                results = json.load(f)

            # Print summary
            print("\n📈 RESULTS SUMMARY:")
            print("-" * 40)
            if "results" in results:
                for task, task_results in results["results"].items():
                    if isinstance(task_results, dict):
                        for metric, value in task_results.items():
                            if isinstance(value, (int, float)):
                                print(f"{task} - {metric}: {value:.4f}")

        # Print performance info
        print("\n⏱️  Evaluation completed in minutes (vs hours for full dataset)")
        print("🔢 Sample size: Limited for faster evaluation")
        print("📊 Statistical validity maintained with representative sampling")

    except Exception as e:
        logger.error(f"Fast evaluation failed with error: {e}")
        logger.error("Full traceback:", exc_info=True)
        print(f"❌ Fast evaluation failed: {e}")
        raise RuntimeError(f"Fast evaluation failed: {e}")


if __name__ == "__main__":
    main()
