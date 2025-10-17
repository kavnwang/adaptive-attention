#!/usr/bin/env python3
"""
Ready-to-run summary generation script with environment-based API key.
Run: python generate_with_key.py

TODO(security): Ensure OPENAI_API_KEY is provided via environment or secret manager.
"""

import os
import sys
from generate_summaries import SummaryGenerator, process_dataset

# ============================================================================
# API KEY CONFIGURATION - load from environment
# ============================================================================
# TODO: Replace with your configuration loader if applicable.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not set. Export it in your environment or create a .env file (not committed)."
    )

# ============================================================================
# CONFIGURATION - Modify these settings as needed
# ============================================================================
CONFIG = {
    # Model settings
    "model": "gpt-5-nano",              # OpenAI model to use
    "num_summaries": 10,                 # Summaries per article
    "target_length_ratio": 0.3,          # 30% of original length
    
    # Dataset settings
    "dataset_split": "train",       # train/validation/test
    "num_articles": 1000,                   # Number of articles to process
    "start_idx": 0,                      # Starting index (for resuming)
    
    # Output settings
    "output_dir": "./generated_summaries",
    "save_interval": 5,                  # Save checkpoint every N articles
}


def main():
    """Main execution function."""
    print("="*80)
    print("  Summary Generation Pipeline")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {CONFIG['model']}")
    print(f"  Summaries per article: {CONFIG['num_summaries']}")
    print(f"  Target length: {CONFIG['target_length_ratio']:.0%} of original")
    print(f"  Dataset: {CONFIG['dataset_split']}")
    print(f"  Articles to process: {CONFIG['num_articles']}")
    print(f"  Output directory: {CONFIG['output_dir']}")
    print(f"\n{'='*80}\n")
    
    # Ask for confirmation
    response = input("Start generation? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        sys.exit(0)
    
    print("\nInitializing generator...\n")
    
    # Initialize generator with API key
    generator = SummaryGenerator(
        api_key=OPENAI_API_KEY,           # ← API key configured here
        model=CONFIG["model"],
        num_summaries=CONFIG["num_summaries"],
        target_length_ratio=CONFIG["target_length_ratio"],
    )
    
    # Process dataset
    process_dataset(
        generator=generator,
        dataset_split=CONFIG["dataset_split"],
        num_articles=CONFIG["num_articles"],
        start_idx=CONFIG["start_idx"],
        output_dir=CONFIG["output_dir"],
        save_interval=CONFIG["save_interval"],
    )
    
    print("\n" + "="*80)
    print("  Generation Complete!")
    print("="*80)
    print(f"\nResults saved to: {CONFIG['output_dir']}/")
    print("\nTo analyze results, run:")
    print(f"  python analyze_generated_summaries.py {CONFIG['output_dir']}/*.json")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
