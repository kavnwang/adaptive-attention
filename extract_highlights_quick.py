#!/usr/bin/env python3
"""
Quick script to extract CNN/DailyMail highlights without API calls.
Just run: python extract_highlights_quick.py
"""

from extract_highlights import HighlightExtractor, process_dataset

# ============================================================================
# CONFIGURATION - Modify these settings as needed
# ============================================================================
CONFIG = {
    # Dataset settings
    "dataset_split": "train",            # train/validation/test
    "num_articles": 10000,               # Number of articles to process (None = all)
    "start_idx": 0,                      # Starting index (for resuming)
    
    # Output settings
    "output_dir": "./extracted_highlights",
    "save_interval": 1000,               # Save checkpoint every N articles
}


def main():
    """Main execution function."""
    print("="*80)
    print("  CNN/DailyMail Highlight Extraction")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {CONFIG['dataset_split']}")
    print(f"  Articles to process: {CONFIG['num_articles'] or 'ALL'}")
    print(f"  Starting from index: {CONFIG['start_idx']}")
    print(f"  Output directory: {CONFIG['output_dir']}")
    print(f"  Save checkpoints every: {CONFIG['save_interval']} articles")
    print(f"\n  💰 Cost: $0.00 (no API calls needed!)")
    print(f"\n{'='*80}\n")
    
    # Ask for confirmation
    if CONFIG['num_articles'] and CONFIG['num_articles'] > 50000:
        response = input("⚠️  Processing >50K articles. Continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled.")
            return
    
    print("Starting extraction...\n")
    
    # Initialize extractor (no API key needed!)
    extractor = HighlightExtractor()
    
    # Process dataset
    process_dataset(
        extractor=extractor,
        dataset_split=CONFIG["dataset_split"],
        num_articles=CONFIG["num_articles"],
        start_idx=CONFIG["start_idx"],
        output_dir=CONFIG["output_dir"],
        save_interval=CONFIG["save_interval"],
    )
    
    print("\n" + "="*80)
    print("  Extraction Complete!")
    print("="*80)
    print(f"\nResults saved to: {CONFIG['output_dir']}/")
    print("\nOutput format is compatible with generated summaries.")
    print("Each article has its original CNN/DailyMail highlights as 'summaries'.")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
