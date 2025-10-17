#!/usr/bin/env python3
"""
Extract existing highlights from CNN/DailyMail dataset without generating new summaries.
This is free (no OpenAI API calls) and provides the original ground-truth summaries.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
from datasets import load_dataset


class HighlightExtractor:
    """Extract and process existing highlights from CNN/DailyMail articles."""
    
    def __init__(self):
        """Initialize the highlight extractor."""
        pass
    
    def extract_highlights_from_article(
        self,
        article: str,
        highlights: str,
        article_id: str,
    ) -> Dict:
        """
        Extract and split highlights for a single article.
        
        Args:
            article: Article text
            highlights: Highlight text (multiple highlights separated by newlines)
            article_id: Unique article identifier
            
        Returns:
            Dictionary with article data and extracted highlights
        """
        article_word_count = len(article.split())
        
        # Split highlights by newline to get individual summaries
        individual_highlights = [
            h.strip() 
            for h in highlights.split('\n') 
            if h.strip()
        ]
        
        # Create summary entries for each highlight
        summaries = []
        for i, highlight in enumerate(individual_highlights):
            summaries.append({
                "summary_id": i,
                "text": highlight,
                "word_count": len(highlight.split()),
                "source": "cnn_dailymail_highlights",
            })
        
        # Calculate average summary length ratio
        if summaries:
            avg_summary_words = sum(s['word_count'] for s in summaries) / len(summaries)
            avg_length_ratio = avg_summary_words / article_word_count if article_word_count > 0 else 0
        else:
            avg_summary_words = 0
            avg_length_ratio = 0
        
        return {
            "article_id": article_id,
            "article": article,
            "article_word_count": article_word_count,
            "num_summaries": len(summaries),
            "avg_summary_words": avg_summary_words,
            "avg_length_ratio": avg_length_ratio,
            "summaries": summaries,
            "original_highlights": highlights,
        }


def process_dataset(
    extractor: HighlightExtractor,
    dataset_split: str = "train",
    num_articles: int = None,
    start_idx: int = 0,
    output_dir: str = "./extracted_highlights",
    save_interval: int = 1000,
):
    """
    Process CNN/DailyMail dataset and extract highlights.
    
    Args:
        extractor: HighlightExtractor instance
        dataset_split: Dataset split to use ('train', 'validation', 'test')
        num_articles: Number of articles to process (None = all)
        start_idx: Starting index in dataset
        output_dir: Directory to save results
        save_interval: Save checkpoint after this many articles
    """
    print("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0", split=dataset_split)
    
    # Determine range
    end_idx = start_idx + num_articles if num_articles else len(dataset)
    end_idx = min(end_idx, len(dataset))
    
    print(f"\nProcessing articles {start_idx} to {end_idx-1} ({end_idx - start_idx} total)")
    print(f"Extracting existing highlights from CNN/DailyMail")
    print(f"Dataset split: {dataset_split}")
    print(f"No API calls needed - FREE! ✨\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    results = []
    stats = {
        "start_time": datetime.now().isoformat(),
        "method": "highlight_extraction",
        "source": "cnn_dailymail_highlights",
        "dataset_split": dataset_split,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }
    
    # Process articles
    for idx in tqdm(range(start_idx, end_idx), desc="Extracting highlights"):
        article_data = dataset[idx]
        
        try:
            result = extractor.extract_highlights_from_article(
                article=article_data['article'],
                highlights=article_data['highlights'],
                article_id=article_data['id'],
            )
            
            results.append(result)
            
            # Save checkpoint periodically
            if (len(results) % save_interval == 0) or (idx == end_idx - 1):
                checkpoint_file = output_path / f"highlights_{start_idx}_{idx+1}.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        "stats": stats,
                        "results": results,
                    }, f, indent=2)
                
                if len(results) % save_interval == 0:
                    print(f"\nCheckpoint saved: {checkpoint_file}")
                
        except Exception as e:
            print(f"\nError processing article {idx}: {e}")
            continue
    
    # Calculate final statistics
    stats["end_time"] = datetime.now().isoformat()
    stats["total_articles_processed"] = len(results)
    stats["total_summaries_extracted"] = sum(r["num_summaries"] for r in results)
    
    if results:
        stats["avg_summaries_per_article"] = stats["total_summaries_extracted"] / len(results)
        stats["avg_summary_length"] = sum(
            sum(s["word_count"] for s in r["summaries"]) / len(r["summaries"])
            for r in results if r["summaries"]
        ) / len([r for r in results if r["summaries"]])
        stats["avg_article_length"] = sum(r["article_word_count"] for r in results) / len(results)
    
    # Save final results
    final_file = output_path / f"highlights_{start_idx}_{end_idx}_final.json"
    with open(final_file, 'w') as f:
        json.dump({
            "stats": stats,
            "results": results,
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("HIGHLIGHT EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total articles processed: {stats['total_articles_processed']}")
    print(f"Total highlights extracted: {stats['total_summaries_extracted']}")
    print(f"Avg highlights per article: {stats.get('avg_summaries_per_article', 0):.1f}")
    print(f"Avg highlight length: {stats.get('avg_summary_length', 0):.1f} words")
    print(f"Avg article length: {stats.get('avg_article_length', 0):.1f} words")
    print(f"Output file: {final_file}")
    print(f"Cost: $0.00 (no API calls!) 🎉")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract existing highlights from CNN/DailyMail dataset (no OpenAI API needed)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--num-articles",
        type=int,
        default=None,
        help="Number of articles to process (default: all)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index in dataset (default: 0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./extracted_highlights",
        help="Output directory (default: ./extracted_highlights)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N articles (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Initialize extractor (no API key needed!)
    extractor = HighlightExtractor()
    
    # Process dataset
    process_dataset(
        extractor=extractor,
        dataset_split=args.split,
        num_articles=args.num_articles,
        start_idx=args.start_idx,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
