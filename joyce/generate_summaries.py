#!/usr/bin/env python3
"""
Pipeline to generate multiple summaries for CNN/DailyMail articles using OpenAI API.
Each article generates 10 different summaries at ~30% of original length.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm
from openai import OpenAI, RateLimitError, BadRequestError, NotFoundError
from datasets import load_dataset


class SummaryGenerator:
    """Generate multiple diverse summaries for articles using OpenAI API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-nano",
        num_summaries: int = 10,
        target_length_ratio: float = 0.3,
        temperature_range: tuple = (0.7, 1.0),
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        """
        Initialize the summary generator.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use (gpt-5-nano is cost-effective)
            num_summaries: Number of summaries to generate per article
            target_length_ratio: Target summary length as ratio of original (0.3 = 30%)
            temperature_range: (min, max) temperature for diversity
            max_retries: Maximum retry attempts for API calls
            retry_delay: Delay in seconds between retries
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        fallback_model = "gpt-5-nano"
        self.model = model
        self.num_summaries = num_summaries
        self.target_length_ratio = target_length_ratio
        self.temperature_range = temperature_range
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature_supported = True

        # Validate model availability and fall back if necessary
        try:
            self.client.models.retrieve(self.model)
        except NotFoundError:
            print(f"\nModel '{self.model}' unavailable. Falling back to '{fallback_model}'.")
            self.model = fallback_model
        except Exception as exc:
            print(f"\nWarning: Could not validate model '{self.model}' ({exc}). Continuing anyway.")
        
    def create_prompt(self, article: str, target_words: int, style_variation: int) -> str:
        """
        Create a prompt for summary generation with style variations.
        
        Args:
            article: The article text to summarize
            target_words: Target word count for the summary
            style_variation: Integer to create diverse prompts (0-9)
        """
        # Different style instructions for diversity
        style_instructions = [
            "Write a comprehensive summary focusing on the key facts and events.",
            "Create a summary that emphasizes the main narrative and timeline of events.",
            "Summarize by highlighting the most important points and their implications.",
            "Write a summary that captures the essential information in a journalistic style.",
            "Create a concise summary focusing on the who, what, when, where, and why.",
            "Summarize the article by extracting and organizing the critical details.",
            "Write a summary that presents the main story in a clear, structured way.",
            "Create a summary focusing on the key developments and their significance.",
            "Summarize by identifying and explaining the most newsworthy elements.",
            "Write a summary that distills the article into its core message and facts.",
        ]
        
        style_instruction = style_instructions[style_variation % len(style_instructions)]
        
        prompt = f"""Please read the following news article and create a summary.

{style_instruction}

Target length: approximately {target_words} words (about 30% of the original article length).

Article:
{article}

Summary:"""
        
        return prompt
    
    def generate_summary(
        self,
        article: str,
        target_words: int,
        temperature: float,
        style_variation: int,
    ) -> Optional[str]:
        """
        Generate a single summary using OpenAI API.
        
        Args:
            article: Article text
            target_words: Target word count
            temperature: Sampling temperature
            style_variation: Style variation index
            
        Returns:
            Generated summary or None if failed
        """
        prompt = self.create_prompt(article, target_words, style_variation)
        
        for attempt in range(self.max_retries):
            try:
                request_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a professional news summarizer. Create clear, accurate, and concise summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_completion_tokens": int(target_words * 2),  # Allow some flexibility
                }

                if self.temperature_supported and temperature is not None:
                    request_kwargs["temperature"] = temperature

                try:
                    response = self.client.chat.completions.create(**request_kwargs)
                except BadRequestError as e:
                    error_msg = str(e).lower()
                    if "temperature" in error_msg and "unsupported" in error_msg:
                        if self.temperature_supported:
                            print("\nTemperature parameter not supported for this model. Using default temperature.")
                        self.temperature_supported = False
                        request_kwargs.pop("temperature", None)
                        response = self.client.chat.completions.create(**request_kwargs)
                    else:
                        raise
                
                summary = response.choices[0].message.content.strip()
                return summary
                
            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"\nRate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"\nFailed after {self.max_retries} attempts: {e}")
                    return None
                    
            except Exception as e:
                print(f"\nError generating summary: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return None
        
        return None
    
    def generate_summaries_for_article(
        self,
        article: str,
        article_id: str,
    ) -> Dict:
        """
        Generate multiple summaries for a single article.
        
        Args:
            article: Article text
            article_id: Unique article identifier
            
        Returns:
            Dictionary with article data and generated summaries
        """
        article_word_count = len(article.split())
        target_words = int(article_word_count * self.target_length_ratio)
        
        # Generate temperature values for diversity
        temperatures = [
            self.temperature_range[0] + 
            (self.temperature_range[1] - self.temperature_range[0]) * i / (self.num_summaries - 1)
            for i in range(self.num_summaries)
        ]
        
        summaries = []
        for i in range(self.num_summaries):
            summary = self.generate_summary(
                article=article,
                target_words=target_words,
                temperature=temperatures[i],
                style_variation=i,
            )
            
            if summary:
                recorded_temperature = temperatures[i] if self.temperature_supported else 1.0
                summaries.append({
                    "summary_id": i,
                    "text": summary,
                    "word_count": len(summary.split()),
                    "temperature": recorded_temperature,
                    "style_variation": i,
                })
            
            # Brief pause to avoid rate limits
            time.sleep(0.5)
        
        return {
            "article_id": article_id,
            "article": article,
            "article_word_count": article_word_count,
            "target_summary_words": target_words,
            "num_summaries_generated": len(summaries),
            "summaries": summaries,
        }


def process_dataset(
    generator: SummaryGenerator,
    dataset_split: str = "train",
    num_articles: Optional[int] = None,
    start_idx: int = 0,
    output_dir: str = "./generated_summaries",
    save_interval: int = 10,
):
    """
    Process CNN/DailyMail dataset and generate summaries.
    
    Args:
        generator: SummaryGenerator instance
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
    print(f"Generating {generator.num_summaries} summaries per article")
    print(f"Model: {generator.model}")
    print(f"Target length ratio: {generator.target_length_ratio:.0%}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    results = []
    stats = {
        "start_time": datetime.now().isoformat(),
        "model": generator.model,
        "num_summaries_per_article": generator.num_summaries,
        "target_length_ratio": generator.target_length_ratio,
        "dataset_split": dataset_split,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }
    
    # Process articles
    for idx in tqdm(range(start_idx, end_idx), desc="Processing articles"):
        article_data = dataset[idx]
        
        try:
            result = generator.generate_summaries_for_article(
                article=article_data['article'],
                article_id=article_data['id'],
            )
            
            # Add original highlights for comparison
            result['original_highlights'] = article_data['highlights']
            results.append(result)
            
            # Save checkpoint periodically
            if (len(results) % save_interval == 0) or (idx == end_idx - 1):
                checkpoint_file = output_path / f"summaries_{start_idx}_{idx+1}.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        "stats": stats,
                        "results": results,
                    }, f, indent=2)
                print(f"\nCheckpoint saved: {checkpoint_file}")
                
        except Exception as e:
            print(f"\nError processing article {idx}: {e}")
            continue
    
    # Save final results
    stats["end_time"] = datetime.now().isoformat()
    stats["total_articles_processed"] = len(results)
    stats["total_summaries_generated"] = sum(r["num_summaries_generated"] for r in results)
    
    final_file = output_path / f"summaries_{start_idx}_{end_idx}_final.json"
    with open(final_file, 'w') as f:
        json.dump({
            "stats": stats,
            "results": results,
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("SUMMARY GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total articles processed: {stats['total_articles_processed']}")
    print(f"Total summaries generated: {stats['total_summaries_generated']}")
    print(f"Output file: {final_file}")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate multiple summaries for CNN/DailyMail articles using OpenAI API"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--num-summaries",
        type=int,
        default=10,
        help="Number of summaries per article (default: 10)"
    )
    parser.add_argument(
        "--length-ratio",
        type=float,
        default=0.3,
        help="Target summary length ratio (default: 0.3 = 30%%)"
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
        default="./generated_summaries",
        help="Output directory (default: ./generated_summaries)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save checkpoint every N articles (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SummaryGenerator(
        api_key=args.api_key,
        model=args.model,
        num_summaries=args.num_summaries,
        target_length_ratio=args.length_ratio,
    )
    
    # Process dataset
    process_dataset(
        generator=generator,
        dataset_split=args.split,
        num_articles=args.num_articles,
        start_idx=args.start_idx,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
