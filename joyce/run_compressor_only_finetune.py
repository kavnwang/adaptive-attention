#!/usr/bin/env python3
"""
Compressor-only finetune runner

Loads an existing visual-compressor checkpoint, then continues training while
freezing every parameter except the compression layer at a given text layer L.

This is a thin wrapper around train_visual_compressor.py that:
- Forces --train_only_compressor
- Passes through your chosen --load_text_checkpoint
- Lets you pick a separate --save_dir to avoid overwriting your prior run

Example
  python run_compressor_only_finetune.py \
    --annotations_json data/clevr-mini/clevr_summaries.json \
    --images_root data/clevr-mini \
    --load_text_checkpoint exp/visual_compressor_training/final_model.pt \
    --save_dir exp/visual_compressor_training_compressor_only_v1 \
    --num_items 200 --num_epochs 1 --batch_size 2

You can override any defaults (vision model, layer index, etc.).
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="Run compressor-only finetune from an existing visual-compressor checkpoint")

    # Required data
    ap.add_argument("--annotations_json", type=str, required=True, help="CLEVR-style annotations JSON")
    ap.add_argument("--images_root", type=str, required=True, help="Root for relative image paths")

    # Checkpoints
    ap.add_argument("--load_text_checkpoint", type=str, default="exp/visual_compressor_training/final_model.pt",
                    help="Existing trained checkpoint to initialize from (text + compressor + optional vision_proj)")
    ap.add_argument("--save_dir", type=str, default="exp/visual_compressor_training_compressor_only",
                    help="Directory to save new checkpoints/final model")

    # Models
    ap.add_argument("--text_model_path", type=str, default="exp/transformer_160M")
    ap.add_argument("--vision_model", type=str, default="HuggingFaceTB/SmolVLM2-500M-Instruct")
    ap.add_argument("--layer_idx", type=int, default=6)
    ap.add_argument("--compress_num_tokens", type=int, default=32)
    ap.add_argument("--compress_depth", type=int, default=0)

    # Data subset
    ap.add_argument("--num_items", type=int, default=None)
    ap.add_argument("--max_summary_len", type=int, default=64)
    ap.add_argument("--max_summaries_per_image", type=int, default=4)

    # Training
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_epochs", type=int, default=1)
    ap.add_argument("--compressor_lr", type=float, default=5e-4)
    ap.add_argument("--upper_layers_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--gradient_clip", type=float, default=1.0)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--check_grads_every", type=int, default=100)
    ap.add_argument("--device", type=str, default=None, help="Override device; defaults to CUDA if available")
    ap.add_argument("--num_workers", type=int, default=4)

    # Debug (optional)
    ap.add_argument("--debug_memory_dir", type=str, default=None)
    ap.add_argument("--debug_memory_every", type=int, default=0)
    ap.add_argument("--debug_memory_limit", type=int, default=5)
    ap.add_argument("--debug_memory_top_k", type=int, default=5)

    args = ap.parse_args()

    # Build argv for train_visual_compressor.py
    argv = [
        "train_visual_compressor.py",
        "--text_model_path", args.text_model_path,
        "--vision_model", args.vision_model,
        "--layer_idx", str(args.layer_idx),
        "--compress_num_tokens", str(args.compress_num_tokens),
        "--compress_depth", str(args.compress_depth),
        "--annotations_json", args.annotations_json,
        "--images_root", args.images_root,
        "--batch_size", str(args.batch_size),
        "--num_epochs", str(args.num_epochs),
        "--compressor_lr", str(args.compressor_lr),
        "--upper_layers_lr", str(args.upper_layers_lr),
        "--weight_decay", str(args.weight_decay),
        "--gradient_clip", str(args.gradient_clip),
        "--save_dir", args.save_dir,
        "--save_every", str(args.save_every),
        "--check_grads_every", str(args.check_grads_every),
        "--num_workers", str(args.num_workers),
        "--train_only_compressor",
        "--load_text_checkpoint", args.load_text_checkpoint,
    ]

    # Optional flags/values
    if args.num_items is not None:
        argv += ["--num_items", str(args.num_items)]
    if args.max_summary_len is not None:
        argv += ["--max_summary_len", str(args.max_summary_len)]
    if args.max_summaries_per_image is not None:
        argv += ["--max_summaries_per_image", str(args.max_summaries_per_image)]
    if args.device:
        argv += ["--device", args.device]
    if args.debug_memory_dir and args.debug_memory_every > 0:
        argv += [
            "--debug_memory_dir", args.debug_memory_dir,
            "--debug_memory_every", str(args.debug_memory_every),
            "--debug_memory_limit", str(args.debug_memory_limit),
            "--debug_memory_top_k", str(args.debug_memory_top_k),
        ]

    # Import and delegate to training main()
    import train_visual_compressor as tvc
    old_argv = sys.argv
    try:
        sys.argv = argv
        tvc.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

