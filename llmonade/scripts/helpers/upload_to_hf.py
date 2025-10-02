#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import tempfile

from huggingface_hub import HfApi

from llmonade.utils.convert_dcp_to_hf import save_pretrained
from torchtitan.tools.logging import init_logger, logger


def upload_to_huggingface(
    checkpoint_dir: str,
    step: int,
    config_path: str,
    tokenizer_path: str,
    hf_repo_id: str,
    private: bool = False,
    commit_message: str = None,
    token: str = None,
    create_pr: bool = False,
):
    """
    Convert a Flame DCP checkpoint to HF format and upload it to HuggingFace Hub

    Args:
        checkpoint_dir: Path to the checkpoint directory (e.g., exp/transformer-340M-distributed)
        step: Training step of the checkpoint to convert
        config_path: Path to the model config file
        tokenizer_path: Path to the tokenizer
        hf_repo_id: HuggingFace repository ID (e.g., "username/model-name")
        private: Whether to create a private repository
        commit_message: Custom commit message
        token: HuggingFace token (will use environment variable if not provided)
        create_pr: Whether to create a PR instead of pushing directly
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint/step-{step}")
    logger.info(f"Converting checkpoint at step {step} located at {checkpoint_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert DCP to HF format
        save_pretrained(
            output_path=tmpdir,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            tokenizer_path=tokenizer_path,
        )

        logger.info(f"Uploading model to HuggingFace Hub: {hf_repo_id}")

        # Use HF Hub API to upload the model
        api = HfApi(token=token)

        # Create repository if it doesn't exist
        try:
            api.create_repo(repo_id=hf_repo_id, private=private, exist_ok=True)
            logger.info(f"Repository created or already exists: {hf_repo_id}")
        except Exception as e:
            logger.error(f"Error creating repository: {e}")
            raise

        # Upload the converted model
        if not commit_message:
            commit_message = f"Upload model from Flame checkpoint at step {step}"

        try:
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=hf_repo_id,
                commit_message=commit_message,
                create_pr=create_pr,
            )
            logger.info(f"Successfully uploaded model to {hf_repo_id}")
        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            raise


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser("Upload a Flame model to HuggingFace Hub")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint directory (e.g., exp/transformer-340M-distributed)",
    )
    parser.add_argument(
        "--step", type=int, required=True, help="Training step to use (e.g., 100)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model config file (e.g., configs/transformer_340M.json)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to the tokenizer (e.g., fla-hub/transformer-1.3B-100B)",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create a private repository"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (will use HF_TOKEN env variable if not provided)",
    )
    parser.add_argument(
        "--commit_message", type=str, default=None, help="Custom commit message"
    )
    parser.add_argument(
        "--create_pr",
        action="store_true",
        help="Create a PR instead of pushing directly",
    )

    args = parser.parse_args()

    upload_to_huggingface(
        checkpoint_dir=args.checkpoint_path,
        step=args.step,
        config_path=args.config,
        tokenizer_path=args.tokenizer,
        hf_repo_id=args.hf_repo_id,
        private=args.private,
        commit_message=args.commit_message,
        token=args.token,
        create_pr=args.create_pr,
    )
