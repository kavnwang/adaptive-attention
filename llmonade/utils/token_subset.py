# -*- coding: utf-8 -*-
"""
Token subset optimization utilities for LLMonade.

This module implements functionality similar to Unsloth's fix_untrained_tokens,
which optimizes model initialization by identifying tokens that appear in the
training dataset and adjusting embeddings for unused tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Set, Optional, Union, List
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset, IterableDataset
from torchtitan.tools.logging import logger
import numpy as np
from torch.distributed.tensor import DTensor


def identify_used_tokens(
    dataset: Union[Dataset, IterableDataset],
    tokenizer: PreTrainedTokenizer,
    max_samples: Optional[int] = None,
    sample_field: str = "input_ids",
) -> Set[int]:
    """
    Identify which token IDs actually appear in the dataset.
    
    Args:
        dataset: The training dataset
        tokenizer: The tokenizer
        max_samples: Maximum number of samples to scan (None for all)
        sample_field: Field name containing token IDs
        
    Returns:
        Set of token IDs that appear in the dataset
    """
    used_tokens = set()
    samples_scanned = 0
    
    logger.info("Scanning dataset to identify used tokens...")
    
    # Handle both Dataset and IterableDataset
    if isinstance(dataset, IterableDataset):
        # For IterableDataset, we need to iterate
        for sample in dataset:
            if sample_field in sample:
                tokens = sample[sample_field]
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
                elif isinstance(tokens, np.ndarray):
                    tokens = tokens.tolist()
                used_tokens.update(tokens)
            samples_scanned += 1
            if max_samples and samples_scanned >= max_samples:
                break
    else:
        # For regular Dataset, we can use indexing
        num_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        for i in range(num_samples):
            sample = dataset[i]
            if sample_field in sample:
                tokens = sample[sample_field]
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
                elif isinstance(tokens, np.ndarray):
                    tokens = tokens.tolist()
                used_tokens.update(tokens)
            samples_scanned += 1
    
    logger.info(f"Scanned {samples_scanned} samples, found {len(used_tokens)} unique tokens")
    return used_tokens


def optimize_embeddings_for_token_subset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    used_tokens: Set[int],
    eps: float = 1e-16,
    scale_factor: float = 1e-5,
) -> None:
    """
    Optimize model embeddings based on which tokens are actually used.
    Simplified version that uses masking operations compatible with DTensors.
    
    This function modifies embeddings for unused tokens to very small values,
    which can significantly reduce initial loss for models trained on limited
    vocabulary datasets.
    
    Args:
        model: The model to optimize
        tokenizer: The tokenizer
        used_tokens: Set of token IDs that appear in training data
        eps: Small epsilon value for numerical stability
        scale_factor: Scaling factor for unused token embeddings
    """
    # Get embedding layers first to determine actual vocabulary size
    embed_tokens = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    
    if embed_tokens is None or lm_head is None:
        logger.warning("Could not find embedding layers, skipping optimization")
        return
    
    # Use actual embedding size instead of tokenizer vocab size
    # This handles cases where models pad vocabulary for efficiency (e.g., Pythia)
    actual_vocab_size = embed_tokens.weight.shape[0]
    tokenizer_vocab_size = tokenizer.vocab_size
    
    # Only consider tokens within tokenizer's vocabulary range
    unused_tokens = set(range(tokenizer_vocab_size)) - used_tokens
    
    logger.info(f"Optimizing embeddings: {len(used_tokens)} used tokens, {len(unused_tokens)} unused tokens")
    logger.info(f"Tokenizer vocab size: {tokenizer_vocab_size}, Model vocab size: {actual_vocab_size}")
    
    with torch.no_grad():
        # Check if we're dealing with DTensor (distributed tensor)
        is_dtensor = isinstance(embed_tokens.weight, DTensor)
        
        if is_dtensor:
            # For DTensor, we need to handle sharding properly
            # Get the local tensor which might be a shard of the full embedding
            local_weight = embed_tokens.weight.to_local()
            local_device = local_weight.device
            local_vocab_size = local_weight.shape[0]  # This might be less than actual_vocab_size
            
            # We need to figure out which tokens this shard is responsible for
            # This depends on the sharding strategy. For now, let's handle the most common case
            # where embeddings are sharded along dimension 0 (vocabulary dimension)
            
            # Get sharding information
            from torch.distributed.tensor import Shard
            placements = embed_tokens.weight.placements
            
            # Find if dim 0 is sharded
            is_vocab_sharded = False
            shard_dim = None
            for i, placement in enumerate(placements):
                if isinstance(placement, Shard) and placement.dim == 0:
                    is_vocab_sharded = True
                    shard_dim = i
                    break
            
            if is_vocab_sharded:
                # Calculate which token indices this rank handles
                import torch.distributed as dist
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                
                # Calculate the token range for this rank
                tokens_per_rank = actual_vocab_size // world_size
                start_token = rank * tokens_per_rank
                end_token = start_token + local_vocab_size
                
                # Create mask only for tokens this rank handles
                local_unused_mask = torch.zeros(local_vocab_size, dtype=torch.bool, device=local_device)
                for token_id in unused_tokens:
                    if start_token <= token_id < end_token:
                        local_token_idx = token_id - start_token
                        if local_token_idx < local_vocab_size:
                            local_unused_mask[local_token_idx] = True
                
                # Expand mask to match local embedding dimensions
                local_unused_mask_expanded = local_unused_mask.unsqueeze(1).expand_as(local_weight)
                
                # Scale down unused embeddings using mask
                new_weight = torch.where(
                    local_unused_mask_expanded,
                    local_weight * scale_factor,
                    local_weight
                )
                
                # Add small noise to unused embeddings
                noise = eps * torch.randn_like(local_weight)
                new_weight = torch.where(
                    local_unused_mask_expanded,
                    new_weight + noise,
                    new_weight
                )
                
                # Update the local tensor
                embed_tokens.weight._local_tensor.copy_(new_weight)
            else:
                # If not sharded on vocab dimension, create full mask
                unused_mask = torch.zeros(local_vocab_size, dtype=torch.bool, device=local_device)
                for token_id in unused_tokens:
                    if token_id < local_vocab_size:
                        unused_mask[token_id] = True
                
                unused_mask_expanded = unused_mask.unsqueeze(1).expand_as(local_weight)
                
                # Apply transformations
                new_weight = torch.where(
                    unused_mask_expanded,
                    local_weight * scale_factor,
                    local_weight
                )
                
                noise = eps * torch.randn_like(local_weight)
                new_weight = torch.where(
                    unused_mask_expanded,
                    new_weight + noise,
                    new_weight
                )
                
                embed_tokens.weight._local_tensor.copy_(new_weight)
        else:
            # Regular tensor path (non-distributed)
            unused_mask = torch.zeros(actual_vocab_size, dtype=torch.bool, device=embed_tokens.weight.device)
            
            for token_id in unused_tokens:
                unused_mask[token_id] = True
            
            unused_mask_expanded = unused_mask.unsqueeze(1).expand_as(embed_tokens.weight)
            
            # Scale down unused embeddings using mask
            embed_tokens.weight.data = torch.where(
                unused_mask_expanded,
                embed_tokens.weight.data * scale_factor,
                embed_tokens.weight.data
            )
            
            # Add small noise to unused embeddings to prevent zero gradients
            noise = eps * torch.randn_like(embed_tokens.weight)
            embed_tokens.weight.data = torch.where(
                unused_mask_expanded,
                embed_tokens.weight.data + noise,
                embed_tokens.weight.data
            )
        
        # Handle output embeddings if not tied
        if lm_head.weight is not embed_tokens.weight:
            is_lm_head_dtensor = isinstance(lm_head.weight, DTensor)
            
            if is_lm_head_dtensor:
                # For DTensor lm_head - similar handling as embed_tokens
                local_lm_weight = lm_head.weight.to_local()
                local_lm_device = local_lm_weight.device
                local_lm_vocab_size = local_lm_weight.shape[0]
                
                # Check if lm_head is sharded on vocab dimension
                from torch.distributed.tensor import Shard
                lm_placements = lm_head.weight.placements
                
                is_lm_vocab_sharded = False
                for i, placement in enumerate(lm_placements):
                    if isinstance(placement, Shard) and placement.dim == 0:
                        is_lm_vocab_sharded = True
                        break
                
                if is_lm_vocab_sharded:
                    # Handle sharded lm_head
                    import torch.distributed as dist
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    
                    # Calculate the token range for this rank
                    # Note: lm_head might have different size than embed_tokens
                    lm_actual_vocab_size = lm_head.weight.shape[0] if not is_lm_vocab_sharded else actual_vocab_size
                    tokens_per_rank = lm_actual_vocab_size // world_size
                    start_token = rank * tokens_per_rank
                    end_token = start_token + local_lm_vocab_size
                    
                    # Create mask only for tokens this rank handles
                    local_lm_unused_mask = torch.zeros(local_lm_vocab_size, dtype=torch.bool, device=local_lm_device)
                    for token_id in unused_tokens:
                        if start_token <= token_id < end_token:
                            local_token_idx = token_id - start_token
                            if local_token_idx < local_lm_vocab_size:
                                local_lm_unused_mask[local_token_idx] = True
                    
                    local_lm_unused_mask_expanded = local_lm_unused_mask.unsqueeze(1).expand_as(local_lm_weight)
                else:
                    # Not sharded on vocab dim
                    local_lm_unused_mask = torch.zeros(local_lm_vocab_size, dtype=torch.bool, device=local_lm_device)
                    for token_id in unused_tokens:
                        if token_id < local_lm_vocab_size:
                            local_lm_unused_mask[token_id] = True
                    
                    local_lm_unused_mask_expanded = local_lm_unused_mask.unsqueeze(1).expand_as(local_lm_weight)
                
                # Apply transformations
                new_lm_weight = torch.where(
                    local_lm_unused_mask_expanded,
                    local_lm_weight * scale_factor,
                    local_lm_weight
                )
                
                # Add noise
                noise = eps * torch.randn_like(local_lm_weight)
                new_lm_weight = torch.where(
                    local_lm_unused_mask_expanded,
                    new_lm_weight + noise,
                    new_lm_weight
                )
                
                # Update the local tensor
                lm_head.weight._local_tensor.copy_(new_lm_weight)
            else:
                # Regular tensor path
                # Check if lm_head has different shape than embeddings
                lm_vocab_size = lm_head.weight.shape[0]
                lm_unused_mask = torch.zeros(lm_vocab_size, dtype=torch.bool, device=lm_head.weight.device)
                
                for token_id in unused_tokens:
                    if token_id < lm_vocab_size:
                        lm_unused_mask[token_id] = True
                
                lm_unused_mask_expanded = lm_unused_mask.unsqueeze(1).expand_as(lm_head.weight)
                
                lm_head.weight.data = torch.where(
                    lm_unused_mask_expanded,
                    lm_head.weight.data * scale_factor,
                    lm_head.weight.data
                )
                
                # Add noise to output embeddings too
                noise = eps * torch.randn_like(lm_head.weight)
                lm_head.weight.data = torch.where(
                    lm_unused_mask_expanded,
                    lm_head.weight.data + noise,
                    lm_head.weight.data
                )
    
    logger.info("Embedding optimization complete")


def apply_token_subset_optimization(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Union[Dataset, IterableDataset],
    max_scan_samples: Optional[int] = 10000,
    eps: float = 1e-16,
    scale_factor: float = 1e-5,
    enable: bool = True,
) -> Optional[Set[int]]:
    """
    Apply token subset optimization to model based on training data.
    This is the main entry point that combines token identification and embedding optimization.
    
    Args:
        model: The model to optimize
        tokenizer: The tokenizer
        train_dataset: The training dataset
        max_scan_samples: Maximum samples to scan for token identification
        eps: Small epsilon value for numerical stability
        scale_factor: Scaling factor for unused token embeddings
        enable: Whether to apply the optimization (useful for toggling)
        
    Returns:
        Set of used tokens if optimization was applied, None otherwise
    """
    if not enable:
        logger.info("Token subset optimization disabled")
        return None
    
    logger.info("Applying token subset optimization (Unsloth-style fix_untrained_tokens)")
    
    # Identify which tokens are used in the dataset
    used_tokens = identify_used_tokens(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_samples=max_scan_samples,
        sample_field="input_ids" if hasattr(train_dataset, "__getitem__") else "input_ids"
    )
    
    # Optimize embeddings based on token usage
    optimize_embeddings_for_token_subset(
        model=model,
        tokenizer=tokenizer,
        used_tokens=used_tokens,
        eps=eps,
        scale_factor=scale_factor
    )
    
    return used_tokens


def create_token_subset_mask(
    vocab_size: int,
    used_tokens: Set[int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a boolean mask for used tokens.
    
    Args:
        vocab_size: Total vocabulary size
        used_tokens: Set of token IDs that are used
        device: Device to create mask on
        
    Returns:
        Boolean tensor of shape [vocab_size] where True indicates used tokens
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for token_id in used_tokens:
        mask[token_id] = True
    return mask