# -*- coding: utf-8 -*-

from __future__ import annotations

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

import os

# Import bento models to register custom model types
import fla  # noqa

# Add imports for the Triton cache fix
import logging

try:
    import torch.distributed as dist
except ImportError:
    dist = None

logger = logging.getLogger(__name__)


# Setup per-rank Triton cache directory before model loading
def setup_triton_cache():
    """Setup separate Triton cache directories for each GPU rank to avoid race conditions."""
    if dist is not None and dist.is_initialized():
        global_rank = dist.get_rank()
    else:
        global_rank = int(os.environ.get("LOCAL_RANK", 0))

    base_cache_dir = os.environ.get("TRITON_CACHE_DIR", "~/tmp/triton_cache_user_owned")
    base_cache_dir = os.path.expanduser(base_cache_dir)
    rank_cache_dir = os.path.join(base_cache_dir, f"rank_{global_rank}")
    os.makedirs(rank_cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = rank_cache_dir
    logger.info(f"Using Triton cache directory: {rank_cache_dir}")


# Call the setup function at module level
setup_triton_cache()


@register_model("bento")
class CustomLMWrapper(HFLM):
    def __init__(self, **kwargs) -> CustomLMWrapper:
        print("Loading bento model with args:", kwargs)
        # TODO: provide options for doing inference with different kernels
        super().__init__(**kwargs)


if __name__ == "__main__":
    cli_evaluate()
