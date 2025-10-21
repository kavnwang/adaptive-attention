# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class CompressionPredictorConfig(PretrainedConfig):
    """
    Configuration for CompressionPredictor.

    This module is intended to estimate an appropriate compression ratio
    given a sequence of compressed hidden states and a scalar encoding of
    the added NLL (Delta) between compressed vs. full attention.

    Note: This is a stub config to enable registration and wiring; modeling
    is intentionally minimal for now.
    """

    model_type = "compression_predictor"

    def __init__(
        self,
        # Input feature sizes
        hidden_size: int = 2048,  # dimension of compressed hidden states
        delta_embed_dim: int = 64,  # sinusoidal/other embedding size for Delta

        # Simple MLP-style head defaults (stub; not implemented here)
        predictor_hidden_size: int = 512,
        predictor_num_layers: int = 2,
        dropout: float = 0.0,

        # Target space control
        regression_target: bool = True,  # predict float ratio if True; else classify bins
        num_ratio_bins: int = 0,         # used if regression_target is False
        min_ratio: float = 0.0,          # clamp lower bound for ratio prediction
        max_ratio: float = 1.0,          # clamp upper bound for ratio prediction

        # Misc common fields
        initializer_range: float = 0.02,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.delta_embed_dim = delta_embed_dim

        self.predictor_hidden_size = predictor_hidden_size
        self.predictor_num_layers = predictor_num_layers
        self.dropout = dropout

        self.regression_target = regression_target
        self.num_ratio_bins = num_ratio_bins
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

