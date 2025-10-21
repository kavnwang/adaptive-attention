# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from fla.models.compression_predictor.configuration_compression_predictor import (
    CompressionPredictorConfig,
)


@dataclass
class CompressionPredictorOutput(ModelOutput):
    """
    Output of CompressionPredictor.

    Attributes:
        compression_ratio: (torch.FloatTensor): shape [batch] or [batch, 1]
            Predicted ratio p / seq_len in [0, 1].
        logits: (Optional[torch.FloatTensor]): optional raw logits if using classification bins.
    """

    compression_ratio: torch.FloatTensor
    logits: Optional[torch.FloatTensor] = None


class CompressionPredictorPreTrainedModel(PreTrainedModel):
    config_class = CompressionPredictorConfig
    base_model_prefix = "compression_predictor"

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()


class CompressionPredictorModel(CompressionPredictorPreTrainedModel):
    """
    Barebones predictor that will later learn to map:
        (compressed_hidden_states, delta_embedding) -> compression_ratio

    This is a non-functional stub for registration and configuration only.
    """

    def __init__(self, config: CompressionPredictorConfig):
        super().__init__(config)
        # Intentionally minimal; actual modeling to be implemented later.
        self.post_init()

    def forward(
        self,
        compressed_hidden_states: torch.FloatTensor,
        delta: Optional[torch.FloatTensor] = None,
        seq_lens: Optional[torch.LongTensor] = None,
        return_logits: bool = False,
        **kwargs,
    ) -> CompressionPredictorOutput:
        """
        Args:
            compressed_hidden_states: Tensor of shape [batch, seq_c, hidden_size]
            delta: Optional scalar(s) per batch item, shape [batch] or [batch, 1]
            seq_lens: Optional sequence lengths, shape [batch]
            return_logits: Whether to include raw logits output (if classification is used later).

        Returns:
            CompressionPredictorOutput with a placeholder compression_ratio.
        """
        batch = compressed_hidden_states.shape[0]

        # Placeholder prediction: zeros tensor in [0, 1] range.
        # Replace with real logic when implementing the model.
        compression_ratio = torch.zeros(batch, device=compressed_hidden_states.device, dtype=torch.float32)
        logits = torch.zeros(batch, 1, device=compressed_hidden_states.device, dtype=torch.float32) if return_logits else None

        return CompressionPredictorOutput(
            compression_ratio=compression_ratio,
            logits=logits,
        )

