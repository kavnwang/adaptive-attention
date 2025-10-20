# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.transformer_continual.configuration_transformer_continual import TransformerContinualConfig
from fla.models.transformer_continual.modeling_transformer_continual import (
    TransformerContinualModel,
    TransformerContinualForCausalLM,
)

AutoConfig.register(TransformerContinualConfig.model_type, TransformerContinualConfig, exist_ok=True)
AutoModel.register(TransformerContinualConfig, TransformerContinualModel, exist_ok=True)
AutoModelForCausalLM.register(TransformerContinualConfig, TransformerContinualForCausalLM, exist_ok=True)

__all__ = [
    "TransformerContinualConfig",
    "TransformerContinualModel",
    "TransformerContinualForCausalLM",
]

