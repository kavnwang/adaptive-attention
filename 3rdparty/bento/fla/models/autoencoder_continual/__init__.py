# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.autoencoder_continual.configuration_autoencoder_continual import AutoencoderContinualConfig
from fla.models.autoencoder_continual.modeling_autoencoder_continual import (
    AutoencoderContinualModel,
    AutoencoderContinualForCausalLM,
)

AutoConfig.register(AutoencoderContinualConfig.model_type, AutoencoderContinualConfig, exist_ok=True)
AutoModel.register(AutoencoderContinualConfig, AutoencoderContinualModel, exist_ok=True)
AutoModelForCausalLM.register(AutoencoderContinualConfig, AutoencoderContinualForCausalLM, exist_ok=True)

__all__ = [
    "AutoencoderContinualConfig",
    "AutoencoderContinualModel",
    "AutoencoderContinualForCausalLM",
]

