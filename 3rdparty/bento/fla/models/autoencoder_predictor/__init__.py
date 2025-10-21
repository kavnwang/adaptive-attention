# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

# Use the predictor variant (this module) rather than continual
from fla.models.autoencoder_predictor.configuration_autoencoder_continual import (
    AutoencoderPredictorConfig,
)
from fla.models.autoencoder_predictor.modeling_autoencoder_continual import (
    AutoencoderModel as AutoencoderPredictorModelImpl,
    AutoencoderForCausalLM as AutoencoderPredictorForCausalLMImpl,
)

# Register the predictor model type and mappings
AutoConfig.register(
    AutoencoderPredictorConfig.model_type, AutoencoderPredictorConfig, exist_ok=True
)
AutoModel.register(
    AutoencoderPredictorConfig, AutoencoderPredictorModelImpl, exist_ok=True
)
AutoModelForCausalLM.register(
    AutoencoderPredictorConfig, AutoencoderPredictorForCausalLMImpl, exist_ok=True
)

# Friendly exported names
AutoencoderPredictorModel = AutoencoderPredictorModelImpl
AutoencoderPredictorForCausalLM = AutoencoderPredictorForCausalLMImpl

__all__ = [
    "AutoencoderPredictorConfig",
    "AutoencoderPredictorModel",
    "AutoencoderPredictorForCausalLM",
]
