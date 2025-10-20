# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.autoencoder.configuration_autoencoder import AutoencoderConfig
from fla.models.autoencoder.modeling_autoencoder import AutoencoderForCausalLM, AutoencoderModel

AutoConfig.register(AutoencoderConfig.model_type, AutoencoderConfig, exist_ok=True)
AutoModel.register(AutoencoderConfig, AutoencoderModel, exist_ok=True)
AutoModelForCausalLM.register(AutoencoderConfig, AutoencoderForCausalLM, exist_ok=True)


__all__ = ['AutoencoderConfig', 'AutoencoderForCausalLM', 'AutoencoderModel']
