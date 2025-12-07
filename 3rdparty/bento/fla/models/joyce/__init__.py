# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.joyce.configuration_joyce import JoyceConfig
from fla.models.joyce.modeling_joyce import (
    JoyceModel,
    JoyceForCausalLM,
)

AutoConfig.register(JoyceConfig.model_type, JoyceConfig, exist_ok=True)
AutoModel.register(JoyceConfig, JoyceModel, exist_ok=True)
AutoModelForCausalLM.register(JoyceConfig, JoyceForCausalLM, exist_ok=True)

__all__ = [
    "JoyceConfig",
    "JoyceModel",
    "JoyceForCausalLM",
]
