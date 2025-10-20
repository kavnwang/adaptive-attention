# 3rdparty/bento/fla/models/joyce/__init__.py
"""
Unified Joyce entry-point:
- `JoyceAEConfig` / `JoyceAutoencoderForPreTraining` expose the new standalone AE.
- `JoycePretrainConfig` / `JoyceAEPretrainModel` are kept for backward compatibility
  with earlier HF-style wrappers.
"""
from .config_joyce import JoyceAEConfig
from .modeling_joyce_pretrain import JoyceAutoencoderForPreTraining

# Backwards compatibility (HF-style pretrain wrappers)
try:
    from fla.models.joyce_model.configuration_joyce import JoycePretrainConfig
    from fla.models.joyce_model.modeling_joyce import JoyceAEPretrainModel
except ModuleNotFoundError:  # pragma: no cover - optional component
    JoycePretrainConfig = None
    JoyceAEPretrainModel = None

__all__ = [
    "JoyceAEConfig",
    "JoyceAutoencoderForPreTraining",
    "JoycePretrainConfig",
    "JoyceAEPretrainModel",
]

# Optional Auto* registry hook
try:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    AutoConfig.register(JoyceAEConfig.model_type, JoyceAEConfig)
    if JoycePretrainConfig is not None and JoyceAEPretrainModel is not None:
        AutoConfig.register(JoycePretrainConfig.model_type, JoycePretrainConfig, exist_ok=True)
        AutoModel.register(JoycePretrainConfig, JoyceAEPretrainModel, exist_ok=True)
        AutoModelForCausalLM.register(JoycePretrainConfig, JoyceAEPretrainModel, exist_ok=True)
except Exception:
    pass
