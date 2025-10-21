# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel

from fla.models.compression_predictor.configuration_compression_predictor import (
    CompressionPredictorConfig,
)
from fla.models.compression_predictor.modeling_compression_predictor import (
    CompressionPredictorModel,
)


AutoConfig.register(
    CompressionPredictorConfig.model_type, CompressionPredictorConfig, exist_ok=True
)
AutoModel.register(CompressionPredictorConfig, CompressionPredictorModel, exist_ok=True)


__all__ = [
    "CompressionPredictorConfig",
    "CompressionPredictorModel",
]

