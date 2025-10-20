# bento/fla/models/joyce_controller/configuration_joyce_controller.py
from transformers.configuration_utils import PretrainedConfig

class JoyceControllerConfig(PretrainedConfig):
    model_type = "joyce_controller"

    def __init__(
        self,
        hidden_size: int = 1024,                # dim of layer-L hidden states
        controller_hidden_size: int = 512,      # internal MLP dim
        num_delta_freqs: int = 32,              # sinusoidal freqs for Δ
        delta_embed_scale: float = 1.0,         # scales Δ before embedding
        max_abs_delta: float = 8.0,             # clamp |Δ|
        pooler: str = "mean_ln",                # ["mean_ln", "attn", "mean"]
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        # training / labeling conveniences
        ratio_min: float = 0.0,                 # lower bound for r in (0,1]
        ratio_max: float = 1.0,                 # upper bound for r
        # optional: how many compressed tokens were used to make the shrunken seq
        num_compressed_states: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.controller_hidden_size = controller_hidden_size
        self.num_delta_freqs = num_delta_freqs
        self.delta_embed_scale = delta_embed_scale
        self.max_abs_delta = max_abs_delta
        self.pooler = pooler
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.num_compressed_states = num_compressed_states
