from __future__ import annotations
from typing import Optional, Dict, Any
from transformers import PretrainedConfig

class JoyceAEConfig(PretrainedConfig):
    model_type = "joyce_pretrain"

    def __init__(
        self,
        # --- standard knobs (accepted for symmetry; not strictly required) ---
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        use_cache: bool = False,
        torch_dtype: Optional[str] = None,
        initializer_range: float = 0.02,
        hidden_act: str = "swish",
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = False,
        attention_bias: bool = False,
        norm_eps: float = 1e-5,

        # --- geometry; will be auto-filled from base_model if missing ---
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        num_hidden_layers: Optional[int] = None,
        max_position_embeddings: Optional[int] = None,
        vocab_size: Optional[int] = None,

        # --- JOYCE specific ---
        compress_after_layer: int = 12,
        seq_len: int = 8192,
        num_compressed_tokens: int = 1024,
        latent_dim: int = 256,
        compression_depth: int = 1,
        upsample_depth: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        tie_mixers: bool = True,
        use_refine_attn: bool = True,

        # --- base model descriptor ---
        base_model: Optional[Dict[str, Any]] = None,

        **kwargs,
    ):
        super().__init__(**kwargs)

        # store “standard” keys unchanged so your infra can log/echo them
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.torch_dtype = torch_dtype
        self.initializer_range = initializer_range
        self.hidden_act = hidden_act
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        self.attention_bias = attention_bias
        self.norm_eps = norm_eps

        # geometry (may be None for now; we’ll fill from the base model)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size

        # joyce
        self.compress_after_layer = compress_after_layer
        self.seq_len = seq_len
        self.num_compressed_tokens = num_compressed_tokens
        self.latent_dim = latent_dim
        self.compression_depth = compression_depth
        self.upsample_depth = upsample_depth
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.tie_mixers = tie_mixers
        self.use_refine_attn = use_refine_attn
        self.base_model = base_model or {}
