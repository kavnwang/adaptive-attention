from __future__ import annotations
from typing import Dict, Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from fla.models.joyce.config_joyce import JoyceAEConfig
from fla.models.joyce.modeling_joyce_pretrain import JoyceAutoencoderForPreTraining

def build_model(cfg_dict: Dict[str, Any]):
    """
    Standard entrypoint used by llmonade.train to create the model from
    a JSON. Mirrors transformer builder signature.
    """
    jcfg = JoyceAEConfig(**cfg_dict)

    # Resolve base model
    base_spec = jcfg.base_model or {}
    base_cfg_src = base_spec.get("config_path") or base_spec.get("hub_id")
    assert base_cfg_src, "joyce_pretrain: provide base_model.config_path or base_model.hub_id"

    base_cfg = AutoConfig.from_pretrained(base_cfg_src, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_config(base_cfg)

    # If geometry wasn’t specified in JSON, inherit from base model.
    if jcfg.hidden_size is None: jcfg.hidden_size = getattr(base_cfg, "hidden_size")
    if jcfg.num_heads is None: jcfg.num_heads = getattr(base_cfg, "num_attention_heads", None) or getattr(base_cfg, "num_heads")
    if jcfg.num_kv_heads is None: jcfg.num_kv_heads = getattr(base_cfg, "num_key_value_heads", None) or jcfg.num_heads
    if jcfg.num_hidden_layers is None: jcfg.num_hidden_layers = getattr(base_cfg, "num_hidden_layers")
    if jcfg.max_position_embeddings is None: jcfg.max_position_embeddings = getattr(base_cfg, "max_position_embeddings", jcfg.seq_len)
    if jcfg.vocab_size is None: jcfg.vocab_size = getattr(base_cfg, "vocab_size")

    # Sanity: keep seq_len ≤ max_position_embeddings unless you know what you’re doing.
    if jcfg.seq_len > int(jcfg.max_position_embeddings or jcfg.seq_len):
        raise ValueError(f"joyce_pretrain: seq_len={jcfg.seq_len} exceeds base max_position_embeddings={jcfg.max_position_embeddings}")

    model = JoyceAutoencoderForPreTraining(base_model=base_model, cfg=jcfg)
    
    # Ensure Joyce components have the same dtype as the base model
    base_dtype = next(base_model.parameters()).dtype
    model = model.to(dtype=base_dtype)
    
    return model
