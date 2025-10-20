# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules.l2warp import l2_warp
from transformers.utils.deprecation import deprecate_kwarg

from fla.models.autoencoder.modeling_autoencoder import (
    AutoencoderPreTrainedModel as BasePreTrained,
    AutoencoderModel as BaseModel,
    AutoencoderForCausalLM as BaseForCausalLM,
)
from fla.models.autoencoder_continual.configuration_autoencoder_continual import (
    AutoencoderContinualConfig,
)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


class AutoencoderContinualPreTrainedModel(BasePreTrained):
    config_class = AutoencoderContinualConfig


class AutoencoderContinualModel(AutoencoderContinualPreTrainedModel, BaseModel):
    """
    Thin wrapper so training can use model_type="autoencoder_continual"
    while reusing the base autoencoder implementation.
    """

    def __init__(self, config: AutoencoderContinualConfig) -> "AutoencoderContinualModel":
        super().__init__(config)


class AutoencoderContinualForCausalLM(AutoencoderContinualPreTrainedModel, BaseForCausalLM):
    def __init__(self, config: AutoencoderContinualConfig):
        super().__init__(config)

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Any]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        masked_tokens_for_loss = int(getattr(self.config, "masked_tokens", 0) or 0)
        compression_tokens_for_loss = int(getattr(self.config, "compression_tokens", 0) or 0)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        model_outputs = outputs
        ae_inputs = None
        if isinstance(outputs, tuple) and len(outputs) == 2:
            ae_inputs, model_outputs = outputs

        hidden_states = (
            model_outputs.last_hidden_state if return_dict and hasattr(model_outputs, "last_hidden_state") else outputs[0]
        )

        # Optional logits for last-k; not used for loss
        logits = None if getattr(self.config, "fuse_linear_cross_entropy", False) else self.lm_head(
            hidden_states[:, -logits_to_keep:]
        )

        loss = None
        if labels is not None:
            # Build criterion
            if getattr(self, "criterion", None) is None:
                if getattr(self.config, "fuse_linear_cross_entropy", False):
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=getattr(self.config, "use_l2warp", False))
                elif getattr(self.config, "fuse_cross_entropy", False):
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion

            labels = labels.to(hidden_states.device)
            # Shift left by 1 for next-token prediction
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)

            # Determine which part to supervise and align labels/predictions
            seq_len_out = hidden_states.size(1)
            if masked_tokens_for_loss > 0:
                m = masked_tokens_for_loss
                c = min(compression_tokens_for_loss, m)
                hs_tail = hidden_states[:, c:, :]
                labels_tail = labels[:, m:]
                tail_len = min(hs_tail.size(1), labels_tail.size(1))
                hs_for_loss = hs_tail[:, :tail_len, :]
                labels_for_loss = labels_tail[:, :tail_len]
            else:
                keep_len = min(seq_len_out, labels.size(1))
                hs_for_loss = hidden_states[:, :keep_len, :]
                labels_for_loss = labels[:, :keep_len]

            if getattr(self.config, "fuse_linear_cross_entropy", False):
                loss = criterion(hs_for_loss, labels_for_loss, self.lm_head.weight, self.lm_head.bias)
            else:
                logits_for_loss = self.lm_head(hs_for_loss)
                loss = criterion(
                    logits_for_loss.contiguous().view(-1, logits_for_loss.size(-1)),
                    labels_for_loss.contiguous().view(-1),
                )
                loss = l2_warp(loss, logits_for_loss) if getattr(self.config, "use_l2warp", False) else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values if hasattr(model_outputs, "past_key_values") else None,
            hidden_states=model_outputs.hidden_states if hasattr(model_outputs, "hidden_states") else None,
            attentions=model_outputs.attentions if hasattr(model_outputs, "attentions") else None,
        )

# -*- coding: utf-8 -*-



import math
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.layers.compress import Compress
from fla.layers.upsample import Upsample
from fla.models.autoencoder.configuration_autoencoder import AutoencoderConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules import GatedMLP as TransformerMLP
from fla.modules import RMSNorm
from fla.modules.l2warp import l2_warp

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


class TransformerBlock(GradientCheckpointingLayer):

    def __init__(self, config: AutoencoderConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx
        )

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[Any]
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class AutoencoderPreTrainedModel(PreTrainedModel):

    config_class = AutoencoderConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['TransformerBlock']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class AutoencoderModel(AutoencoderPreTrainedModel):

    def __init__(
        self,
        config: AutoencoderConfig
    ) -> AutoencoderModel:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.compress = Compress(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            seq_len=config.seq_len,
            compression_ratio=config.compression_ratio,
            compression_depth=config.compression_depth
        )
        self.upsample = Upsample(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            seq_len=int(config.seq_len * config.compression_ratio),
            upsample_ratio=1.0 / config.compression_ratio,
            upsample_depth=config.upsample_depth
        )

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Any]
    ) -> Tuple[torch.FloatTensor, Union[Tuple, CausalLMOutputWithPast]]:
        if output_attentions:
            warnings.warn(
                "`TransformerModel` does not support output attention weights now, so `output_attentions` is set to `False`."
            )
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # embed positions
        hidden_states = inputs_embeds
        masked_tokens: int = self.config.masked_tokens
        compression_tokens: int = self.config.compression_tokens
        compression_layer_idx: int = self.config.compression_layer_idx

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = None


        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if (compression_layer_idx is not None) and (idx <= compression_layer_idx) and (masked_tokens > 0):
                bsz, seq_len, _ = hidden_states.size()
                m = min(masked_tokens, seq_len)
                prefix = hidden_states[:, :m, :]
                tail = hidden_states[:, m:, :]
                mask_prefix = (attention_mask[:, :m] if attention_mask is not None else None)
                mask_tail = (attention_mask[:, m:] if attention_mask is not None else None)
                out_prefix = layer(
                    prefix,
                    attention_mask=mask_prefix,
                    past_key_values=None,
                    output_attentions=False,
                    use_cache=False,
                    seqlen_offsets=0,
                    **kwargs,
                )[0]
                if tail.size(1) > 0:
                    out_tail = layer(
                        tail,
                        attention_mask=mask_tail,
                        past_key_values=None,
                        output_attentions=False,
                        use_cache=False,
                        seqlen_offsets=m,
                        **kwargs,
                    )[0]
                else:
                    out_tail = tail
                hidden_states = torch.cat([out_prefix, out_tail], dim=1)
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs
                )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_attns += (layer_outputs[1],)
            if compression_layer_idx is not None and idx == compression_layer_idx and masked_tokens > 0:
                bsz, seq_len, dim = hidden_states.size()
                m = min(masked_tokens, seq_len)
                c = min(compression_tokens, m)
                prefix = hidden_states[:, :m, :]
                tail = hidden_states[:, m:, :]
                compressed_prefix = self.compress(prefix)
                hidden_states = torch.cat([compressed_prefix, tail], dim=1)

        hidden_states = self.norm(hidden_states)
        if compression_layer_idx is None:
            inputs = hidden_states
            hidden_states = self.compress(hidden_states)
            hidden_states = self.upsample(hidden_states)
            hidden_states = self.norm(hidden_states)
        else:
            inputs = None

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attns] if v is not None)

        return (inputs, BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        ))


class AutoencoderForCausalLM(AutoencoderPreTrainedModel, FLAGenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoencoderModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Any]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        masked_tokens_for_loss = self.config.masked_tokens
        compression_tokens_for_loss = self.config.compression_tokens

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        model_outputs = outputs
        ae_inputs = None
        if isinstance(outputs, tuple) and len(outputs) == 2:
            ae_inputs, model_outputs = outputs

        hidden_states = (
            model_outputs.last_hidden_state
            if return_dict and hasattr(model_outputs, "last_hidden_state")
            else outputs[0]
        )

        logits = None if self.config.fuse_linear_cross_entropy else self.lm_head(hidden_states[:, -logits_to_keep:])

        # Next-token prediction loss with masked prefix excluded
        loss = None
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            # Shift left by 1 for next-token prediction
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            # Determine which part to supervise and align labels/predictions
            seq_len_out = hidden_states.size(1)
            if masked_tokens_for_loss > 0:
                # Align tail: predictions at indices [c:] correspond to labels at [m:]
                m = masked_tokens_for_loss
                c = min(compression_tokens_for_loss, m)
                hs_tail = hidden_states[:, c:, :]
                labels_tail = labels[:, m:]
                tail_len = min(hs_tail.size(1), labels_tail.size(1))
                hs_for_loss = hs_tail[:, :tail_len, :]
                labels_for_loss = labels_tail[:, :tail_len]
            else:
                # Default: full sequence up to available outputs
                keep_len = min(seq_len_out, labels.size(1))
                hs_for_loss = hidden_states[:, :keep_len, :]
                labels_for_loss = labels[:, :keep_len]
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hs_for_loss, labels_for_loss, self.lm_head.weight, self.lm_head.bias)
            else:
                logits_for_loss = self.lm_head(hs_for_loss)
                loss = criterion(
                    logits_for_loss.contiguous().view(-1, logits_for_loss.size(-1)),
                    labels_for_loss.contiguous().view(-1)
                )
                loss = l2_warp(loss, logits_for_loss) if self.config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )
