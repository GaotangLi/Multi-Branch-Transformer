"""Transformers loading and generation for Multi-Branch Transformer.

This portable reference depends only on PyTorch and Transformers. It uses
standard SDPA and supports an actual incremental KV cache, including Joint-KV.
"""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_mbt import MBTConfig
from .modeling_mbt_core import MBTCore, MBTRMSNorm


class MBTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MBTConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["MBTLayer", "MBTMoE"]
    _tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}
    _keep_in_fp32_modules_strict = ["expert_bias"]
    _supports_sdpa = True
    _supports_flash_attn = False

    def __init__(self, config):
        super().__init__(config)
        self.model = MBTCore(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, MBTRMSNorm):
            nn.init.ones_(module.weight)

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    @classmethod
    def _supports_default_dynamic_cache(cls):
        # Physical blocks and cache entries differ with Joint-KV. Build our
        # DynamicCache from actual shared-cache tensors, rather than one per block.
        return False

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, labels=None,
        use_cache=None, output_attentions=False, output_hidden_states=False,
        return_dict=None, logits_to_keep=0, **kwargs,
    ):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if past_key_values is not None:
            if not isinstance(past_key_values, DynamicCache):
                raise TypeError("MBT supports DynamicCache; static/quantized caches are not implemented")
            if getattr(past_key_values, "offloading", False):
                raise ValueError("Offloaded caches are not supported by this reference runtime")
            legacy = tuple((entry[0], entry[1]) for entry in past_key_values)
            if not legacy or all(key is None for key, _ in legacy):
                legacy = None
        else:
            legacy = None
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=legacy, use_cache=use_cache, inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states, output_attentions=output_attentions,
        )
        hidden = output.last_hidden_state
        if labels is not None and not (isinstance(logits_to_keep, int) and logits_to_keep == 0):
            raise ValueError("Use all logits (logits_to_keep=0) when supplying labels")
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden = hidden[:, -logits_to_keep:, :]
        elif isinstance(logits_to_keep, torch.Tensor):
            hidden = hidden[:, logits_to_keep, :]
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].float().contiguous()
            shift_labels = labels[:, 1:].to(logits.device).contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        cache = DynamicCache(output.past_key_values) if use_cache else None
        result = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=cache,
            hidden_states=output.hidden_states, attentions=output.attentions,
        )
        return result if return_dict else result.to_tuple()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, position_ids=None, next_sequence_length=None,
        is_first_iteration=False, **kwargs,
    ):
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past_length:
            if next_sequence_length is not None:
                input_ids = input_ids[:, -next_sequence_length:]
            elif input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]
            else:
                input_ids = input_ids[:, -1:]
            inputs_embeds = None
        tokens = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
        if position_ids is not None:
            position_ids = position_ids[:, -tokens:]
        model_inputs = {
            "input_ids": input_ids.contiguous() if inputs_embeds is None else None,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }
        if "logits_to_keep" in kwargs:
            model_inputs["logits_to_keep"] = kwargs["logits_to_keep"]
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        past_key_values.reorder_cache(beam_idx)
        return past_key_values


MBTConfig.register_for_auto_class()
MBTForCausalLM.register_for_auto_class("AutoModelForCausalLM")
