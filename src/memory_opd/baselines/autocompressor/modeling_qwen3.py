"""Qwen3 port of the independently implemented AutoCompressor interface."""
from __future__ import annotations
import torch
from torch import nn
from transformers import Qwen3ForCausalLM
from .modeling_qwen2 import Qwen2AutoCompressorForCausalLM


class Qwen3AutoCompressorForCausalLM(Qwen3ForCausalLM):
    """Qwen3 equivalent; recurrent mechanics are architecture-neutral."""

    def __init__(self, config):
        summary_length = int(getattr(config, "summary_length", 0))
        if summary_length < 0:
            raise ValueError("config.summary_length must be non-negative")
        config.summary_length = summary_length
        config.accumulate_summary = bool(getattr(config, "accumulate_summary", True))
        super().__init__(config)
        self.embed_summary = nn.Embedding(summary_length, config.hidden_size)
        if summary_length:
            eos_token_id = config.eos_token_id
            if isinstance(eos_token_id, (list, tuple)): eos_token_id = eos_token_id[0]
            if eos_token_id is None: eos_token_id = 0
            with torch.no_grad():
                eos = self.get_input_embeddings().weight[int(eos_token_id)]
                self.embed_summary.weight.copy_(eos.expand_as(self.embed_summary.weight))

    _summary_token_embeds = Qwen2AutoCompressorForCausalLM._summary_token_embeds
    _forward_segment = Qwen2AutoCompressorForCausalLM._forward_segment
    forward = Qwen2AutoCompressorForCausalLM.forward
