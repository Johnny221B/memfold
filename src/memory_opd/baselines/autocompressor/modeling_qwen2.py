"""AutoCompressor-style recurrent summary vectors for Qwen2/Qwen2.5.

This is an independent adaptation based on the algorithm described by
Chevalier et al. (EMNLP 2023), not copied upstream source.  A segment is
processed as ``[prior summaries, segment tokens, learned summary tokens]``.
The final hidden states at the learned summary-token positions become the
summary vectors supplied to the next segment.

Reproduction assumption: Qwen2 uses ordinary consecutive RoPE positions for
the complete composed sequence.  The official repository only implements OPT
and a custom Llama-2 FlashAttention model and does not specify Qwen2 behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class AutoCompressorOutput(CausalLMOutputWithPast):
    """Causal-LM output plus the accumulated summary-vector soft prompt."""

    softprompt: Optional[torch.FloatTensor] = None


def _normalize_segment_lengths(
    total_length: int, segment_lengths: Optional[Union[int, Sequence[int]]]
) -> list[int]:
    if segment_lengths is None:
        return [total_length]
    if isinstance(segment_lengths, int):
        if segment_lengths <= 0:
            raise ValueError("segment length must be positive")
        full, remainder = divmod(total_length, segment_lengths)
        lengths = [segment_lengths] * full
        if remainder:
            lengths.append(remainder)
        return lengths or [total_length]

    lengths = [int(length) for length in segment_lengths]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("segment_lengths must contain positive integers")
    if sum(lengths) != total_length:
        raise ValueError(
            f"segment_lengths sum to {sum(lengths)}, expected {total_length}"
        )
    return lengths


class Qwen2AutoCompressorForCausalLM(Qwen2ForCausalLM):
    """Qwen2 causal LM augmented with recurrent learned summary tokens."""

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
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0]
            if eos_token_id is None:
                eos_token_id = 0
            with torch.no_grad():
                eos_embedding = self.get_input_embeddings().weight[int(eos_token_id)]
                self.embed_summary.weight.copy_(eos_embedding.expand_as(self.embed_summary.weight))

    def _summary_token_embeds(self, batch_size: int, device: torch.device) -> torch.Tensor:
        indices = torch.arange(self.config.summary_length, device=device)
        indices = indices.unsqueeze(0).expand(batch_size, -1)
        return self.embed_summary(indices)

    def _forward_segment(
        self,
        segment_embeds: torch.Tensor,
        segment_attention_mask: torch.Tensor,
        softprompt: torch.Tensor,
        append_summary: bool,
        output_hidden_states: bool,
        output_attentions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        batch_size = segment_embeds.size(0)
        summary_embeds = self._summary_token_embeds(batch_size, segment_embeds.device)
        summary_embeds = summary_embeds.to(segment_embeds.dtype)
        if not append_summary:
            summary_embeds = summary_embeds[:, :0]

        composed = torch.cat([softprompt.to(segment_embeds.dtype), segment_embeds, summary_embeds], dim=1)
        prefix_mask = torch.ones(
            batch_size,
            softprompt.size(1),
            dtype=segment_attention_mask.dtype,
            device=segment_attention_mask.device,
        )
        summary_mask = torch.ones(
            batch_size,
            summary_embeds.size(1),
            dtype=segment_attention_mask.dtype,
            device=segment_attention_mask.device,
        )
        composed_mask = torch.cat([prefix_mask, segment_attention_mask, summary_mask], dim=1)
        position_ids = composed_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(composed_mask == 0, 0)

        outputs = self.model(
            inputs_embeds=composed,
            attention_mask=composed_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        prefix_length = softprompt.size(1)
        segment_length = segment_embeds.size(1)
        segment_hidden = hidden[:, prefix_length : prefix_length + segment_length]
        new_summary = hidden[:, prefix_length + segment_length :]
        return segment_hidden, new_summary, outputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        segment_lengths: Optional[Union[int, Sequence[int]]] = None,
        softprompt: Optional[torch.FloatTensor] = None,
        output_softprompt: bool = False,
        detach_softprompt_between_segments: bool = False,
        detach_softprompt_every_n_segments: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> AutoCompressorOutput:
        if kwargs.get("past_key_values") is not None:
            raise NotImplementedError("cached generation is implemented separately after training precheck")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("pass input_ids or inputs_embeds, not both")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.get_input_embeddings()(input_ids)

        batch_size, total_length = inputs_embeds.shape[:2]
        lengths = _normalize_segment_lengths(total_length, segment_lengths)
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, total_length, dtype=torch.long, device=inputs_embeds.device
            )
        if attention_mask.shape != (batch_size, total_length):
            raise ValueError("attention_mask shape must match the token sequence")
        if labels is not None and labels.shape != (batch_size, total_length):
            raise ValueError("labels shape must match the token sequence")

        if softprompt is None:
            softprompt = inputs_embeds[:, :0]
        if softprompt.ndim != 3 or softprompt.shape[0] != batch_size:
            raise ValueError("softprompt must have shape [batch, summary_tokens, hidden]")

        embed_segments = torch.split(inputs_embeds, lengths, dim=1)
        mask_segments = torch.split(attention_mask, lengths, dim=1)
        token_hidden_states: list[torch.Tensor] = []
        last_outputs = None
        if detach_softprompt_between_segments:
            if detach_softprompt_every_n_segments is not None:
                raise ValueError("use only one softprompt detach option")
            detach_softprompt_every_n_segments = 1
        if detach_softprompt_every_n_segments is not None and detach_softprompt_every_n_segments <= 0:
            raise ValueError("detach_softprompt_every_n_segments must be positive")

        for index, (segment_embeds, segment_mask) in enumerate(zip(embed_segments, mask_segments)):
            append_summary = index < len(embed_segments) - 1 or output_softprompt
            segment_hidden, new_summary, last_outputs = self._forward_segment(
                segment_embeds,
                segment_mask,
                softprompt,
                append_summary,
                bool(output_hidden_states),
                bool(output_attentions),
            )
            token_hidden_states.append(segment_hidden)
            if new_summary.size(1):
                if self.config.accumulate_summary:
                    softprompt = torch.cat([softprompt, new_summary], dim=1)
                else:
                    softprompt = new_summary
            if (
                detach_softprompt_every_n_segments is not None
                and (index + 1) % detach_softprompt_every_n_segments == 0
                and index + 1 < len(embed_segments)
            ):
                softprompt = softprompt.detach()

        hidden = torch.cat(token_hidden_states, dim=1)
        logits = self.lm_head(hidden).float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = AutoCompressorOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=last_outputs.hidden_states if output_hidden_states else None,
            attentions=last_outputs.attentions if output_attentions else None,
            softprompt=softprompt,
        )
        if return_dict is False:
            return tuple(value for value in output.values() if value is not None)
        return output
