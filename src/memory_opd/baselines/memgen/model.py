"""Scale-independent Stage-1 wrapper around the pinned official MemGen Weaver."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel
from torch import nn


def load_official_weaver(source: Path):
    path = source / "memgen/model/weaver.py"
    spec = importlib.util.spec_from_file_location("pinned_memgen_weaver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned MemGen Weaver from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MemGenWeaver


class MemGenWeaverStage1(nn.Module):
    """Frozen reasoner plus trainable official Weaver and bridge projections."""

    def __init__(
        self,
        reasoner: nn.Module,
        weaver_base: nn.Module,
        official_source: Path,
        *,
        prompt_latents: int = 8,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        weaver_adapter_path: Path | None = None,
    ) -> None:
        super().__init__()
        if prompt_latents <= 0:
            raise ValueError("prompt_latents must be positive")
        for parameter in reasoner.parameters():
            parameter.requires_grad_(False)
        self.reasoner = reasoner
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if weaver_adapter_path is None:
            peft_weaver = PeftModel(weaver_base, config, adapter_name="weaver")
        else:
            peft_weaver = PeftModel.from_pretrained(
                weaver_base,
                weaver_adapter_path,
                adapter_name="weaver",
                is_trainable=True,
            )
        weaver_class = load_official_weaver(official_source)
        self.weaver = weaver_class(
            peft_weaver, prompt_latents_len=prompt_latents, inference_latents_len=prompt_latents
        )
        reasoner_hidden = int(reasoner.config.hidden_size)
        weaver_hidden = int(weaver_base.config.hidden_size)
        self.reasoner_to_weaver = nn.Linear(reasoner_hidden, weaver_hidden)
        self.weaver_to_reasoner = nn.Linear(weaver_hidden, reasoner_hidden)

    def preserve_trainable_fp32(self) -> None:
        """Keep optimizer-owned tensors in FP32 after placing frozen compute in BF16."""
        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

    def augment_prompt(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the official prompt augmentation with explicit mixed-precision boundaries."""
        weaver_dtype = prompt_embeds.dtype
        weaver_inputs = self.reasoner_to_weaver(prompt_embeds.float()).to(weaver_dtype)
        batch_size = prompt_mask.shape[0]
        query = self.weaver.prompt_latent_ln(
            self.weaver.prompt_query_latents
        ) * self.weaver.prompt_latent_scale
        query = query.to(weaver_inputs.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        latent_mask = torch.ones(
            query.shape[:-1], dtype=prompt_mask.dtype, device=prompt_mask.device
        )
        latent_positions = (
            position_ids.max(dim=1)[0].unsqueeze(1)
            + torch.arange(query.shape[1], device=position_ids.device)
            + 1
        )
        output = self.weaver.model(
            inputs_embeds=torch.cat([weaver_inputs, query], dim=1),
            attention_mask=torch.cat([prompt_mask, latent_mask], dim=1),
            position_ids=torch.cat([position_ids.long(), latent_positions.long()], dim=1),
            output_hidden_states=True,
            return_dict=True,
        )
        return output.hidden_states[-1][:, -query.shape[1]:, :], latent_mask, latent_positions

    def project_to_reasoner(self, latents: torch.Tensor) -> torch.Tensor:
        reasoner_dtype = self.reasoner.get_input_embeddings().weight.dtype
        return self.weaver_to_reasoner(latents.float()).to(reasoner_dtype)

    def forward(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Any:
        if prompt_ids.shape != prompt_mask.shape or target_ids.shape != target_mask.shape:
            raise ValueError("IDs and masks must have matching shapes")
        position_ids = prompt_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(prompt_mask == 0, 0)
        prompt_embeds = self.reasoner.get_input_embeddings()(prompt_ids)
        latents, latent_mask, latent_positions = self.augment_prompt(
            prompt_embeds, prompt_mask, position_ids
        )
        reasoner_latents = self.project_to_reasoner(latents)
        target_embeds = self.reasoner.get_input_embeddings()(target_ids)
        inputs_embeds = torch.cat([prompt_embeds, reasoner_latents, target_embeds], dim=1)
        attention_mask = torch.cat([prompt_mask, latent_mask, target_mask], dim=1)
        target_positions = latent_positions[:, -1:] + target_mask.long().cumsum(-1)
        all_positions = torch.cat([position_ids, latent_positions, target_positions], dim=1)
        ignored = torch.full_like(prompt_ids, -100)
        latent_labels = torch.full_like(latent_mask, -100, dtype=target_ids.dtype)
        target_labels = target_ids.masked_fill(target_mask == 0, -100)
        labels = torch.cat([ignored, latent_labels, target_labels], dim=1)
        return self.reasoner(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=all_positions,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

    def trainable_parameter_groups(self) -> dict[str, int]:
        groups = {
            "weaver": sum(p.numel() for p in self.weaver.parameters() if p.requires_grad),
            "bridges": sum(
                p.numel()
                for module in (self.reasoner_to_weaver, self.weaver_to_reasoner)
                for p in module.parameters()
                if p.requires_grad
            ),
            "reasoner": sum(p.numel() for p in self.reasoner.parameters() if p.requires_grad),
        }
        return groups
