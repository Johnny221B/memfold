"""Small, testable pieces of the ALFWorld adaptation of MemGen Trigger GRPO."""

from __future__ import annotations

import torch


INVALID_TRIGGER_ACTION = -100


def group_advantages(rewards: list[float]) -> torch.Tensor:
    """Return upstream-style within-group standardized rewards."""
    values = torch.tensor(rewards, dtype=torch.float32)
    if values.numel() < 2:
        raise ValueError("Trigger GRPO requires at least two generations")
    centered = values - values.mean()
    return centered / (values.std(unbiased=True) + 1e-4)


def selected_trigger_logps(
    logits: torch.Tensor,
    prompt_length: int,
    augmentation_mask: torch.Tensor,
) -> torch.Tensor:
    """Align Trigger logits with official generation decisions and select log-p."""
    if logits.ndim != 3 or logits.size(-1) != 2:
        raise ValueError("Trigger logits must have shape [batch, sequence, 2]")
    if augmentation_mask.ndim != 2:
        raise ValueError("augmentation_mask must have shape [batch, completion]")
    clipped = logits[:, prompt_length - 1 : -1]
    if clipped.shape[:2] != augmentation_mask.shape:
        raise ValueError(
            f"unaligned Trigger tensors: logits={tuple(clipped.shape)} "
            f"mask={tuple(augmentation_mask.shape)}"
        )
    valid = augmentation_mask != INVALID_TRIGGER_ACTION
    actions = augmentation_mask.masked_fill(~valid, 0)
    logps = clipped.log_softmax(dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return logps[valid]
