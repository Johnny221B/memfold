"""Pure Vanilla GRPO objective functions."""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Normalize rewards independently within each rollout group."""

    if rewards.ndim == 1:
        rewards = rewards.unsqueeze(0)
        squeeze = True
    elif rewards.ndim == 2:
        squeeze = False
    else:
        raise ValueError("rewards must have shape [G] or [B, G]")
    if rewards.shape[-1] < 2:
        raise ValueError("GRPO requires at least two samples per group")
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    scale = rewards.std(dim=-1, keepdim=True, unbiased=False)
    advantages = centered / (scale + epsilon)
    return advantages.squeeze(0) if squeeze else advantages


def clipped_grpo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float,
    mask: torch.Tensor | None = None,
    reference_logp: torch.Tensor | None = None,
    kl_beta: float = 0.0,
) -> torch.Tensor:
    """Token-masked GRPO surrogate with the standard low-variance KL penalty."""

    if new_logp.shape != old_logp.shape:
        raise ValueError("new_logp and old_logp must have identical shapes")
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must lie in (0, 1)")
    while advantages.ndim < new_logp.ndim:
        advantages = advantages.unsqueeze(-1)
    ratio = torch.exp(new_logp - old_logp)
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    per_token = -surrogate
    if reference_logp is not None:
        if reference_logp.shape != new_logp.shape:
            raise ValueError("reference_logp must match new_logp")
        if kl_beta < 0:
            raise ValueError("kl_beta must be non-negative")
        log_ratio = reference_logp - new_logp
        per_token = per_token + kl_beta * (torch.exp(log_ratio) - log_ratio - 1.0)
    if mask is None:
        return per_token.mean()
    if mask.shape != new_logp.shape:
        raise ValueError("mask must match log-probability shape")
    denominator = mask.to(new_logp.dtype).sum().clamp_min(1.0)
    return (per_token * mask).sum() / denominator
