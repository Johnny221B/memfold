"""SEED-style mixed OPD/GRPO objective for fixed-K soft-memory answers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AdvantageMetrics:
    zero_variance: bool
    reward_variance: float


@dataclass(frozen=True)
class SoftSeedMetrics:
    opd_loss: torch.Tensor
    grpo_loss: torch.Tensor
    reference_kl: torch.Tensor
    gate_mean: torch.Tensor
    gate_active_ratio: torch.Tensor
    teacher_gap_mean: torch.Tensor


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(device=value.device, dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1)


def group_normalized_advantages(
    rewards: torch.Tensor, *, epsilon: float = 1e-6
) -> tuple[torch.Tensor, AdvantageMetrics]:
    if rewards.ndim != 1 or not rewards.numel():
        raise ValueError("rewards must be a non-empty vector")
    variance = rewards.float().var(unbiased=False)
    zero = float(variance.detach().cpu()) <= epsilon
    advantages = torch.zeros_like(rewards) if zero else (
        (rewards - rewards.mean()) / (rewards.std(unbiased=False) + epsilon)
    )
    return advantages, AdvantageMetrics(zero, float(variance.detach().cpu()))


def soft_seed_mixed_loss(
    *,
    current_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    reference_log_prob: torch.Tensor | None = None,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    opd_weight: float,
    grpo_weight: float,
    reference_kl_weight: float = 0.0,
    gate_beta: float = 5.0,
    clip_range: float = 0.2,
) -> tuple[torch.Tensor, SoftSeedMetrics]:
    if not math.isfinite(reference_kl_weight) or reference_kl_weight < 0:
        raise ValueError("reference_kl_weight must be finite and nonnegative")
    if reference_kl_weight > 0 and reference_log_prob is None:
        raise ValueError("reference_log_prob is required when reference KL is enabled")
    tensors = (old_log_prob, teacher_log_prob, response_mask)
    if reference_kl_weight > 0:
        tensors += (reference_log_prob,)
    if current_log_prob.ndim != 2 or any(
        value.shape != current_log_prob.shape for value in tensors
    ):
        raise ValueError("all token tensors must share [group, response] shape")
    if advantages.shape != current_log_prob.shape[:1]:
        raise ValueError("advantages must have shape [group]")
    mask = response_mask.to(current_log_prob.dtype)
    teacher = teacher_log_prob.detach()
    gap = (teacher - current_log_prob.detach()).detach()
    gate = torch.sigmoid(float(gate_beta) * gap).detach()
    opd_loss = _masked_mean(gate * (teacher - current_log_prob), mask)

    ratio = torch.exp(current_log_prob - old_log_prob.detach())
    token_advantage = advantages[:, None].to(current_log_prob.dtype)
    unclipped = ratio * token_advantage
    clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * token_advantage
    grpo_loss = -_masked_mean(torch.minimum(unclipped, clipped), mask)

    # Do not evaluate the reference branch when disabled: 0 * inf is NaN.
    reference_kl = current_log_prob.new_zeros(())
    if reference_kl_weight > 0:
        ref_gap = reference_log_prob.detach() - current_log_prob
        reference_kl = _masked_mean(torch.exp(ref_gap) - ref_gap - 1.0, mask)
    loss = (
        float(opd_weight) * opd_loss
        + float(grpo_weight) * grpo_loss
        + float(reference_kl_weight) * reference_kl
    )
    metrics = SoftSeedMetrics(
        opd_loss=opd_loss.detach(),
        grpo_loss=grpo_loss.detach(),
        reference_kl=reference_kl.detach(),
        gate_mean=_masked_mean(gate, mask).detach(),
        gate_active_ratio=_masked_mean((gate > 0.5).float(), mask).detach(),
        teacher_gap_mean=_masked_mean(gap, mask).detach(),
    )
    return loss, metrics
