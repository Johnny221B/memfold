"""SEED-compatible sampled-token OPD objective.

The equations and aggregation follow jinyangwu/SEED revision
2cf2fadca3c5aba28da68e8e1405182ba8d90e6c, core_algos.compute_opd_loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SeedOPDMetrics:
    active_token_ratio: torch.Tensor
    gate_mean: torch.Tensor
    gate_active_ratio: torch.Tensor
    teacher_gap_mean: torch.Tensor


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1)


def seed_sampled_token_opd_loss(
    student_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    opd_step_mask: torch.Tensor | None = None,
    gate_beta: float = 5.0,
) -> tuple[torch.Tensor, SeedOPDMetrics]:
    """Score the same student on-policy tokens under ordinary/privileged contexts."""
    if student_log_prob.shape != teacher_log_prob.shape or student_log_prob.shape != response_mask.shape:
        raise ValueError(
            "student, teacher, and response mask must have identical [batch, response] shapes")
    mask = response_mask.to(device=student_log_prob.device,
                            dtype=student_log_prob.dtype)
    if opd_step_mask is not None:
        step = opd_step_mask.to(device=mask.device, dtype=mask.dtype)
        if step.ndim == 1:
            if step.shape[0] != mask.shape[0]:
                raise ValueError(
                    "sample-level OPD mask has the wrong batch size")
            step = step[:, None]
        elif step.shape != mask.shape:
            raise ValueError("token-level OPD mask has the wrong shape")
        mask = mask * step
    if not torch.any(mask > 0):
        zero = student_log_prob.sum() * 0.0
        metric_zero = student_log_prob.new_tensor(0.0)
        return zero, SeedOPDMetrics(metric_zero, metric_zero, metric_zero, metric_zero)
    teacher = teacher_log_prob.detach()
    gap = (teacher - student_log_prob.detach()).detach()
    gate = torch.sigmoid(float(gate_beta) * gap).detach()
    loss = _masked_mean(gate * (teacher - student_log_prob), mask)
    metrics = SeedOPDMetrics(
        active_token_ratio=(mask > 0).float().mean(),
        gate_mean=_masked_mean(gate, mask),
        gate_active_ratio=_masked_mean((gate > 0.5).float(), mask),
        teacher_gap_mean=_masked_mean(gap, mask),
    )
    return loss, metrics
