"""CPU oracle for SEED's gated OPD loss.

Training must use the pinned upstream implementation, not this oracle.  This
small function exists so objective math and gradient direction can be checked
without importing the complete veRL/Ray/vLLM stack.
"""

from __future__ import annotations

import torch


def compute_opd_loss_reference(
    log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    opd_step_mask: torch.Tensor | None = None,
    gate_beta: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return upstream-equivalent token-mean OPD loss and detached gate."""
    if log_prob.shape != teacher_log_prob.shape or log_prob.shape != response_mask.shape:
        raise ValueError("log_prob, teacher_log_prob, and response_mask must have identical shapes")
    mask = response_mask.to(dtype=log_prob.dtype)
    if opd_step_mask is not None:
        step_mask = opd_step_mask.to(device=log_prob.device, dtype=log_prob.dtype)
        if step_mask.ndim == 1:
            if step_mask.shape[0] != log_prob.shape[0]:
                raise ValueError("sample-level OPD mask has the wrong batch size")
            step_mask = step_mask.unsqueeze(-1)
        elif step_mask.shape != log_prob.shape:
            raise ValueError("token-level OPD mask has the wrong shape")
        mask = mask * step_mask
    teacher = teacher_log_prob.detach()
    gate = torch.sigmoid(float(gate_beta) * (teacher - log_prob.detach())).detach()
    denominator = mask.sum()
    if denominator.item() == 0:
        return log_prob.sum() * 0.0, gate
    return (gate * (teacher - log_prob) * mask).sum() / denominator, gate
