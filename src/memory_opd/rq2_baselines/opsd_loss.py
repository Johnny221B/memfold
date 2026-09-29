"""Official-style generalized JSD objective for the RQ2 OPSD baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    token_clip: float | None = 0.05,
) -> torch.Tensor:
    """Match the full student vocabulary to a frozen teacher with generalized JSD.

    This follows ``siyan-zhao/OPSD``: the mixture assigns ``1-beta`` to the
    student and ``beta`` to the teacher. Clipping is applied to each vocabulary
    contribution before it is summed for a token, matching the updated official
    implementation. ``labels == -100`` positions are excluded.
    """

    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("student and teacher logits must have the same [..., vocab] shape")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if token_clip is not None and token_clip <= 0.0:
        raise ValueError("token_clip must be positive when provided")
    if labels is not None and labels.shape != student_logits.shape[:-1]:
        raise ValueError("labels must match the non-vocabulary logit dimensions")

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.detach() / temperature, dim=-1)
    if beta == 0.0:
        contributions = F.kl_div(
            student_log_probs, teacher_log_probs, reduction="none", log_target=True
        )
    elif beta == 1.0:
        contributions = F.kl_div(
            teacher_log_probs, student_log_probs, reduction="none", log_target=True
        )
    else:
        beta_tensor = student_logits.new_tensor(beta)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                (
                    student_log_probs + torch.log1p(-beta_tensor),
                    teacher_log_probs + torch.log(beta_tensor),
                )
            ),
            dim=0,
        )
        teacher_kl = F.kl_div(
            mixture_log_probs, teacher_log_probs, reduction="none", log_target=True
        )
        student_kl = F.kl_div(
            mixture_log_probs, student_log_probs, reduction="none", log_target=True
        )
        contributions = beta_tensor * teacher_kl + (1.0 - beta_tensor) * student_kl
    if token_clip is not None:
        contributions = contributions.clamp(max=token_clip)
    per_token = contributions.sum(dim=-1)
    if labels is not None:
        selected = per_token[labels != -100]
        if selected.numel() == 0:
            raise ValueError("labels mask excludes every token")
        return selected.mean()
    return per_token.mean()
