import torch

from memory_opd.opd.soft_seed import group_normalized_advantages, soft_seed_mixed_loss


def test_group_advantages_are_centered_and_detect_zero_variance():
    advantages, metrics = group_normalized_advantages(torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert torch.isclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert not metrics.zero_variance
    zero, zero_metrics = group_normalized_advantages(torch.ones(4))
    assert torch.equal(zero, torch.zeros(4))
    assert zero_metrics.zero_variance


def test_mixed_loss_has_policy_gradient_and_detaches_teacher():
    current = torch.tensor([[-0.4, -0.6], [-0.8, -0.2]], requires_grad=True)
    old = current.detach().clone()
    teacher = torch.tensor([[-0.2, -0.3], [-0.4, -0.1]], requires_grad=True)
    reference = current.detach().clone()
    loss, metrics = soft_seed_mixed_loss(
        current_log_prob=current,
        old_log_prob=old,
        teacher_log_prob=teacher,
        reference_log_prob=reference,
        advantages=torch.tensor([1.0, -1.0]),
        response_mask=torch.ones_like(current),
        opd_weight=0.01,
        grpo_weight=1.0,
        reference_kl_weight=0.01,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert current.grad is not None and torch.any(current.grad != 0)
    assert teacher.grad is None
    assert torch.isclose(metrics.reference_kl, torch.tensor(0.0), atol=1e-6)


def _loss_inputs():
    current = torch.tensor([[-0.4, -0.6]], requires_grad=True)
    return dict(current_log_prob=current, old_log_prob=current.detach().clone(),
                teacher_log_prob=torch.tensor([[-0.2, -0.3]]),
                advantages=torch.ones(1), response_mask=torch.ones_like(current),
                opd_weight=0.1, grpo_weight=1.0)


def test_default_zero_kl_needs_no_reference_and_ignores_nonfinite_reference():
    args = _loss_inputs()
    loss, metrics = soft_seed_mixed_loss(**args)
    bad_reference = torch.full_like(args['current_log_prob'], float('inf'))
    other, other_metrics = soft_seed_mixed_loss(**args, reference_log_prob=bad_reference)
    assert torch.isfinite(loss) and torch.equal(loss, other)
    assert metrics.reference_kl.item() == other_metrics.reference_kl.item() == 0
    loss.backward()
    assert torch.isfinite(args['current_log_prob'].grad).all()


def test_positive_kl_adds_expected_penalty_and_gradient():
    args = _loss_inputs()
    current = args['current_log_prob']
    reference = torch.tensor([[-0.8, -0.9]], requires_grad=True)
    base, _ = soft_seed_mixed_loss(**args)
    loss, metrics = soft_seed_mixed_loss(**args, reference_log_prob=reference,
                                         reference_kl_weight=0.2)
    gap = reference.detach() - current
    expected = (gap.exp() - gap - 1).mean()
    assert torch.allclose(loss - base, 0.2 * expected)
    assert torch.allclose(metrics.reference_kl, expected.detach())
    gradient = torch.autograd.grad(loss - base, current)[0]
    assert torch.allclose(gradient, 0.2 * (1 - gap.exp()) / current.numel())
    assert reference.grad is None


def test_kl_requires_reference_and_valid_weight():
    import pytest
    with pytest.raises(ValueError, match='reference_log_prob is required'):
        soft_seed_mixed_loss(**_loss_inputs(), reference_kl_weight=0.1)
    for invalid in [-1.0, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='finite and nonnegative'):
            soft_seed_mixed_loss(**_loss_inputs(), reference_kl_weight=invalid)
