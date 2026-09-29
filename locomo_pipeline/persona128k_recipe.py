"""Explicit Persona128K-aligned mechanics; LoCoMo data/task stay unchanged."""
import torch
import torch.nn.functional as F

RECIPES = ('legacy', 'persona128k')


def lora_options(recipe='legacy'):
    if recipe not in RECIPES:
        raise ValueError('unknown recipe')
    return (dict(r=16, lora_alpha=32, lora_dropout=.05,
                 target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
            if recipe == 'persona128k' else
            dict(r=8, lora_alpha=16, lora_dropout=0, target_modules=['q_proj','v_proj']))


def check_adapter(config, recipe):
    if recipe == 'legacy':
        return
    expected = lora_options(recipe)
    if any(config.get(k) != v for k,v in expected.items() if k != 'target_modules') or set(config.get('target_modules', [])) != set(expected['target_modules']):
        raise ValueError('persona128k requires freshly trained memory-writer initialization r16 attention+MLP adapter; old r8 is incompatible')
    if config.get('rank_pattern') or config.get('alpha_pattern'):
        raise ValueError('per-module LoRA overrides do not match persona128k')


def pool_hidden(hidden, pool_tokens=32, recipe='legacy'):
    if hidden.ndim != 3 or hidden.shape[1] == 0 or pool_tokens < 1:
        raise ValueError('nonempty [batch, tokens, hidden] and positive window required')
    if recipe == 'persona128k':
        # Match original reduction dtype, including the final partial window.
        return torch.stack([x.mean(1) for x in hidden.split(pool_tokens, dim=1)], dim=1)
    if recipe != 'legacy':
        raise ValueError('unknown recipe')
    return F.adaptive_avg_pool1d(hidden.float().transpose(1,2), min(pool_tokens,hidden.shape[1])).transpose(1,2)


def paired_example(order, index, same_context):
    own = order[index]
    for offset in range(1, len(order)):
        candidate = order[(index + offset) % len(order)]
        if (candidate['context_id'] == own['context_id']) == same_context:
            return candidate
    raise ValueError('requires a distinct same-context view' if same_context else 'requires another context')


def assert_fp32_optimizer(optimizer):
    if any(p.dtype != torch.float32 for g in optimizer.param_groups for p in g['params']):
        raise ValueError('FP32 master parameter contract')
    for state in optimizer.state.values():
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in state and (state[key].dtype != torch.float32 or not torch.isfinite(state[key]).all()):
                raise ValueError('FP32 finite optimizer moment contract')
