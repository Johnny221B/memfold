"""Enforce the user-confirmed trainable scopes, independent of dataset and trainer."""
SCOPES={
    'memory_writer_initialization':frozenset({'backbone_lora'}),
    'compressor_reconstruction':frozenset({'compressor'}),
    'on_policy_optimization':frozenset({'backbone_lora'}),
}

def check_trainable_scope(procedure,components,optimizer=None):
    """components maps role to (name, parameter) pairs; include ALL model parameters.

    Frozen backbone must still participate in autograd during compressor reconstruction so loss can
    backpropagate through it into soft tokens. Do not use no_grad for that forward.
    This guard is usable with torch parameters without importing torch here.
    """
    allowed=SCOPES[procedure]
    valid_roles={'backbone_base','backbone_lora','compressor','teacher','reference'}
    if set(components)-valid_roles: raise ValueError('unknown component role')
    active={}; seen=set()
    for role,pairs in components.items():
        for name,param in pairs:
            if id(param) in seen: raise ValueError('parameter registered twice')
            seen.add(id(param))
            if param.requires_grad:
                if role not in allowed: raise ValueError(f'{procedure}: unexpected trainable {role}.{name}')
                active[id(param)]=role
    if set(active.values())!=set(allowed): raise ValueError(f'{procedure}: missing expected trainable component')
    if optimizer is not None:
        ids=[id(p) for g in optimizer.param_groups for p in g['params']]
        if len(ids)!=len(set(ids)) or set(ids)!=set(active):
            raise ValueError('optimizer parameters do not exactly match trainable parameters')
    return {'procedure':procedure,'trainable_roles':sorted(allowed),'trainable_parameter_tensors':len(active)}
