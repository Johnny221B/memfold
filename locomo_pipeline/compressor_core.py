"""Legacy-compatible bridge weights and explicit training boundaries."""
from pathlib import Path
import sys
import torch
import torch.nn.functional as F


def legacy(root):
    sys.path.insert(0,str(Path(root).resolve()/'src'))
    from memory_opd.soft_reconstruction import ContextResampler, ContextToSoftTokens, SoftTokenProjector
    return ContextResampler, ContextToSoftTokens, SoftTokenProjector


def build(root, config):
    Resampler, Bridge, Projector=legacy(root)
    return Bridge(Resampler(config['context_dim'],latent_dim=config['latent_dim'],
        token_count=config['token_count'],layers=config['layers'],heads=config['heads'],
        context_residual=config.get('context_residual',False)),
        Projector(config['latent_dim'],config['lm_dim']))


def load(root, path, hidden):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    config=saved['config']
    if config['context_dim']!=hidden or config['lm_dim']!=hidden:
        raise ValueError('bridge encoder/projector dimensions do not match backbone')
    bridge=build(root,config)
    bridge.load_state_dict(saved['bridge'],strict=True)
    return bridge,config,saved.get('provenance',{})


def configure_scope(model, bridge, procedure, joint=False):
    model.requires_grad_(False)
    bridge.requires_grad_(True)
    if procedure=='auxiliary_reasoning_adaptation' or (procedure=='reader_initialization' and joint):
        for name,p in model.named_parameters():
            if 'lora_' in name:
                p.requires_grad_(True)
        if not any(p.requires_grad for p in model.parameters()):
            raise ValueError('trainable LoRA required for reasoning/joint reader_initialization')
    if procedure=='reader_initialization' and joint:
        bridge.requires_grad_(False)
        bridge.resampler.decoder.layers[-1].requires_grad_(True)
        bridge.resampler.output_norm.requires_grad_(True)
        bridge.projector.requires_grad_(True)


def audit_scope(model, bridge, optimizer, procedure, joint=False):
    expected_lora=procedure=='auxiliary_reasoning_adaptation' or (procedure=='reader_initialization' and joint)
    if any(p.requires_grad for n,p in model.named_parameters() if 'lora_' not in n):
        raise ValueError('base backbone must remain frozen')
    if any(p.requires_grad for n,p in model.named_parameters() if 'lora_' in n)!=expected_lora:
        raise ValueError('LoRA scope mismatch')
    active=[p for m in (model,bridge) for p in m.parameters() if p.requires_grad]
    actual=[p for group in optimizer.param_groups for p in group['params']]
    if len(actual)!=len({id(p) for p in actual}) or {id(p) for p in actual}!={id(p) for p in active}:
        raise ValueError('optimizer does not exactly cover trainable parameters')
    return {'backbone_lora_trainable':expected_lora,
            'compressor_trainable_parameters':sum(p.numel() for p in bridge.parameters() if p.requires_grad)}


def soft_anchor_loss(soft, initial_soft):
    """Historical 32K relative output-space MSE, NOT parameter-space drift."""
    if soft.shape != initial_soft.shape:
        raise ValueError('anchor soft-token shape mismatch')
    reference = initial_soft.detach().float()
    return F.mse_loss(soft.float(), reference) / reference.square().mean().clamp_min(1e-8)


def vectors(soft):
    return F.normalize(soft.float().mean(1),dim=-1)


def separation(own, other, maximum_cosine=.8):
    return F.relu((2-2*maximum_cosine)**.5-torch.linalg.vector_norm(vectors(own)-vectors(other),dim=-1)).mean()


def warmup_loss(soft, labels, off_diagonal_gram=False):
    v=vectors(soft)
    cos=v@v.T
    labels=torch.as_tensor(labels,device=v.device)
    same=labels[:,None]==labels[None,:]
    diagonal=torch.eye(len(labels),device=v.device,dtype=torch.bool)
    if not (~same).any() or not (same & ~diagonal).any():
        raise ValueError('warmup needs two contexts and two views per context')
    distances=torch.linalg.vector_norm(v[:,None]-v[None,:],dim=-1)
    negative=F.relu((.4)**.5-distances[~same]).mean()
    alignment=(1-cos[same & ~diagonal]).mean()
    prototypes=F.normalize(torch.stack([v[labels==c].mean(0) for c in labels.unique()]),dim=-1)
    gram=prototypes@prototypes.T
    gram_regularization=(gram-torch.eye(len(prototypes),device=v.device)).square().mean()
    if off_diagonal_gram:
        gram_regularization=gram[~torch.eye(len(prototypes),device=v.device,dtype=torch.bool)].square().mean()
    return negative+.1*alignment+.1*gram_regularization


def target_nll(model, prompt, target, soft, maximum_sequence):
    """Frozen decoder forward MUST retain gradients into soft tokens."""
    if soft.shape[1]+prompt.shape[1]+target.shape[1]>maximum_sequence:
        raise ValueError('decoder context overflow; no silent truncation')
    ids=torch.cat([prompt,target],1)
    token_embeds=model.get_input_embeddings()(ids)
    embeds=torch.cat([soft.to(token_embeds.dtype),token_embeds],1)
    logits=model(inputs_embeds=embeds,attention_mask=torch.ones(embeds.shape[:2],device=embeds.device),
        use_cache=False).logits[:,soft.shape[1]+prompt.shape[1]-1:-1]
    parts=[F.cross_entropy(x.float().reshape(-1,x.shape[-1]),y.reshape(-1),reduction='sum')
           for x,y in zip(logits.split(64,1),target.split(64,1))]
    return sum(parts)/target.numel()
