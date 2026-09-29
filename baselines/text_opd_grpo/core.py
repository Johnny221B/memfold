"""GRPO-aligned sampled-token SDPO primitives and atomic local role transport."""
import json,os,time,traceback
from pathlib import Path
import torch

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save(obj,tmp);tmp.replace(path)
def load(path):return torch.load(path,map_location='cpu',weights_only=False)
def wait(path,root,timeout=1800):
    path=Path(path);end=time.monotonic()+timeout
    while not path.exists():
        failures=list(Path(root).glob('failure-*.json'))
        if failures:raise RuntimeError(f'role failed; see {failures[0]}')
        if time.monotonic()>end:raise TimeoutError(str(path))
        time.sleep(.1)
    return load(path)
def fail(root,role):
    Path(root,f'failure-{role}.json').write_text(json.dumps(dict(role=role,error=traceback.format_exc())))
def grpo_advantages(rewards):
    x=torch.as_tensor(rewards,dtype=torch.float32).reshape(-1,8)
    return ((x-x.mean(1,keepdim=True))/(x.std(1,keepdim=True)+1e-4)).flatten()
def mixed_advantage(grpo,teacher,old,feedback):
    return (.9*grpo+.1*(teacher-old).detach()*float(feedback)).detach()
def loss_per_response(current,old,advantage,sampling_logp,epsilon=.2):
    ratio=(current-old.detach()).exp()
    correction=(old.detach()-sampling_logp).sum().exp()
    correction=torch.where(correction>3.,torch.zeros_like(correction),correction)
    return -torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage).mean()*correction

def feedback_prompts(rows):
    import re
    result=[]
    for i,r in enumerate(rows):
        good=[j for j,x in enumerate(rows) if x['qid']==r['qid'] and x['reward']==1 and j!=i]
        if good:
            solution=re.sub(r'<think>.*?</think>\s*','',rows[good[0]]['text'],flags=re.S)
            prompt=r['raw_prompt']+'\nCorrect solution:\n\n'+solution+'\n\nCorrectly solve the original question.'
        else:prompt=r['raw_prompt']
        result.append((prompt,bool(good)))
    return result

def local_slots(rank):
    return [i if i<16 else None for i in range(rank,18,3)]
def response_logp(model,prompt,response,temperature=1.2):
    device=next(model.parameters()).device
    ids=torch.tensor([prompt+response],device=device)
    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=len(response)+1).logits[:, :-1].float()/temperature
    assert logits.shape[1]==len(response)
    targets=torch.tensor(response,device=device)[None,:,None]
    return logits.log_softmax(-1).gather(-1,targets).squeeze(0).squeeze(-1)


def text_loss_parts(current_grpo, current_opd, old, teacher, advantage, sampling_logp, gate_beta=5.):
    from memory_opd.opd.seed_loss import seed_sampled_token_opd_loss
    grpo=loss_per_response(current_grpo,old,advantage,sampling_logp)
    opd,metrics=seed_sampled_token_opd_loss(current_opd[None],teacher[None],torch.ones_like(current_opd)[None],gate_beta=gate_beta)
    return grpo,opd,metrics

def response_logps(model,prompt,response):
    device=next(model.parameters()).device
    ids=torch.tensor([prompt+response],device=device)
    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=len(response)+1).logits[:,:-1].float()
    targets=torch.tensor(response,device=device)[None,:,None]
    return tuple((logits/t).log_softmax(-1).gather(-1,targets).flatten() for t in [1.2,1.])
