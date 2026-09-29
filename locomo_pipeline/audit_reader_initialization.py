"""Fixed paired QA-NLL before/after joint reader initialization; no test-set model selection."""
import argparse
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer,AutoModelForCausalLM
from .compressor_core import load,target_nll
from .compressor_data import read,save,sha,memory_bank
from .train_compressor import state_bank,prompt_and_target
from .persona128k_recipe import pool_hidden

ROOT=Path(__file__).resolve().parent.parent


def encode_states(model,tok,segments,device,recipe='legacy'):
    parts=[]
    for text in segments:
        ids=tok.encode(text,add_special_tokens=False)
        for start in range(0,len(ids),2048):
            tokens=torch.tensor([ids[start:start+2048]],device=device)
            h=model.model(input_ids=tokens,use_cache=False).last_hidden_state
            parts.append(pool_hidden(h,32,recipe).to(torch.float16 if recipe=='persona128k' else torch.bfloat16))
    return torch.cat(parts,1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','memories','cache','output']: p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('refusing overwrite')
    final=json.loads((a.run/'train/result.json').read_text())
    if final['status']!='completed' or final['provenance']['smoke_only']: raise ValueError('completed formal reader initialization required')
    cfg=json.loads((a.run/'train/run_config.json').read_text())['args']
    recipe=cfg.get('recipe','legacy')
    bridge_dtype=torch.float32 if recipe=='persona128k' else torch.bfloat16
    model_path=Path(cfg['model']); prepared=Path(cfg['prepared'])
    torch.set_num_threads(4); torch.manual_seed(42); device=torch.device('cuda:0')
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    selected={s:sorted(read(prepared/s/'qa.jsonl'),key=lambda q:sha(('42|'+q['question_id']).encode()))[:50] for s in ['train','validation']}
    versions_to_test=[('before',Path(cfg['writer_adapter']),Path(cfg['bridge'])),
                      ('after',Path(final['adapter']),Path(final['bridge']))]
    a.output.mkdir(parents=True)
    save(a.output/'protocol.json',dict(selected={s:[q['question_id'] for q in qs] for s,qs in selected.items()},
        checkpoints={name:dict(adapter_sha256=sha((adapter/'adapter_model.safetensors').read_bytes()),bridge_sha256=sha(bridge.read_bytes())) for name,adapter,bridge in versions_to_test},
        memory_hashes={s:sha((a.memories/(s+'.jsonl')).read_bytes()) for s in selected},
        target='gold QA answer for both before and after; not reasoning',metric='teacher-forced NLL, not accuracy',
        no_gold_retrieval=True,donor='next conversation in sorted same split, cyclic',training_launched=False))
    base=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    banks={}
    for s in selected:
        banks[s]={}
        for r in read(a.memories/(s+'.jsonl')):
            if r['split']!=s: raise ValueError('split mismatch')
            banks[s].setdefault(r['context_id'],[]).append(r)
    if set(banks['train'])&set(banks['validation']): raise ValueError('validation leakage')
    settings=SimpleNamespace(model=model_path,memories=a.memories/'train.jsonl',cache=a.cache,chunk_tokens=2048,pool_tokens=32,recipe=recipe)
    states={'train':state_bank(settings,banks['train'],base.config.hidden_size,device,bridge_dtype)[0],'validation':{}}
    with torch.no_grad():
        for c,rows in banks['validation'].items():
            states['validation'][c]=encode_states(base,tok,[r['memory_text'] for r in rows],device,recipe).to(bridge_dtype)
    reports={}
    for name,adapter,bridge_path in versions_to_test:
        bridge,_,provenance=load(ROOT,bridge_path,base.config.hidden_size)
        if provenance.get('recipe','legacy')!=recipe:
            raise ValueError('evaluation recipe/checkpoint mismatch')
        bridge=bridge.to(device=device,dtype=bridge_dtype).eval().requires_grad_(False)
        policy=PeftModel.from_pretrained(base,adapter,is_trainable=False).eval().requires_grad_(False)
        reports[name]={}
        with torch.no_grad():
            for s,qs in selected.items():
                soft={c:bridge(x,torch.ones(x.shape[:2],device=device,dtype=torch.long)) for c,x in states[s].items()}
                contexts=sorted(soft); values=[]
                for q in qs:
                    c=q['context_id']
                    if set(q['source_session_ids'])!={r['id'] for r in banks[s][c]}: raise ValueError('memory coverage')
                    donor=contexts[(contexts.index(c)+1)%len(contexts)]
                    prompt,target=prompt_and_target(tok,'Answer the question concisely and completely.\nQuestion: '+q['question'],q['answer'],device,512)
                    nll={arm:float(target_nll(policy,prompt,target,x,32768)) for arm,x in
                         [('own',soft[c]),('shuffled',soft[donor]),('zero',torch.zeros_like(soft[c]))]}
                    if any(not torch.isfinite(torch.tensor(v)) for v in nll.values()): raise ValueError('nonfinite NLL')
                    row=dict(checkpoint=name,split=s,question_id=q['question_id'],donor=donor,nll=nll)
                    values.append(row)
                    with (a.output/'rows.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                reports[name][s]=dict(n=len(values),mean_nll={arm:mean(r['nll'][arm] for r in values) for arm in ['own','shuffled','zero']},
                    own_beats_shuffled=mean(r['nll']['own']<r['nll']['shuffled'] for r in values),
                    own_beats_zero=mean(r['nll']['own']<r['nll']['zero'] for r in values))
                print(json.dumps(dict(checkpoint=name,split=s,scores=reports[name][s])),flush=True)
        base=policy.unload().eval().requires_grad_(False)
        del policy,bridge
    save(a.output/'report.json',dict(status='completed',scores=reports,metric='QA NLL, not accuracy',training_launched=False))


if __name__=='__main__': main()
