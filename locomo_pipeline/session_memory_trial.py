"""Complete self-memory sessions, frozen reader, paired K256/K512 experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from .component_alignment_trial import ROOT,RUN,norm,append
from .compressor_data import read,save,sha


def prepare(out):
    from transformers import AutoTokenizer
    source=RUN/'self_memory_retry_v1/prepared'
    cfg0=json.loads((RUN/'tokenwise_v1/protocol.json').read_text())
    tok=AutoTokenizer.from_pretrained(cfg0['model'],local_files_only=True)
    rows=read(source/'train.jsonl')+read(source/'validation.jsonl')
    sessions={};probes=[]
    for r in rows:
        ms=json.loads(r['memory_text'])['memories'];pairs=Counter((m['subject'],m['attribute']) for m in ms)
        attrs=Counter(m['attribute'] for m in ms);candidates=[]
        for i,m in enumerate(ms):
            subject,attribute,value=(m[k] for k in ['subject','attribute','value'])
            base=f'the memory record with subject "{subject}" and attribute "{attribute}"'
            if pairs[(subject,attribute)]==1:
                kind='relationship' if m.get('memory_type')=='relationship' or m.get('facet') in ['relationship','family'] else 'event' if m.get('memory_type')=='event' else 'fact'
                candidates.append(dict(kind=kind,field='value',question=f'What is the value field of {base}? Return exactly that field, with no explanation.',answer=value,index=i))
                date=m.get('temporal',{}).get('start')
                if date:candidates.append(dict(kind='time',field='temporal.start',question=f'What is the temporal.start field of {base}? Return exactly that field, with no explanation.',answer=date,index=i))
            if attrs[attribute]==1:
                candidates.append(dict(kind='entity',field='subject',question=f'What is the subject field of the memory record with attribute "{attribute}"? Return exactly that field, with no explanation.',answer=subject,index=i))
        candidates=[x for x in candidates if isinstance(x['answer'],str) and x['answer'].strip() and
            1<=len(tok.encode(x['answer'],add_special_tokens=False))<=24 and norm(x['answer']) not in norm(x['question'])]
        candidates.sort(key=lambda x:sha((r['id']+'|'+str(x['index'])+'|'+x['kind']).encode()))
        chosen=[]
        for kind in ['entity','event','time','relationship','fact']:
            cc=[x for x in candidates if x['kind']==kind]
            if cc:chosen.append(cc[0])
        if len(chosen)<2:raise ValueError('Too few unambiguous fields: '+r['id'])
        # Every original session remains present, with full byte-identical memory text.
        sessions[r['id']]=dict(r,tokens=len(tok.encode(r['memory_text'],add_special_tokens=False)))
        for j,x in enumerate(chosen):probes.append(dict(x,id=r['id']+f'/probe{j}',session=r['id'],context_id=r['context_id'],split=r['split']))
    train_contexts={r['context_id'] for r in rows if r['split']=='train'}
    val_contexts=sorted({r['context_id'] for r in rows if r['split']=='validation'})
    assert len(val_contexts)==2 and not train_contexts & set(val_contexts)
    panel={}
    # Deterministic length-stratified sessions, not selected on correctness.
    for split,cs in [('development',[val_contexts[0]]),('heldout',[val_contexts[1]]),('train_probe',sorted(train_contexts))]:
        selected=[]
        if split=='train_probe':
            selected=[sorted([r for r in rows if r['context_id']==c],key=lambda r:sha(r['id'].encode()))[0]['id'] for c in cs]
        else:
            ss=sorted([sessions[r['id']] for r in rows if r['context_id'] in cs],key=lambda r:r['tokens'])
            selected=[ss[round(i*(len(ss)-1)/5)]['id'] for i in range(6)]
        panel[split]=[p['id'] for p in probes if p['session'] in selected]
    bridge=RUN/'tokenwise_v1/unpooled/diverse/bridge.pt'
    files=[source/'train.jsonl',source/'validation.jsonl',bridge,Path(cfg0['adapter'])/'adapter_model.safetensors',Path(__file__),
           ROOT/'locomo_pipeline/compressor_core.py',ROOT/'locomo_pipeline/train_compressor.py']
    out.mkdir(parents=True)
    save(out/'data.json',dict(sessions=sessions,probes=probes,panels=panel))
    save(out/'protocol.json',dict(model=cfg0['model'],adapter=cfg0['adapter'],initial_mapper=str(bridge),
        sources={str(p):sha(p.read_bytes()) for p in files},data_hash=sha((out/'data.json').read_bytes()),
        train_contexts=sorted(train_contexts),development_context=val_contexts[0],heldout_context=val_contexts[1],
        input='complete original memory_text, byte-identical including all metadata; whole-session base encoding, no pooling/truncation/schema compaction',
        mapper_updates=2000,mapper_batch=8,mapper_tokens_per_session=128,mapper_lr=1e-4,
        mapper_objective='normalized token embedding MSE on sampled TRAIN positions; compression always uses ALL positions',
        compression_updates=400,compression_batch=4,compression_lr=1e-4,outputs=[256,512],
        compressor='same K-independent weights: pooled seed + .02 cross-attention residual, relative positional encodings',
        objective='field-answer CE + .2 relu(.5 + own CE - wrong-session CE)',
        frozen_compression=['base_encoder','reader','shared_adapted_mapper'],
        raw_control_gate='development raw_full exact match >=50%; otherwise no compressor training',
        selection='fixed updates; heldout is report-only, never checkpoint selection',
        metric='strict normalized field-readout EM, not LoCoMo QA accuracy or temporal reasoning',
        prior_exposure='validation conversations were used in earlier audits; not a fresh blind benchmark',
        question_blind=True,promotion_allowed=False))


def objects(cfg):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda:0').eval().requires_grad_(False)
    h=base.config.hidden_size
    mapper=torch.nn.Sequential(torch.nn.LayerNorm(h),torch.nn.Linear(h,1024),torch.nn.GELU(),torch.nn.Linear(1024,h)).to('cuda:0')
    return tok,base,mapper


def load_config(out):
    cfg=json.loads((out/'protocol.json').read_text())
    for p,hsh in cfg['sources'].items():assert sha(Path(p).read_bytes())==hsh
    assert sha((out/'data.json').read_bytes())==cfg['data_hash']
    return cfg,json.loads((out/'data.json').read_text())


def adapt(out):
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    cfg,d=load_config(out);torch.set_num_threads(4);torch.manual_seed(401)
    tok,base,mapper=objects(cfg)
    mapper.load_state_dict(torch.load(cfg['initial_mapper'],map_location='cpu',weights_only=True)['state_dict'],strict=True)
    cache=out/'cache';cache.mkdir();states={};ids_map={};manifest={}
    needed={p['session'] for p in d['probes'] if p['split']=='train' or p['id'] in sum(d['panels'].values(),[])}
    with torch.no_grad():
        for i,k in enumerate(sorted(needed)):
            r=d['sessions'][k];ids=torch.tensor([tok.encode(r['memory_text'],add_special_tokens=False)],device='cuda:0')
            assert ids.shape[1]==r['tokens']
            if ids.shape[1]>4096:raise ValueError('Unexpected input length; never truncate')
            x=base.model(input_ids=ids,use_cache=False).last_hidden_state.to(torch.float16)
            states[k]=x;ids_map[k]=ids
            name=sha(k.encode())+'.pt';torch.save(dict(session=k,states=x.cpu(),ids=ids.cpu()),cache/name);manifest[k]=name
            if (i+1)%40==0:print(json.dumps(dict(encoded=i+1,total=len(needed))),flush=True)
    save(cache/'manifest.json',manifest)
    opt=torch.optim.AdamW(mapper.parameters(),lr=cfg['mapper_lr'],weight_decay=.01)
    train=[k for k in states if d['sessions'][k]['split']=='train'];rng=random.Random(401)
    frozen=list(base.parameters());versions=[p._version for p in frozen]
    for step in range(1,cfg['mapper_updates']+1):
        opt.zero_grad(set_to_none=True);xx=[];yy=[]
        for k in rng.sample(train,cfg['mapper_batch']):
            pos=torch.randint(states[k].shape[1],(cfg['mapper_tokens_per_session'],),device='cuda:0')
            xx.append(states[k][:,pos].float())
            with torch.no_grad():yy.append(base.get_input_embeddings()(ids_map[k][:,pos]).float())
        x=torch.cat(xx,1);target=torch.cat(yy,1)
        loss=F.mse_loss(mapper(x),target)/target.square().mean().clamp_min(1e-8)
        if not torch.isfinite(loss):raise FloatingPointError('mapper loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(mapper.parameters(),1.)
        if not torch.isfinite(grad):raise FloatingPointError('mapper gradient')
        opt.step();assert versions==[p._version for p in frozen]
        assert not any(p.grad is not None for p in frozen)
        if step%100==0:
            metric=dict(step=step,normalized_mse=float(loss.detach()));append(out/'mapper_metrics.jsonl',metric);print(json.dumps(metric),flush=True)
    torch.save(dict(state_dict=mapper.state_dict(),hidden=base.config.hidden_size,promotion_allowed=False),out/'mapper.pt')
    torch.save(opt.state_dict(),out/'mapper_optimizer.pt')
    mapper.eval().requires_grad_(False)
    mapped={}
    with torch.no_grad():
        for k,x in states.items():mapped[k]=mapper(x.float())
    policy=PeftModel.from_pretrained(base,cfg['adapter'],is_trainable=False).eval().requires_grad_(False)
    scores=evaluate(out/'controls.jsonl',d,d['panels']['development'],tok,policy,
        lambda k:{'raw_full':policy.get_input_embeddings()(ids_map[k]),'mapped_full':mapped[k]},'controls')
    save(out/'controls.json',scores)


def evaluate(path,d,probe_ids,tok,policy,prefix_fn,phase):
    import torch
    from .train_compressor import prompt_and_target
    from .compressor_core import target_nll
    lut={p['id']:p for p in d['probes']};rows=[]
    with torch.no_grad():
        for pid in probe_ids:
            p=lut[pid];pp,yy=prompt_and_target(tok,p['question'],p['answer'],'cuda:0',128)
            answers={};nll={};finish={}
            for arm,x in prefix_fn(p['session']).items():
                nll[arm]=float(target_nll(policy,pp,yy,x,32768))
                if arm in ['wrong','zero']:
                    continue
                emb=torch.cat([x.to(torch.bfloat16),policy.get_input_embeddings()(pp)],1)
                gen=policy.generate(inputs_embeds=emb,attention_mask=torch.ones(emb.shape[:2],device='cuda:0',dtype=torch.long),
                    do_sample=False,max_new_tokens=64,pad_token_id=tok.eos_token_id,use_cache=True)[0]
                answers[arm]=tok.decode(gen,skip_special_tokens=True)
                eos=policy.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
                finish[arm]='stop' if len(gen) and int(gen[-1]) in eos else 'length'
            r=dict(phase=phase,id=pid,session=p['session'],context=p['context_id'],kind=p['kind'],gold=p['answer'],
                tokens=d['sessions'][p['session']]['tokens'],answers=answers,nll=nll,finish=finish,
                correct={a:norm(s)==norm(p['answer']) for a,s in answers.items()})
            append(path,r);rows.append(r)
    return dict(n=len(rows),correct={a:sum(r['correct'][a] for r in rows) for a in rows[0]['correct']},
        mean_nll={a:sum(r['nll'][a] for r in rows)/len(rows) for a in rows[0]['nll']},
        by_kind={kind:dict(n=sum(r['kind']==kind for r in rows),correct={a:sum(r['correct'][a] for r in rows if r['kind']==kind) for a in rows[0]['correct']}) for kind in sorted({r['kind'] for r in rows})})


def build_compressor(h):
    import torch
    import torch.nn.functional as F
    class Resampler(torch.nn.Module):
        def __init__(self):
            super().__init__();self.project=torch.nn.Linear(h,256);self.norm=torch.nn.LayerNorm(h)
            self.attn=torch.nn.MultiheadAttention(256,8,batch_first=True,dropout=0.)
            self.ff=torch.nn.Sequential(torch.nn.LayerNorm(256),torch.nn.Linear(256,512),torch.nn.GELU(),torch.nn.Linear(512,256))
            self.output=torch.nn.Linear(256,h);torch.nn.init.zeros_(self.output.weight);torch.nn.init.zeros_(self.output.bias)
        def pos(self,n,device):
            x=torch.linspace(0,1,n,device=device)[:,None]*torch.exp(torch.arange(128,device=device)/128*6)[None,:]
            return torch.cat([x.sin(),x.cos()],-1)[None]
        def forward(self,x,k):
            seed=F.adaptive_avg_pool1d(x.transpose(1,2),k).transpose(1,2)
            memory=self.project(self.norm(x))+.1*self.pos(x.shape[1],x.device)
            query=self.project(self.norm(seed))+.1*self.pos(k,x.device)
            z,_=self.attn(query,memory,memory,need_weights=False)
            z=z+self.ff(z)
            return seed+.02*self.output(z)
    return Resampler()


def worker(out,k):
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from .train_compressor import prompt_and_target
    from .compressor_core import target_nll
    cfg,d=load_config(out);torch.set_num_threads(4);torch.manual_seed(501)
    dest=out/f'k{k}';dest.mkdir()
    tok,base,mapper=objects(cfg)
    mapper.load_state_dict(torch.load(out/'mapper.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
    mapper.eval().requires_grad_(False);mapped={};raw={}
    with torch.no_grad():
        for sid,name in json.loads((out/'cache/manifest.json').read_text()).items():
            item=torch.load(out/'cache'/name,map_location='cpu',weights_only=True);assert item['session']==sid
            mapped[sid]=mapper(item['states'].to('cuda:0').float())
            raw[sid]=base.get_input_embeddings()(item['ids'].to('cuda:0'))
    policy=PeftModel.from_pretrained(base,cfg['adapter'],is_trainable=False).eval().requires_grad_(False)
    policy.config.use_cache=False;policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    # Reset immediately before creation: exactly identical initial weights across K.
    torch.manual_seed(501);compressor=build_compressor(base.config.hidden_size).to('cuda:0')
    initial=[p.detach().clone() for p in compressor.parameters()]
    frozen=list(policy.parameters())+list(mapper.parameters());versions=[p._version for p in frozen]
    opt=torch.optim.AdamW(compressor.parameters(),lr=cfg['compression_lr'],weight_decay=.01)
    train=[p for p in d['probes'] if p['split']=='train'];sessions=sorted({p['session'] for p in train});rng=random.Random(501)
    encoded={p['id']:prompt_and_target(tok,p['question'],p['answer'],'cuda:0',128) for p in train}
    def prefix(sid):
        x=compressor(mapped[sid],k);assert x.shape[1]==k;return x
    for step in range(1,cfg['compression_updates']+1):
        opt.zero_grad(set_to_none=True);parts=[]
        for p in rng.sample(train,cfg['compression_batch']):
            donors=[s for s in sessions if d['sessions'][s]['context_id']!=p['context_id']]
            donor=rng.choice(donors);pp,yy=encoded[p['id']]
            own=target_nll(policy,pp,yy,prefix(p['session']),32768)
            wrong=target_nll(policy,pp,yy,prefix(donor),32768)
            loss=own+.2*F.relu(.5+own-wrong)
            if not torch.isfinite(loss):raise FloatingPointError('compressor loss')
            (loss/cfg['compression_batch']).backward();parts.append((float(own.detach()),float(wrong.detach())))
        grad=torch.nn.utils.clip_grad_norm_(compressor.parameters(),1.)
        if not torch.isfinite(grad):raise FloatingPointError('compressor gradient')
        opt.step();assert versions==[p._version for p in frozen]
        assert not any(p.requires_grad or p.grad is not None for p in frozen)
        metric=dict(step=step,own_nll=sum(x[0] for x in parts)/len(parts),wrong_nll=sum(x[1] for x in parts)/len(parts),grad=float(grad))
        append(dest/'metrics.jsonl',metric)
        if step%40==0:print(json.dumps(metric),flush=True)
    changed=sum(not torch.equal(p,a) for p,a in zip(compressor.parameters(),initial))
    assert changed>0
    torch.save(dict(state_dict=compressor.state_dict(),hidden=base.config.hidden_size,output_tokens=k,
        mapper_hash=sha((out/'mapper.pt').read_bytes()),protocol_hash=sha((out/'protocol.json').read_bytes()),promotion_allowed=False),dest/'compressor.pt')
    torch.save(opt.state_dict(),dest/'optimizer.pt')
    reports={}
    for split,ids in d['panels'].items():
        local=sorted({next(p['session'] for p in d['probes'] if p['id']==pid) for pid in ids})
        def arms(sid):
            donor=local[(local.index(sid)+1)%len(local)]
            return dict(own=prefix(sid),wrong=prefix(donor),zero=torch.zeros_like(prefix(sid)),raw_full=raw[sid],mapped_full=mapped[sid])
        reports[split]=evaluate(dest/'audit.jsonl',d,ids,tok,policy,arms,split)
        print(json.dumps(dict(split=split,report=reports[split])),flush=True)
    save(dest/'result.json',dict(status='completed',k=k,updates=cfg['compression_updates'],reports=reports,
        changed_tensors=changed,frozen_unchanged=True,promotion_allowed=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',choices=['adapt','256','512']);a=p.parse_args();out=a.output.resolve()
    if a.worker:
        adapt(out) if a.worker=='adapt' else worker(out,int(a.worker));return
    if out.exists():raise FileExistsError(out)
    from .run_persona128k_alignment import gpu_free
    if not gpu_free([0,1]):raise RuntimeError('GPU0/1 busy')
    prepare(out)
    def launch(mode,gpu):
        with (out/(mode+'.log')).open('x') as log:
            rc=subprocess.call([sys.executable,'-u','-m',__spec__.name,'--output',str(out),'--worker',mode],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'TOKENIZERS_PARALLELISM':'false'})
        if rc:raise RuntimeError(mode+' failed')
    save(out/'status.json',dict(status='running',phase='mapper',updated=time.time()))
    try:
        launch('adapt',0)
        control=json.loads((out/'controls.json').read_text())
        if control['correct']['raw_full']/control['n']<.5:
            save(out/'status.json',dict(status='completed',stop_reason='raw_text_control_failed',updated=time.time()));return
        save(out/'status.json',dict(status='running',phase='compression',updated=time.time()))
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(launch,mode,gpu) for gpu,mode in enumerate(['256','512'])]
            for job in jobs:job.result()
        save(out/'comparison.json',{str(k):json.loads((out/f'k{k}'/'result.json').read_text()) for k in [256,512]})
        save(out/'status.json',dict(status='completed',updated=time.time()))
    except Exception as exc:
        save(out/'status.json',dict(status='failed',error=repr(exc),updated=time.time()));raise


if __name__=='__main__':main()
