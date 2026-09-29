"""Frozen-component interventions on paired factual values; no validation training."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from .compressor_data import read,save,sha

ROOT=Path(__file__).resolve().parent.parent
RUN=ROOT/'locomo_pipeline/runs/persona128k_gpu0123_v1'


def norm(text):
    return re.sub(r'\s+',' ',text.casefold()).strip(' \n\t.\"\'`')


def append(path,row):
    with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()


def prepare(out):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-4B',local_files_only=True)
    chosen={}; used=set()
    for split,count in [('train',64),('validation',16)]:
        candidates=[]
        for r in read(RUN/'self_memory_retry_v1/prepared'/f'{split}.jsonl'):
            for i,m in enumerate(json.loads(r['memory_text'])['memories']):
                if not all(isinstance(m.get(k),str) and m[k].strip() for k in ('subject','attribute','value')):continue
                question=f'What is the value of "{m["attribute"]}" for "{m["subject"]}"? Return exactly the value, with no explanation.'
                if not 1<=len(tok.encode(m['value'],add_special_tokens=False))<=6 or norm(m['value']) in norm(question):continue
                candidates.append(dict(id=r['id']+':'+str(i),context_id=r['context_id'],question=question,
                    record={k:m[k] for k in ('subject','attribute','value')}))
        selected=[]
        contexts=sorted({r['context_id'] for r in candidates})
        per=count//len(contexts)
        for c in contexts:
            cc=sorted([r for r in candidates if r['context_id']==c],key=lambda r:sha(('component42|'+r['id']).encode()))
            group=[]
            for r in cc:
                value=norm(r['record']['value'])
                memory=json.dumps({'memories':[r['record']]},ensure_ascii=False,separators=(',',':'))
                if value in used or len(tok.encode(memory,add_special_tokens=False))>128:continue
                if any(norm(x['record']['value']) in norm(r['question']) or value in norm(x['question']) for x in selected+group):continue
                used.add(value);group.append(r)
                if len(group)==per:break
            if len(group)!=per:raise ValueError('insufficient distinct eligible values per context')
            selected.extend(group)
        if len(selected)!=count:raise ValueError('fact panel size')
        chosen[split]=selected
    data=[]
    for split,rows in chosen.items():
        for i,r in enumerate(rows):
            donor=rows[(i+1)%len(rows)]['record']['value']
            if norm(donor) in norm(r['question']):raise ValueError('counterfactual value leaked in question')
            for variant,value in [('original',r['record']['value']),('changed',donor)]:
                record=dict(r['record'],value=value)
                data.append(dict(id=r['id']+'/'+variant,source_id=r['id'],context_id=r['context_id'],split=split,
                    variant=variant,question=r['question'],answer=value,
                    memory=json.dumps({'memories':[record]},ensure_ascii=False,separators=(',',':')),
                    pair_id=r['id']+('/changed' if variant=='original' else '/original')))
    train_values={norm(r['answer']) for r in data if r['split']=='train'}
    if any(norm(r['answer']) in train_values for r in data if r['split']=='validation'):raise ValueError('held-out value overlap')
    save(out/'examples.json',data)
    source_files=[RUN/'self_memory_retry_v1/prepared'/f'{s}.jsonl' for s in ('train','validation')]
    final=json.loads((RUN/'reader_initialization_after_retry_v1/train/result.json').read_text())
    source_files += [Path(final['bridge']),Path(final['adapter'])/'adapter_model.safetensors',Path(__file__)]
    save(out/'protocol.json',dict(reader_initialization_adapter=final['adapter'],reader_initialization_bridge=final['bridge'],
        model=str(ROOT/'models/Qwen3-4B'),source_sha256={str(f):sha(f.read_bytes()) for f in source_files},
        examples_sha256=sha((out/'examples.json').read_bytes()),train_examples=128,validation_examples=32,
        train_contexts=sorted({r['context_id'] for r in data if r['split']=='train'}),
        validation_contexts=sorted({r['context_id'] for r in data if r['split']=='validation'}),
        data_definition='structural subject/attribute/value probes derived from self-memory; changed values are synthetic counterfactuals, not real facts',
        no_qa_gold_used=True,updates=200,effective_batch=4,learning_rates=dict(reader=1e-5,bridge=1e-4),
        objective='answer CE + .5 relu(.1 + own CE - same-question changed-value CE)',
        arms=['reader_only','bridge_only','joint'],selection='fixed steps, no validation-based stopping',
        diagnostic_readers=['base','memory_writer_initialization','auxiliary','reader_initialization'],
        caution='cross-reader swaps alone cannot distinguish information loss from interface incompatibility; frozen-component interventions test repairability',
        promotion_allowed=False))


def execute(out,mode):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from .compressor_core import load,target_nll
    from .train_compressor import prompt_and_target
    from .audit_reader_initialization import encode_states
    from .persona128k_recipe import assert_fp32_optimizer
    cfg=json.loads((out/'protocol.json').read_text());data=json.loads((out/'examples.json').read_text())
    for path,digest in cfg['source_sha256'].items():
        if sha(Path(path).read_bytes())!=digest:raise ValueError('original source changed')
    if sha((out/'examples.json').read_bytes())!=cfg['examples_sha256']:raise ValueError('examples changed')
    dest=out/mode;dest.mkdir()
    torch.set_num_threads(4);torch.manual_seed(42);random.seed(42);device=torch.device('cuda:0')
    tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    bridge,config,parent=load(ROOT,Path(cfg['reader_initialization_bridge']),base.config.hidden_size)
    bridge=bridge.to(device=device,dtype=torch.float32).eval().requires_grad_(False)
    lut={r['id']:r for r in data};train=[r for r in data if r['split']=='train'];val=[r for r in data if r['split']=='validation']
    with torch.no_grad():
        states={r['id']:encode_states(base,tok,[r['memory']],device,'persona128k').float() for r in data}
    encoded={r['id']:prompt_and_target(tok,r['question'],r['answer'],device,64) for r in data}
    def soft(r):
        x=states[r['id']]
        return bridge(x,torch.ones(x.shape[:2],device=device,dtype=torch.long)).to(torch.bfloat16)

    def audit(model,reader_name,selected):
        model.eval();bridge.eval();results=[]
        with torch.no_grad():
            for r in selected:
                pp,yy=encoded[r['id']]
                ss=soft(r)
                ids=torch.tensor([tok.encode(r['memory'],add_special_tokens=False)],device=device)
                raw=model.get_input_embeddings()(ids)
                if raw.shape[1]>256:raise ValueError('transparent prefix cannot fit K256')
                transparent=torch.cat([torch.zeros((1,256-raw.shape[1],raw.shape[2]),device=device,dtype=raw.dtype),raw],1)
                prefixes=dict(soft=ss,text=raw,embedding256=transparent,zero=torch.zeros_like(ss))
                for arm,prefix in prefixes.items():
                    emb=torch.cat([prefix,model.get_input_embeddings()(pp)],1)
                    gen=model.generate(inputs_embeds=emb,attention_mask=torch.ones(emb.shape[:2],device=device,dtype=torch.long),
                        max_new_tokens=32,do_sample=False,repetition_penalty=1.,pad_token_id=tok.eos_token_id,use_cache=True)[0]
                    answer=tok.decode(gen,skip_special_tokens=True)
                    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
                    row=dict(reader=reader_name,id=r['id'],split=r['split'],variant=r['variant'],arm=arm,
                        gold=r['answer'],answer=answer,correct=norm(answer)==norm(r['answer']),
                        finish_reason='stop' if len(gen) and int(gen[-1]) in eos else 'length',
                        nll=float(target_nll(model,pp,yy,prefix,32768)))
                    results.append(row);append(dest/'audit.jsonl',row)
                other=soft(lut[r['pair_id']])
                append(dest/'representation.jsonl',dict(reader=reader_name,id=r['id'],
                    mean_soft_cosine=float(F.cosine_similarity(ss.float().mean(1),other.float().mean(1))),
                    relative_pair_distance=float((ss.float()-other.float()).norm()/ss.float().norm().clamp_min(1e-8)),
                    soft_rms=float(ss.float().square().mean().sqrt()),embedding_rms=float(raw.float().square().mean().sqrt())))
        report={arm:dict(n=sum(r['arm']==arm for r in results),correct=sum(r['correct'] for r in results if r['arm']==arm),
            mean_nll=sum(r['nll'] for r in results if r['arm']==arm)/sum(r['arm']==arm for r in results)) for arm in prefixes}
        save(dest/(reader_name+'_scores.json'),report);print(json.dumps(dict(reader=reader_name,scores=report)),flush=True)
        return report

    if mode=='diagnose':
        adapters=dict(base=None,memory_writer_initialization=RUN/'memory_writer_initialization/epoch_3',
            auxiliary=RUN/'continuation/pretraining/reasoning/epoch_3/auxiliary_lora',reader_initialization=Path(cfg['reader_initialization_adapter']))
        reports={}
        for name,adapter in adapters.items():
            model=base if adapter is None else PeftModel.from_pretrained(base,adapter,is_trainable=False).eval().requires_grad_(False)
            reports[name]=audit(model,name,val)
            if adapter is not None:base=model.unload().eval().requires_grad_(False)
        save(dest/'result.json',dict(status='completed',scores=reports,training_launched=False));return

    policy=PeftModel.from_pretrained(base,cfg['reader_initialization_adapter'],is_trainable=True)
    policy.requires_grad_(False)
    reader_train=mode in ('reader_only','joint');bridge_train=mode in ('bridge_only','joint')
    if reader_train:
        for name,par in policy.named_parameters():
            if 'lora_' in name:par.requires_grad_(True);par.data=par.data.float()
    bridge.requires_grad_(bridge_train)
    # Keep dropout disabled in all arms so the intervention is the optimizer scope.
    policy.eval();bridge.eval();policy.config.use_cache=False
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    groups=[]
    if reader_train:groups.append(dict(params=[p for p in policy.parameters() if p.requires_grad],lr=1e-5))
    if bridge_train:groups.append(dict(params=[p for p in bridge.parameters() if p.requires_grad],lr=1e-4))
    opt=torch.optim.AdamW(groups,weight_decay=.01)
    frozen=[p for m in (policy,bridge) for p in m.parameters() if not p.requires_grad]
    versions=[p._version for p in frozen];active=[p for g in groups for p in g['params']]
    initial=[p.detach().clone() for p in active]
    rng=random.Random(42);order=[]
    for step in range(1,201):
        if len(order)<4:
            fresh=list(train);rng.shuffle(fresh);order.extend(fresh)
        batch=order[:4];order=order[4:];opt.zero_grad(set_to_none=True);totals=[]
        for r in batch:
            pp,yy=encoded[r['id']];own=target_nll(policy,pp,yy,soft(r),32768)
            wrong=target_nll(policy,pp,yy,soft(lut[r['pair_id']]),32768)
            rank=F.relu(.1+own-wrong);loss=own+.5*rank
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
            (loss/4).backward();totals.append((float(loss.detach()),float(own.detach()),float(wrong.detach())))
        grad=torch.nn.utils.clip_grad_norm_(active,1.)
        if not torch.isfinite(grad) or grad<=0:raise FloatingPointError('invalid gradient')
        opt.step();assert_fp32_optimizer(opt)
        if versions!=[p._version for p in frozen] or any(p.grad is not None for p in frozen):raise ValueError('frozen component changed')
        metric=dict(step=step,loss=sum(t[0] for t in totals)/4,own_nll=sum(t[1] for t in totals)/4,
            wrong_nll=sum(t[2] for t in totals)/4,gradient_norm=float(grad),example_ids=[r['id'] for r in batch])
        append(dest/'metrics.jsonl',metric)
        if step%20==0:print(json.dumps(metric),flush=True)
    changed=sum(not torch.equal(x,p.detach()) for x,p in zip(initial,active))
    if not changed:raise ValueError('no trained parameter changed')
    del initial
    torch.save(dict(bridge=bridge.state_dict(),config=config,provenance=dict(parent,
        procedure='component_alignment',component_intervention=mode,parent_bridge_sha256=sha(Path(cfg['reader_initialization_bridge']).read_bytes()),
        diagnostic_fact_training=True,formal_locomo_qa_training=False)),dest/'bridge.pt')
    policy.save_pretrained(dest/'adapter')
    torch.save(dict(optimizer=opt.state_dict(),step=200),dest/'optimizer.pt')
    policy.gradient_checkpointing_disable();policy.config.use_cache=True
    validation=audit(policy,'after',val)
    training=audit(policy,'train_probe',train[:16])
    # Native full-bank QA NLL checks protect against mistaking a synthetic probe gain for task improvement.
    native={}
    memory_rows=read(RUN/'self_memory_retry_v1/prepared/validation.jsonl')
    contexts={r['context_id'] for r in memory_rows}
    with torch.no_grad(),policy.disable_adapter():
        for c in contexts:
            native[c]=encode_states(base,tok,[r['memory_text'] for r in memory_rows if r['context_id']==c],device,'persona128k').float()
    qa=read(RUN/'eval30_api_v1/questions.jsonl');qa_values=[]
    with torch.no_grad():
        ns={c:bridge(x,torch.ones(x.shape[:2],device=device,dtype=torch.long)).to(torch.bfloat16) for c,x in native.items()}
        for q in qa:
            pp,yy=prompt_and_target(tok,'Answer the question concisely and completely.\nQuestion: '+q['question'],q['answer'],device,512)
            c=q['context_id'];donor=next(x for x in contexts if x!=c)
            rr=dict(question_id=q['question_id'],nll={a:float(target_nll(policy,pp,yy,x,32768)) for a,x in
                [('own',ns[c]),('shuffled',ns[donor]),('zero',torch.zeros_like(ns[c]))]})
            append(dest/'qa30_nll.jsonl',rr);qa_values.append(rr)
    summary={a:sum(r['nll'][a] for r in qa_values)/len(qa_values) for a in ('own','shuffled','zero')}
    for path,digest in cfg['source_sha256'].items():
        if sha(Path(path).read_bytes())!=digest:raise ValueError('original source changed')
    save(dest/'result.json',dict(status='completed',mode=mode,updates=200,validation=validation,train_probe=training,
        native_qa30_mean_nll=summary,changed_tensors=changed,frozen_components_unchanged=True,
        promotion_allowed=False,scope=dict(reader_trainable=reader_train,compressor_trainable=bridge_train),
        note='short synthetic structural probes are not LoCoMo QA accuracy; no automatic replacement of trained checkpoint'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',choices=['diagnose','reader_only','bridge_only','joint']);a=p.parse_args();out=a.output.resolve()
    if a.worker:execute(out,a.worker);return
    from .run_persona128k_alignment import gpu_free
    if out.exists():raise FileExistsError('refusing overwrite')
    if not gpu_free([0,1,2,3]):raise RuntimeError('GPU0-3 busy')
    out.mkdir(parents=True);prepare(out)
    def status(phase,**kw):save(out/'status.json',dict(phase=phase,updated=time.time(),**kw))
    def run(mode,gpu):
        with (out/(mode+'.log')).open('x') as log:
            code=subprocess.call([sys.executable,'-u','-m',__spec__.name,'--output',str(out),'--worker',mode],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'TOKENIZERS_PARALLELISM':'false'})
        if code:raise RuntimeError(mode+' failed; inspect log')
    try:
        status('cross_reader_diagnosis',status='running');run('diagnose',3)
        status('frozen_component_training',status='running')
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(run,mode,gpu) for gpu,mode in enumerate(('reader_only','bridge_only','joint'))]
            for job in jobs:job.result()
        reports={mode:json.loads((out/mode/'result.json').read_text()) for mode in ('reader_only','bridge_only','joint')}
        save(out/'comparison.json',dict(diagnosis=json.loads((out/'diagnose/result.json').read_text()),trials=reports,
            interpretation='frozen-component repairability experiment, not a proof of unique causal blame',promotion_allowed=False))
        status('completed',status='completed')
    except Exception as exc:status('failed',status='failed',error=repr(exc));raise


if __name__=='__main__':main()
