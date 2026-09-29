"""reader initialization: each entire session memory independently maps to512 soft tokens."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from .component_alignment_trial import ROOT, append
from .compressor_data import save, sha, memory_bank


def prepare(out, source=None):
    from .train_shallow_reader_initialization_512 import prepare as original_prepare
    original_prepare(out,source)
    cfg=json.loads((out/'protocol.json').read_text())
    cfg.update(epochs=1,checkpoints=[1],output_tokens_per_session=512,output_tokens=None,
        memory_unit='one complete session memory JSON, NOT individual memory records and NOT whole conversation bank',
        reader_prefix='concatenate512-token outputs from every session in source order; total512 times session count',
        selection='fixed one full epoch pilot, initial versus epoch1; no test-based selection',
        negative='different-conversation session memories cycled to match the own session count; primary own input retains every session',
        checkpoints_from_stopped_global512_run_used=False,
        initialization_note='restart from original shallow512 single-session compressor and prior completed epoch5 reader, NOT the stopped whole-bank512 training',
        evaluation='fixed exploratory LongMemEval-S50 before/after with same per-session512 interface; no LongMemEval training')
    cfg['sources'][str(Path(__file__))]=sha(Path(__file__).read_bytes())
    save(out/'protocol.json',cfg)


def answer_nll(model,prompt,target,soft,maximum_sequence=32768):
    """Only project answer positions to vocabulary; retain all prefix attention."""
    import torch
    import torch.nn.functional as F
    if soft.shape[1]+prompt.shape[1]+target.shape[1]>maximum_sequence: raise ValueError('reader context overflow; no truncation')
    ids=torch.cat([prompt,target],1); emb=model.get_input_embeddings()(ids)
    emb=torch.cat([soft.to(emb.dtype),emb],1)
    logits=model(inputs_embeds=emb,attention_mask=torch.ones(emb.shape[:2],device=emb.device,dtype=torch.long),
                 use_cache=False,logits_to_keep=target.shape[1]+1).logits[:,:-1]
    assert logits.shape[:2]==target.shape
    return F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),target.reshape(-1))


def session_prefix(compressor,states):
    import torch
    parts=[compressor(x,512) for x in states]
    assert all(x.shape[1]==512 for x in parts)
    result=torch.cat(parts,1)
    assert result.shape[1]==512*len(states)
    return result


def worker(out,max_updates=0):
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from .session_memory_trial import build_compressor
    from .train_compressor import prompt_and_target
    from .persona128k_recipe import assert_fp32_optimizer
    cfg=json.loads((out/'protocol.json').read_text());data=json.loads((out/'data.json').read_text())
    for p,digest in cfg['sources'].items(): assert sha(Path(p).read_bytes())==digest
    assert sha((out/'data.json').read_bytes())==cfg['data_hash']
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==4
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    dist.init_process_group('nccl',device_id=device);dest=out/('smoke' if max_updates else 'train')
    try:
        if rank==0:dest.mkdir()
        dist.barrier()
        tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
        base=AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.bfloat16,
               attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        h=base.config.hidden_size
        mapper=torch.nn.Sequential(torch.nn.LayerNorm(h),torch.nn.Linear(h,1024),torch.nn.GELU(),torch.nn.Linear(1024,h)).to(device)
        m=torch.load(cfg['mapper'],map_location='cpu',weights_only=True);assert m['encoder_layer']==cfg.get('encoder_layer_key','first')
        mapper.load_state_dict(m['state_dict'],strict=True);mapper.eval().requires_grad_(False)
        manifest=json.loads((Path(cfg['cache'])/'manifest.json').read_text());bank=memory_bank(data['memories'])
        states={};hashes={}
        with torch.no_grad():
            for c,rows in bank.items():
                states[c]=[]
                for r in rows:
                    path=Path(cfg['cache'])/manifest[r['id']];item=torch.load(path,map_location='cpu',weights_only=True)
                    assert item['session']==r['id'] and item['encoder_layer']==cfg.get('encoder_layer_key','first')
                    assert item['states'].shape[1]==data['sessions'][r['id']]['tokens']
                    assert tok.encode(r['memory_text'],add_special_tokens=False)==item['ids'][0].tolist()
                    states[c].append(mapper(item['states'].to(device).float()));hashes[r['id']]=sha(path.read_bytes())
                assert len(states[c])==len(rows)
        policy=PeftModel.from_pretrained(base,cfg['initial_reader'],is_trainable=True)
        for name,p in policy.named_parameters():
            p.requires_grad_('lora_' in name)
            if p.requires_grad:p.data=p.data.float()
        policy.config.use_cache=False;policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        compressor=build_compressor(h).to(device)
        ck=torch.load(cfg['initial_compressor'],map_location='cpu',weights_only=True)
        assert ck['output_tokens']==512 and ck['mapper_hash']==sha(Path(cfg['mapper']).read_bytes())
        compressor.load_state_dict(ck['state_dict'],strict=True)
        class Objective(torch.nn.Module):
            def __init__(self):super().__init__();self.reader=policy;self.compressor=compressor
            def forward(self,own,wrong,pp,yy):
                own_soft=session_prefix(self.compressor,own)
                wrong_soft=session_prefix(self.compressor,[wrong[i%len(wrong)] for i in range(len(own))])
                assert own_soft.shape==wrong_soft.shape and own_soft.shape[1]==512*len(own)
                ce=answer_nll(self.reader,pp,yy,own_soft);negative=answer_nll(self.reader,pp,yy,wrong_soft)
                margin=F.relu(.5+ce-negative)
                return ce+.2*margin,ce.detach(),negative.detach(),margin.detach()
        objective=Objective();wrapped=DDP(objective,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
        rp=[p for p in policy.parameters() if p.requires_grad];cp=list(compressor.parameters());active=rp+cp
        before=[p.detach().clone() for p in active];frozen=[p for p in policy.parameters() if not p.requires_grad]+list(mapper.parameters())
        versions=[p._version for p in frozen]
        opt=torch.optim.AdamW([dict(params=rp,lr=cfg['reader_lr']),dict(params=cp,lr=cfg['compressor_lr'])],weight_decay=.01)
        encoded={q['question_id']:prompt_and_target(tok,'Answer the question concisely and completely.\nQuestion: '+q['question'],q['answer'],device,512) for q in data['questions']}
        for q in data['questions']:
            pp,yy=encoded[q['question_id']];assert 512*len(states[q['context_id']])+pp.shape[1]+yy.shape[1]<=32768
        if rank==0:save(dest/'cache_audit.json',dict(session_hashes=hashes,contexts={c:dict(sessions=len(v),input_tokens=sum(x.shape[1] for x in v),soft_tokens=512*len(v)) for c,v in states.items()},granularity='one complete session JSON per512',no_truncation=True))
        order=list(data['questions']);random.Random(cfg['seed']+1).shuffle(order);objective.train()
        totals=torch.zeros(4,device=device,dtype=torch.float64);step=0;start=time.time()
        for offset in range(0,len(order),world):
            batch=order[offset:offset+world];present=rank<len(batch);q=batch[rank] if present else order[0]
            donor=random.Random(f"{cfg['seed']}|1|{q['question_id']}").choice([c for c in sorted(states) if c!=q['context_id']])
            pp,yy=encoded[q['question_id']];opt.zero_grad(set_to_none=True)
            loss,ce,wrong,margin=wrapped(states[q['context_id']],states[donor],pp,yy)
            (loss*(world/len(batch) if present else 0.)).backward();grad=torch.nn.utils.clip_grad_norm_(active,1.)
            ok=torch.tensor(int(bool(torch.isfinite(loss)) and bool(torch.isfinite(grad))),device=device);dist.all_reduce(ok,op=dist.ReduceOp.MIN)
            if not ok.item():raise FloatingPointError('nonfinite training')
            opt.step();step+=1;assert_fp32_optimizer(opt)
            assert versions==[p._version for p in frozen] and not any(p.grad is not None for p in frozen)
            stat=torch.tensor([float(ce),float(wrong),float(margin),1.],device=device,dtype=torch.float64) if present else torch.zeros(4,device=device,dtype=torch.float64)
            totals+=stat;dist.all_reduce(stat)
            if rank==0:
                metric=dict(step=step,total_steps=math.ceil(len(order)/world),own_nll=float(stat[0]/stat[3]),wrong_nll=float(stat[1]/stat[3]),margin=float(stat[2]/stat[3]),grad=float(grad),examples=int(stat[3]),elapsed=time.time()-start)
                append(dest/'metrics.jsonl',metric)
                if step%5==0 or max_updates:print(json.dumps(metric),flush=True)
            if max_updates and step>=max_updates:break
        dist.all_reduce(totals)
        if not max_updates:assert int(totals[3])==len(order)
        rc=sum(not torch.equal(a,p) for a,p in zip(before[:len(rp)],rp));cc=sum(not torch.equal(a,p) for a,p in zip(before[len(rp):],cp))
        assert rc>0 and cc>0
        if rank==0:
            ep=dest/'epoch_1';ep.mkdir();policy.save_pretrained(ep/'adapter')
            torch.save(dict(state_dict=compressor.state_dict(),hidden=h,output_tokens_per_session=512,granularity='session_json',procedure='reader_initialization',smoke_only=bool(max_updates),
                mapper_hash=sha(Path(cfg['mapper']).read_bytes()),protocol_hash=sha((out/'protocol.json').read_bytes()),promotion_allowed=False),ep/'compressor.pt')
            torch.save(dict(optimizer=opt.state_dict(),step=step,epoch=1),ep/'optimizer.pt')
            save(dest/'result.json',dict(status='smoke_completed' if max_updates else 'completed',updates=step,examples=int(totals[3]),epochs=1,own_nll=float(totals[0]/totals[3]),wrong_nll=float(totals[1]/totals[3]),reader_changed_tensors=rc,compressor_changed_tensors=cc,frozen_unchanged=True,checkpoint=str(ep),on_policy_optimization_launched=False))
    finally:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--worker',action='store_true');p.add_argument('--max-updates',type=int,default=0)
    a=p.parse_args();out=a.output.resolve()
    if a.worker:worker(out,a.max_updates);return
    from .run_persona128k_alignment import gpu_free
    if out.exists():raise FileExistsError(out)
    visible=os.environ.get('PIPELINE_CUDA_VISIBLE_DEVICES','0,1,2,3')
    physical=[int(x) for x in visible.split(',')]
    if len(physical)!=4 or not gpu_free(physical):raise RuntimeError(f'GPU set busy or invalid: {visible}')
    prepare(out,a.source.resolve() if a.source else None)
    try:
        for name,extra in [('smoke',['--max-updates','2']),('train',[])]:
            save(out/'status.json',dict(status='running',phase=name,updated=time.time()))
            with (out/(name+'.log')).open('x') as log:
                rc=subprocess.call([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m',__spec__.name,'--output',str(out),'--worker',*extra],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':visible,'TOKENIZERS_PARALLELISM':'false'})
            if rc:raise RuntimeError(name+' failed')
        save(out/'status.json',dict(status='completed',evaluation_pending=True,on_policy_optimization_launched=False,updated=time.time()))
    except Exception as exc:save(out/'status.json',dict(status='failed',error=repr(exc),updated=time.time()));raise


if __name__=='__main__':main()
