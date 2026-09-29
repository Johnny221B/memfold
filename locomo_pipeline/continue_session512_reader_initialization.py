"""Resume the verified per-session512 reader initialization through epochs2 and3."""
import argparse,json,math,os,random,subprocess,sys,time
from pathlib import Path
from .component_alignment_trial import ROOT,append
from .compressor_data import save,sha,memory_bank
from .train_session512_reader_initialization import answer_nll,session_prefix


def worker(run):
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from .session_memory_trial import build_compressor
    from .train_compressor import prompt_and_target
    from .persona128k_recipe import assert_fp32_optimizer
    cfg=json.loads((run/'protocol.json').read_text());data=json.loads((run/'data.json').read_text())
    continuation=json.loads((run/'continuation_protocol.json').read_text())
    for path,digest in continuation['sources'].items():assert sha(Path(path).read_bytes())==digest
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==4
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    dist.init_process_group('nccl',device_id=device)
    try:
        tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
        base=AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.bfloat16,
            attn_implementation='sdpa').to(device).eval().requires_grad_(False);h=base.config.hidden_size
        mapper=torch.nn.Sequential(torch.nn.LayerNorm(h),torch.nn.Linear(h,1024),torch.nn.GELU(),torch.nn.Linear(1024,h)).to(device)
        payload=torch.load(cfg['mapper'],map_location='cpu',weights_only=True);assert payload['encoder_layer']==cfg.get('encoder_layer_key','first')
        mapper.load_state_dict(payload['state_dict'],strict=True);mapper.eval().requires_grad_(False)
        manifest=json.loads((Path(cfg['cache'])/'manifest.json').read_text());bank=memory_bank(data['memories']);states={};hashes={}
        with torch.no_grad():
            for context,rows in bank.items():
                states[context]=[]
                for row in rows:
                    path=Path(cfg['cache'])/manifest[row['id']];item=torch.load(path,map_location='cpu',weights_only=True)
                    assert item['session']==row['id'] and item['encoder_layer']==cfg.get('encoder_layer_key','first')
                    assert item['states'].shape[1]==data['sessions'][row['id']]['tokens']
                    assert tok.encode(row['memory_text'],add_special_tokens=False)==item['ids'][0].tolist()
                    states[context].append(mapper(item['states'].to(device).float()));hashes[row['id']]=sha(path.read_bytes())
        resume=run/'train/epoch_1'
        policy=PeftModel.from_pretrained(base,resume/'adapter',is_trainable=True)
        for name,p in policy.named_parameters():
            p.requires_grad_('lora_' in name)
            if p.requires_grad:p.data=p.data.float()
        policy.config.use_cache=False;policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        compressor=build_compressor(h).to(device);payload=torch.load(resume/'compressor.pt',map_location='cpu',weights_only=True)
        assert payload['granularity']=='session_json' and payload['output_tokens_per_session']==512 and not payload['smoke_only']
        compressor.load_state_dict(payload['state_dict'],strict=True)
        class Objective(torch.nn.Module):
            def __init__(self):super().__init__();self.reader=policy;self.compressor=compressor
            def forward(self,own,wrong,prompt,target):
                own_soft=session_prefix(self.compressor,own)
                wrong_soft=session_prefix(self.compressor,[wrong[i%len(wrong)] for i in range(len(own))])
                ce=answer_nll(self.reader,prompt,target,own_soft);negative=answer_nll(self.reader,prompt,target,wrong_soft)
                margin=F.relu(.5+ce-negative);return ce+.2*margin,ce.detach(),negative.detach(),margin.detach()
        objective=Objective();wrapped=DDP(objective,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
        reader_params=[p for p in policy.parameters() if p.requires_grad];compressor_params=list(compressor.parameters());active=reader_params+compressor_params
        frozen=[p for p in policy.parameters() if not p.requires_grad]+list(mapper.parameters());versions=[p._version for p in frozen]
        opt=torch.optim.AdamW([dict(params=reader_params,lr=cfg['reader_lr']),dict(params=compressor_params,lr=cfg['compressor_lr'])],weight_decay=.01)
        saved=torch.load(resume/'optimizer.pt',map_location='cpu',weights_only=True);assert saved['step']==287 and saved['epoch']==1
        opt.load_state_dict(saved['optimizer']);assert [g['lr'] for g in opt.param_groups]==[cfg['reader_lr'],cfg['compressor_lr']];assert_fp32_optimizer(opt)
        encoded={q['question_id']:prompt_and_target(tok,'Answer the question concisely and completely.\nQuestion: '+q['question'],q['answer'],device,512) for q in data['questions']}
        before=[p.detach().clone() for p in active];step=287;summaries=[];start=time.time()
        for epoch in [2,3]:
            order=list(data['questions']);random.Random(cfg['seed']+epoch).shuffle(order);objective.train();totals=torch.zeros(4,device=device,dtype=torch.float64)
            for offset in range(0,len(order),world):
                batch=order[offset:offset+world];present=rank<len(batch);q=batch[rank] if present else order[0]
                donor=random.Random(f"{cfg['seed']}|{epoch}|{q['question_id']}").choice([c for c in sorted(states) if c!=q['context_id']])
                prompt,target=encoded[q['question_id']];opt.zero_grad(set_to_none=True)
                loss,ce,negative,margin=wrapped(states[q['context_id']],states[donor],prompt,target)
                (loss*(world/len(batch) if present else 0.)).backward();grad=torch.nn.utils.clip_grad_norm_(active,1.)
                ok=torch.tensor(int(bool(torch.isfinite(loss)) and bool(torch.isfinite(grad))),device=device);dist.all_reduce(ok,op=dist.ReduceOp.MIN)
                if not ok.item():raise FloatingPointError('nonfinite resumed training')
                opt.step();step+=1;assert_fp32_optimizer(opt)
                assert versions==[p._version for p in frozen] and not any(p.grad is not None for p in frozen)
                stat=torch.tensor([float(ce),float(negative),float(margin),1.],device=device,dtype=torch.float64) if present else torch.zeros(4,device=device,dtype=torch.float64)
                totals+=stat;dist.all_reduce(stat)
                if rank==0:
                    metric=dict(epoch=epoch,step=step,total_steps=861,own_nll=float(stat[0]/stat[3]),wrong_nll=float(stat[1]/stat[3]),margin=float(stat[2]/stat[3]),grad=float(grad),examples=int(stat[3]),elapsed=time.time()-start)
                    append(run/'train/continuation_metrics.jsonl',metric)
                    if step%5==0:print(json.dumps(metric),flush=True)
            dist.all_reduce(totals);assert int(totals[3])==1146
            changed=sum(not torch.equal(a,p) for a,p in zip(before,active));assert changed>0
            if rank==0:
                root=run/f'train/epoch_{epoch}';root.mkdir();policy.save_pretrained(root/'adapter')
                torch.save(dict(state_dict=compressor.state_dict(),hidden=h,output_tokens_per_session=512,granularity='session_json',procedure='reader_initialization',smoke_only=False,
                    mapper_hash=sha(Path(cfg['mapper']).read_bytes()),protocol_hash=sha((run/'continuation_protocol.json').read_bytes()),parent_epoch=epoch-1,promotion_allowed=False),root/'compressor.pt')
                torch.save(dict(optimizer=opt.state_dict(),step=step,epoch=epoch),root/'optimizer.pt')
                summaries.append(dict(epoch=epoch,examples=1146,own_nll=float(totals[0]/totals[3]),wrong_nll=float(totals[1]/totals[3])))
                save(run/'continuation_progress.json',dict(status='running',step=step,epochs=summaries))
            dist.barrier()
        if rank==0:save(run/'continuation_result.json',dict(status='completed',resumed_optimizer=True,updates_total=step,epochs=summaries,checkpoint=str(run/'train/epoch_3'),parameters_changed=changed,frozen_unchanged=True,on_policy_optimization_launched=False))
    finally:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reader-initialization-root',type=Path,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args();run=a.reader_initialization_root.resolve()
    if a.worker:worker(run);return
    from .run_persona128k_alignment import gpu_free
    if (run/'train/epoch_2').exists() or (run/'continuation_protocol.json').exists():raise FileExistsError('continuation already exists')
    visible=os.environ.get('PIPELINE_CUDA_VISIBLE_DEVICES','0,1,2,3')
    physical=[int(x) for x in visible.split(',')]
    if len(physical)!=4 or not gpu_free(physical):raise RuntimeError(f'GPU set busy or invalid: {visible}')
    cfg=json.loads((run/'protocol.json').read_text());sources=[Path(__file__),run/'protocol.json',run/'data.json',run/'train/epoch_1/compressor.pt',run/'train/epoch_1/optimizer.pt',run/'train/epoch_1/adapter/adapter_model.safetensors',Path(cfg['mapper'])]
    save(run/'continuation_protocol.json',dict(from_epoch=1,to_epoch=3,resume_optimizer=True,selection='lowest mean answer NLL on LoCoMo validation among epochs1-3; LongMemEval excluded',sources={str(x):sha(x.read_bytes()) for x in sources},on_policy_optimization_launched=False))
    save(run/'status.json',dict(status='running',phase='continue_epochs2_3',updated=time.time(),on_policy_optimization_launched=False))
    try:
        with (run/'continuation.log').open('x') as log:
            rc=subprocess.call([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m',__spec__.name,'--reader-initialization-root',str(run),'--worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':visible,'TOKENIZERS_PARALLELISM':'false'})
        if rc:raise RuntimeError('continuation failed')
        save(run/'status.json',dict(status='completed',phase='epochs1_3_complete',selection_pending=True,updated=time.time(),on_policy_optimization_launched=False))
    except Exception as exc:save(run/'status.json',dict(status='failed',phase='continue_epochs2_3',error=repr(exc),updated=time.time()));raise


if __name__=='__main__':main()
