import argparse,json,os,time,contextlib
from pathlib import Path
import torch
from core import *

def student(args,cfg):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from peft import get_peft_model,LoraConfig
    from transformers import AutoModelForCausalLM,get_scheduler,set_seed
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');set_seed(cfg['seed'])
    base=AutoModelForCausalLM.from_pretrained(cfg['model'],torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',local_files_only=True).to(rank)
    base.config.use_cache=False
    policy=get_peft_model(base,LoraConfig(r=64,lora_alpha=128,lora_dropout=0,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});policy.enable_input_require_grads()
    ddp=DDP(policy,device_ids=[rank],broadcast_buffers=False)
    params=[p for p in policy.parameters() if p.requires_grad]
    assert all(p.dtype==torch.float32 for p in params)
    optim=torch.optim.AdamW(params,lr=5e-6,weight_decay=0,fused=True)
    scheduler=get_scheduler('linear',optim,num_warmup_steps=0,num_training_steps=369)
    root=args.run
    if rank==0:
        policy.save_pretrained(root/'adapters/v0');save(root/'ready-student.pt',dict(trainable_parameters=sum(p.numel() for p in params),lora_dtype='float32'))
    for step in range(1,args.steps+1):
        batch=wait(root/f'batch-{step}.pt',root);assert batch['version']==step-1;rows=batch['rows']
        old=torch.zeros(16,5,device=rank)
        policy.eval()
        with torch.no_grad():
            for i in local_slots(rank):
                if i is not None:
                    r=rows[i];lp=response_logp(policy,r['prompt_ids'],r['response']);old[i,:len(lp)]=lp
        dist.all_reduce(old)
        if rank==0:save(root/f'old-{step}.pt',old.cpu())
        targets=wait(root/f'targets-{step}.pt',root);assert targets['teacher_version']==step-1
        policy.train();optim.zero_grad();loss_sum=0.
        diagnostics={}
        if cfg['smoke'] and step<=cfg['gradient_diagnostic_steps']:
            # Use policy directly (not DDP.forward) for autograd.grad; all-reduce explicitly.
            grads_g=[torch.zeros_like(p) for p in params];grads_o=[torch.zeros_like(p) for p in params]
            for i in local_slots(rank):
                if i is None:continue
                r=rows[i];n=len(r['response']);lg,lo=response_logps(policy,r['prompt_ids'],r['response'])
                g,o,_=text_loss_parts(lg,lo,old[i,:n],targets['teacher'][i,:n].to(rank),targets['grpo'][i].to(rank),torch.tensor(r['sampling_logp'],device=rank),cfg['gate_beta'])
                gg=torch.autograd.grad(g,params,retain_graph=True);go=torch.autograd.grad(o,params)
                for dst,src in zip(grads_g,gg):dst.add_(src.detach(),alpha=1/16)
                for dst,src in zip(grads_o,go):dst.add_(src.detach(),alpha=1/16)
                del gg,go,lg,lo,g,o
            for buffer in grads_g+grads_o:dist.all_reduce(buffer)
            ng=sum(x.square().sum() for x in grads_g).sqrt();no=sum(x.square().sum() for x in grads_o).sqrt()
            dot=sum((g*o).sum() for g,o in zip(grads_g,grads_o))
            diagnostics=dict(grpo_gradient_norm=float(ng),opd_gradient_norm=float(no),gradient_cosine=float(dot/(ng*no).clamp_min(1e-12)),weighted_opd_to_grpo_gradient_ratio=float(cfg['opd_weight']*no/(cfg['grpo_weight']*ng).clamp_min(1e-12)))
            del grads_g,grads_o
        stats=torch.zeros(4,device=rank)
        for j,i in enumerate(local_slots(rank)):
            with ddp.no_sync() if j<5 else contextlib.nullcontext():
                if i is None:
                    lp=response_logp(ddp,[1,2],[3]);loss=lp.sum()*0
                else:
                    r=rows[i];n=len(r['response']);lg,lo=response_logps(ddp,r['prompt_ids'],r['response'])
                    g,o,om=text_loss_parts(lg,lo,old[i,:n],targets['teacher'][i,:n].to(rank),targets['grpo'][i].to(rank),torch.tensor(r['sampling_logp'],device=rank),cfg['gate_beta'])
                    loss=(cfg['grpo_weight']*g+cfg['opd_weight']*o)*3/16
                    stats+=torch.stack([g.detach(),o.detach(),om.gate_mean,om.teacher_gap_mean])/16
                loss_sum+=float(loss.detach());loss.backward()
        dist.all_reduce(stats)
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        assert torch.isfinite(norm),'nonfinite gradient; teacher update prohibited'
        lr=optim.param_groups[0]['lr'];optim.step();scheduler.step()
        dist.barrier()
        if rank==0:
            policy.save_pretrained(root/f'adapters/v{step}')
            checkpoint_start=time.monotonic();checkpoint_seconds=0.
            if step in cfg['checkpoint_steps'] or step==args.steps:
                ckpt=root/f'checkpoints/step-{step:06d}';policy.save_pretrained(ckpt/'adapter')
                save(ckpt/'optimizer.pt',dict(optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),step=step))
                checkpoint_seconds=time.monotonic()-checkpoint_start
            save(root/f'updated-{step}.pt',dict(step=step,grad_norm=float(norm),learning_rate=lr,checkpoint_save_seconds=checkpoint_seconds,grpo_loss=float(stats[0]),opd_loss=float(stats[1]),gate_mean=float(stats[2]),teacher_gap_mean=float(stats[3]),**diagnostics))
        dist.barrier()
    dist.destroy_process_group()

def teacher(args,cfg):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from memory_opd.compressed_opd import load_compressed_opd_examples,text_reader_messages
    root=args.run;torch.cuda.set_device(0)
    base=AutoModelForCausalLM.from_pretrained(cfg['model'],torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',local_files_only=True).cuda()
    model=PeftModel.from_pretrained(base,cfg['teacher_adapter'],is_trainable=False).eval().requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'])
    examples=load_compressed_opd_examples(Path(cfg['teacher_questions']),Path(cfg['teacher_memories']),expected_split='train')
    prompts={x.question_id:tokenizer.apply_chat_template(text_reader_messages(x.memory_text,x.question,x.options),tokenize=True,add_generation_prompt=True,enable_thinking=False) for x in examples}
    assert len(prompts)==489 and max(map(len,prompts.values()))+5<=32768
    with torch.no_grad():
        compact=response_logp(model,[10,11],[12,13],temperature=1.)
        ids=torch.tensor([[10,11,12,13]],device='cuda');full=model(input_ids=ids,use_cache=False).logits[:,1:3].float()
        expected=full.log_softmax(-1).gather(-1,ids[:,2:,None]).flatten();torch.testing.assert_close(compact,expected,atol=2e-5,rtol=2e-5)
    save(root/'ready-teacher.pt',dict(teacher_frozen=True,teacher_adapter=cfg['teacher_adapter'],compact_logits_verified=True))
    for step in range(1,args.steps+1):
        batch=wait(root/f'batch-{step}.pt',root);assert batch['version']==step-1
        scores=torch.zeros(16,5)
        for i,r in enumerate(batch['rows']):
            ids=prompts[r['qid']]
            with torch.no_grad():lp=response_logp(model,ids,r['response'],temperature=1.)
            scores[i,:len(lp)]=lp.cpu()
        save(root/f'teacher-{step}.pt',dict(version=step-1,logp=scores,feedback=[True]*16))
        wait(root/f'updated-{step}.pt',root)
        save(root/f'ema-{step}.pt',dict(version=step,checkpoint_save_seconds=0.,teacher_frozen=True))

def rollout(args,cfg):
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    root=args.run;index=args.index
    llm=LLM(model=cfg['model'],tensor_parallel_size=1,max_model_len=32768,dtype='bfloat16',gpu_memory_utilization=.70,enable_lora=True,max_lora_rank=64,max_loras=1,max_cpu_loras=1,enforce_eager=True,seed=42+index)
    save(root/f'ready-rollout-{index}.pt',dict(index=index))
    for step in range(1,args.steps+1):
        job=wait(root/f'rollout-{step}.pt',root);assert job['version']==step-1
        item=job['questions'][index//2]
        params=SamplingParams(n=4,temperature=1.2,top_p=1.,max_tokens=5,logprobs=1,seed=42+step*4+index)
        result=llm.generate([{'prompt_token_ids':item['prompt_ids']}],params,lora_request=LoRARequest(f'v{step-1}',step,str(root/f'adapters/v{step-1}')),use_tqdm=False)[0]
        rows=[]
        for o in result.outputs:
            response=list(o.token_ids);assert 0<len(response)<=5
            rows.append(dict(qid=item['question_id'],raw_prompt=item['prompt'],prompt_ids=item['prompt_ids'],gold=item['answer'],response=response,text=o.text.strip(),sampling_logp=[float(x[token].logprob) for x,token in zip(o.logprobs,response)]))
        save(root/f'rollout-{step}-{index}.pt',dict(version=step-1,rows=rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['student','teacher','rollout'],required=True);p.add_argument('--run-dir',dest='run',type=Path,required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--index',type=int,default=0);args=p.parse_args();cfg=json.loads((args.run/'config.json').read_text())
    try:globals()[args.role](args,cfg)
    except BaseException:fail(args.run,args.role+'-'+os.environ.get('LOCAL_RANK',str(args.index)));raise
if __name__=='__main__':main()
