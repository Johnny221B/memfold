"""Synchronous 1 teacher : 3 student : 4 rollout, GRPO-matched effective batch16."""
import sys
import argparse,json,os,signal,subprocess,time,hashlib,shutil,fcntl
from pathlib import Path
import torch
from core import *
R=Path(__file__).resolve().parent;W=R.parent.parent;PY=Path(sys.executable)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=369);p.add_argument('--smoke',action='store_true');p.add_argument('--pilot',action='store_true');p.add_argument('--opd-weight',type=float);p.add_argument('--config',type=Path,default=R/'config.json');a=p.parse_args();a.output=a.output.resolve()
    assert 0<a.steps<=369
    assert a.smoke or (a.pilot and a.steps==37) or a.steps==369,'production run must retain369 optimizer steps'
    assert not a.output.exists(),'output exists; joint student/EMA resume is not implemented'
    with (R/'gpu-allocation.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
            i,mem,util=map(int,line.split(','));assert mem<1024 and util<5,f'GPU {i} busy; do not preempt'
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();assert not active,'active CUDA processes'
        a.output.mkdir(parents=True)
        cfg=json.loads(a.config.read_text());cfg.update(status='running',implementation='single-node-synchronous-ddp-lora',actual_steps=a.steps,smoke=a.smoke,lr_schedule_total_steps=369,rollout_tensor_parallel=1,rollout_replicas=4,vllm_importance_sampling_mode='sequence_mask',vllm_importance_sampling_cap=3.)
        if a.opd_weight is not None:
            assert a.opd_weight>=0;cfg['opd_weight']=a.opd_weight
        cfg['pilot']=a.pilot
        for key in ('train_data','teacher_adapter','teacher_memories','teacher_questions'):
            if key in cfg: cfg[key]=str(Path(cfg[key]).resolve())
        if Path(cfg['model']).exists(): cfg['model']=str(Path(cfg['model']).resolve())
        (a.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(cfg['model']);questions=[json.loads(l) for l in Path(cfg['train_data']).open()]
        for q in questions:
            q['prompt_ids']=tokenizer.apply_chat_template([dict(role='user',content=q['prompt'])],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            assert len(q['prompt_ids'])+5<=32768
        assert len(questions)==489
        from memory_opd.compressed_opd import load_compressed_opd_examples
        ex={x.question_id:x for x in load_compressed_opd_examples(Path(cfg['teacher_questions']),Path(cfg['teacher_memories']),expected_split='train')}
        assert set(ex)=={q['question_id'] for q in questions}
        for q in questions:
            x=ex[q['question_id']];assert x.question==q['question'] and list(x.options)==q['options'] and x.gold_label==q['answer']
        source_hash=hashlib.sha256(Path(cfg['train_data']).read_bytes()).hexdigest()
        (a.output/'data-audit.json').write_text(json.dumps(dict(examples=489,source_sha256=source_hash,max_prompt_tokens=max(len(q['prompt_ids']) for q in questions))))
        generator=torch.Generator().manual_seed(42);plan=[]
        while len(plan)<a.steps:
            order=torch.randperm(len(questions),generator=generator).tolist()
            plan.extend([order[i:i+2] for i in range(0,len(order)-1,2)])
        plan=plan[:a.steps]
        children=[];handles=[];start=time.monotonic()
        def status(state,**kw):
            data=dict(state=state,pid=os.getpid(),time=time.time(),**kw);tmp=a.output/'status.json.tmp';tmp.write_text(json.dumps(data,indent=2));tmp.replace(a.output/'status.json')
        def spawn(role,gpus,index=0):
            args=[str(PY)]
            if role=='student':args+=['-m','torch.distributed.run','--standalone','--nproc_per_node','3']
            args += [str(R/'roles.py'),'--role',role,'--run-dir',str(a.output),'--steps',str(a.steps),'--index',str(index)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',VLLM_WORKER_MULTIPROC_METHOD='spawn',PYTHONPATH=f'{R}:{R}/src',TRITON_CACHE_DIR=str(R/'.cache/triton'),XDG_CACHE_HOME=str(R/'.cache'),HF_HOME=str(W/'cache/huggingface'),TORCHINDUCTOR_CACHE_DIR=str(R/'.cache/inductor'))
            f=(a.output/f'{role}-{index}.log').open('a');handles.append(f)
            child=subprocess.Popen(args,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append(child)
        try:
            spawn('teacher',[0]);spawn('student',[1,2,3])
            for i in range(4):spawn('rollout',[i+4],i)
            status('initializing',children=[p.pid for p in children])
            for name in ['student','teacher']+[f'rollout-{i}' for i in range(4)]:wait(a.output/f'ready-{name}.pt',a.output)
            loop_start=time.monotonic();checkpoint_seconds=0.
            for step,indices in enumerate(plan,1):
                tick=time.monotonic();status('training',step=step,optimizer_steps_completed=step-1,question_indices=indices)
                save(a.output/f'rollout-{step}.pt',dict(version=step-1,questions=[questions[i] for i in indices]))
                rows=[]
                for i in range(4):
                    piece=wait(a.output/f'rollout-{step}-{i}.pt',a.output);assert piece['version']==step-1;rows+=piece['rows']
                assert len(rows)==16
                for r in rows:r['reward']=float(r['text'].strip()==r['gold'])
                save(a.output/f'batch-{step}.pt',dict(version=step-1,rows=rows))
                old=wait(a.output/f'old-{step}.pt',a.output);teacher=wait(a.output/f'teacher-{step}.pt',a.output)
                assert teacher['version']==step-1
                save(a.output/f'targets-{step}.pt',dict(teacher_version=step-1,teacher=teacher['logp'],feedback=teacher['feedback'],grpo=grpo_advantages([r['reward'] for r in rows])))
                update=wait(a.output/f'updated-{step}.pt',a.output);ema=wait(a.output/f'ema-{step}.pt',a.output);assert ema['version']==step
                checkpoint_seconds+=update.get('checkpoint_save_seconds',0.)+ema.get('checkpoint_save_seconds',0.)
                metric=dict(train_loop_seconds=time.monotonic()-loop_start-checkpoint_seconds,cumulative_checkpoint_seconds=checkpoint_seconds,step=step,seconds=time.monotonic()-tick,cumulative_seconds=time.monotonic()-start,real_rollouts=16,unique_questions=len(set(r['qid'] for r in rows)),dummy_slots=2,teacher_version=ema['version'],reward_mean=sum(r['reward'] for r in rows)/16,feedback_samples=sum(teacher['feedback']),**{k:v for k,v in update.items() if k!='step'})
                with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
                print(json.dumps(metric),flush=True)
                # All consumers finished scoring/generating previous adapter version.
                shutil.rmtree(a.output/f'adapters/v{step-1}')
            for child in children:
                code=child.wait(timeout=180);assert code==0,(child.pid,code)
            (a.output/'done').touch();status('completed',optimizer_steps_completed=a.steps,elapsed_seconds=time.monotonic()-start)
        except BaseException:
            status('failed',error=__import__('traceback').format_exc());raise
        finally:
            # Only owned process groups; never signal unrelated GPU workloads.
            for child in children:
                if child.poll() is None:
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for f in handles:f.close()
if __name__=='__main__':main()
