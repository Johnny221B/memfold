"""Bounded local continuation through aligned pretraining, self-memory and reader initialization.

GPU0: pretraining. GPU1-3: concurrent native-schema self extraction.
After both pass: four-GPU reader initialization smoke, five epochs, paired QA NLL audit.
Never invokes remote teachers, drops invalid sessions, or launches on-policy optimization.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from .compressor_data import read, save, sha, memory_bank, qa_rows, reasoning_rows
from .persona128k_recipe import check_adapter

ROOT=Path(__file__).resolve().parent.parent


def gpu_free(gpus):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    usage={int(line.split(',')[0]):int(line.split(',')[1]) for line in output.splitlines()}
    return all(usage.get(g,10**9)<=1000 for g in gpus)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--writer-adapter',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'models/Qwen3-4B')
    p.add_argument('--prepared',type=Path,default=ROOT/'locomo_pipeline/prepared/full_v3_20260906')
    p.add_argument('--api-memories',type=Path,default=ROOT/'locomo_pipeline/prepared/api_v3_compressor_v1/train.jsonl')
    p.add_argument('--reasoning-data',type=Path,default=ROOT/'locomo_pipeline/runs/reasoning_pretrain1142_v1_20260907')
    a=p.parse_args()
    for k,v in vars(a).items(): setattr(a,k,v.resolve())
    if a.output.exists(): raise FileExistsError('refusing overwrite; interrupted runs require explicit review')
    check_adapter(json.loads((a.writer_adapter/'adapter_config.json').read_text()),'persona128k')
    report=json.loads((a.writer_adapter.parent/'report.json').read_text())
    if report['status']!='completed' or a.writer_adapter.name!=f"epoch_{report['best_epoch']}":
        raise ValueError('use the completed memory-writer initialization validation-selected checkpoint')
    bank=memory_bank(read(a.api_memories))
    questions=qa_rows(read(a.reasoning_data/'questions.jsonl'),bank)
    traces=reasoning_rows(read(a.reasoning_data/'traces.jsonl'),questions,sha(a.api_memories.read_bytes()))
    if not gpu_free([0,1,2,3]): raise RuntimeError('GPU0-3 must be free; no jobs displaced')
    paths=[a.api_memories,a.reasoning_data/'questions.jsonl',a.reasoning_data/'traces.jsonl',
           a.writer_adapter/'adapter_config.json',a.writer_adapter/'adapter_model.safetensors',
           a.writer_adapter.parent/'report.json',a.model/'config.json',a.model/'tokenizer.json']
    paths += [a.prepared/s/n for s in ('train','validation') for n in ('qa.jsonl','writer_inputs.jsonl')]
    paths += [a.prepared/'manifest.json',a.prepared/'source_sessions.jsonl']
    paths += [Path(__file__).parent/n for n in ('run_persona128k_alignment.py','train_compressor.py',
              'compressor_core.py','persona128k_recipe.py','extract_self_memory.py','retry_invalid_self_memory.py',
              'prepare_compressor_reconstruction.py','train_joint_reader_initialization.py','audit_reader_initialization.py')]
    hashes={str(path):sha(path.read_bytes()) for path in paths}
    a.output.mkdir(parents=True)
    save(a.output/'protocol.json',dict(recipe='persona128k',args={k:str(v) for k,v in vars(a).items()},
        sources_sha256=hashes,selected_writer_epoch=report['best_epoch'],pretraining_questions=len(questions),
        audited_traces=len(traces),pretraining_sessions=sum(map(len,bank.values())),
        gpus=dict(pretraining=[0],self_memory=[1,2,3],joint_reader_initialization=[0,1,2,3]),
        epochs=dict(compressor_reconstruction=3,auxiliary_reasoning_adaptation=3,reader_initialization=5),warmup_steps=100,effective_reasoning_batch=4,
        self_memory_decoding='greedy native-v3 schema, allowed source evidence IDs, cap16384; no gold QA',
        precision='FP32 compressor/master/Adam, BF16 frozen base',
        invalid_sessions='block reader initialization; no automatic filtering or API replacement',
        on_policy_optimization_launched=False,remote_api_calls=False))
    lock=threading.Lock()
    status=dict(status='running',started=time.time(),tasks={},on_policy_optimization_launched=False)

    def update(name,**fields):
        with lock:
            status['updated']=time.time()
            if name is None: status.update(fields)
            else: status['tasks'].setdefault(name,{}).update(fields)
            save(a.output/'status.tmp',status)
            (a.output/'status.tmp').replace(a.output/'status.json')
            print(json.dumps(dict(task=name,**fields)),flush=True)

    def check_sources():
        for name,digest in hashes.items():
            if sha(Path(name).read_bytes())!=digest: raise ValueError('source changed during run: '+name)

    def run(name,argv,gpus):
        check_sources()
        deadline=time.monotonic()+180
        while not gpu_free(gpus):
            update(name,status='waiting_gpus',gpus=gpus)
            if time.monotonic()>deadline: raise RuntimeError('selected GPU did not become available: '+name)
            time.sleep(10)
        command=[sys.executable,*map(str,argv)]
        save(a.output/(name+'_command.json'),command)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus)),
             'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
        with (a.output/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            update(name,status='running',pid=child.pid,gpus=gpus)
            code=child.wait()
        update(name,status='completed' if code==0 else 'failed',exit_code=code)
        if code: raise RuntimeError(f'{name} failed; inspect {name}.log')
        check_sources()

    pre=a.output/'pretraining'
    extraction=a.output/'self_memory'
    reader_initialization=a.output/'reader_initialization'
    pre.mkdir(); reader_initialization.mkdir()
    common=['--recipe','persona128k','--model',a.model,'--memories',a.api_memories]

    def pretrain():
        try:
            run('api_cache',['-m','locomo_pipeline.train_compressor','cache',*common,'--output',pre/'cache'],[0])
            for name,parent,extra in (
                ('compressor_reconstruction',None,['--epochs','3']),
                ('representation_warmup',pre/'compressor_reconstruction/epoch_3/bridge.pt',['--warmup-steps','100']),
                ('auxiliary_reasoning_adaptation',pre/'representation_warmup/epoch_1/bridge.pt',['--epochs','3','--questions',a.reasoning_data/'questions.jsonl',
                    '--traces',a.reasoning_data/'traces.jsonl','--maximum-target-tokens','256'])):
                args=['-m','locomo_pipeline.train_compressor',name,*common,'--cache',pre/'cache',
                      '--output',pre/name,*extra]
                if parent: args+=['--compressor-checkpoint',parent]
                run(name,args,[0])
                result=json.loads((pre/name/'result.json').read_text())
                expected={'compressor_reconstruction':sum(map(len,bank.values()))*3,'representation_warmup':100,'auxiliary_reasoning_adaptation':((len(questions)+3)//4)*3}[name]
                if result['status']!='completed' or result['updates']!=expected or result['provenance']['smoke_only']:
                    raise ValueError('pretraining completion/coverage mismatch')
        except Exception as exc:
            update('pretraining',status='failed',error=repr(exc)); raise
        update('pretraining',status='completed')

    def extract():
        run('self_extract',['-m','locomo_pipeline.extract_self_memory','run','--model',a.model,
            '--checkpoint',a.writer_adapter,'--prepared',a.prepared,'--output',extraction,
            '--gpus','1','2','3','--structured','--max-new-tokens','16384'],[1,2,3])
        result=json.loads((extraction/'result.json').read_text())
        update('self_memory',status=result['status'],invalid_ids=result.get('invalid_ids',[]))
        if result['status']!='extraction_and_export_completed':
            raise ValueError('self-memory incomplete/invalid; no automatic sample filtering')

    try:
        update(None,status='running')
        # Independent branches: an invalid self-memory result must not erase useful pretraining.
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(pretrain),pool.submit(extract)]
            errors=[]
            for job in jobs:
                try: job.result()
                except Exception as exc: errors.append(repr(exc))
        if errors:
            update(None,status='blocked_before_reader_initialization',errors=errors); return
        memories=extraction/'prepared/train.jsonl'
        run('self_cache',['-m','locomo_pipeline.train_compressor','cache','--recipe','persona128k',
            '--model',a.model,'--memories',memories,'--output',reader_initialization/'cache'],[0])
        args=['-m','locomo_pipeline.train_joint_reader_initialization','--recipe','persona128k','--model',a.model,
            '--memories',memories,'--cache',reader_initialization/'cache','--compressor-checkpoint',pre/'auxiliary_reasoning_adaptation/epoch_3/bridge.pt',
            '--writer-adapter',a.writer_adapter,'--source-self-memories',extraction/'self_memories.jsonl',
            '--prepared',a.prepared,'--epochs','5']
        run('reader_initialization_preflight',args+['--validate-only','--output',reader_initialization/'preflight'],[0])
        for name,extra in [('smoke',['--max-updates','2']),('train',[])]:
            run('reader_initialization_'+name,['-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                *args,'--output',reader_initialization/name,*extra],[0,1,2,3])
            result=json.loads((reader_initialization/name/'result.json').read_text())
            if result['status']!=('smoke_completed' if name=='smoke' else 'completed') or not result['frozen_parameters_unchanged']:
                raise ValueError('reader initialization completion/scope check failed')
        run('qa_audit',['-m','locomo_pipeline.audit_reader_initialization','--run',reader_initialization,
            '--memories',extraction/'prepared','--cache',reader_initialization/'cache','--output',reader_initialization/'paired_qa50'],[0])
        if json.loads((reader_initialization/'paired_qa50/report.json').read_text())['status']!='completed':
            raise ValueError('QA NLL audit incomplete')
        update(None,status='completed',quality_metric='paired QA NLL, not answer accuracy')
    except Exception as exc:
        update(None,status='failed',error=repr(exc)); raise


if __name__=='__main__': main()
