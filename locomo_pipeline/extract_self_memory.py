"""Four-GPU frozen memory-writer initialization extraction prerequisite; never launches training.

One independent vLLM replica per explicitly selected GPU. All train/validation
sessions retained; invalid or length-limited generations block data export.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .prepare import compact, jsonl, sha


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def inputs(prepared):
    rows=[r for split in ['train','validation']
          for r in jsonl(prepared/split/'writer_inputs.jsonl')]
    if not rows or len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate session inputs')
    for r in rows:
        if [m['role'] for m in r['writer_messages']] != ['system','user']:
            raise ValueError('writer must see only question-blind system/user input')
        if sha(r['writer_messages'][1]['content'].encode()) != r['input_sha256']:
            raise ValueError('writer input hash mismatch')
    return rows


def worker(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from .evaluate_memory_writer import decode_v3
    import torch
    torch.set_num_threads(4)
    protocol=json.loads((a.output/'protocol.json').read_text())
    if sha((a.checkpoint/'adapter_model.safetensors').read_bytes()) != protocol['adapter_sha256']:
        raise ValueError('adapter changed after extraction plan')
    rows=inputs(a.prepared)[a.shard::len(protocol['gpus'])]
    jobs={r['job_id']:r for r in jsonl(a.prepared/'source_sessions.jsonl')}
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    rank=json.loads((a.checkpoint/'adapter_config.json').read_text())['r']
    llm=LLM(model=str(a.model),dtype='bfloat16',tensor_parallel_size=1,
        enable_lora=True,max_lora_rank=rank,max_model_len=32768,
        gpu_memory_utilization=.8,max_num_seqs=16,max_num_batched_tokens=8192,
        enforce_eager=True,seed=42,enable_prefix_caching=True)
    adapter=LoRARequest('locomo-memory_writer_initialization',1,str(a.checkpoint))
    cap=protocol['max_new_tokens']
    params=SamplingParams(temperature=0,max_tokens=cap,repetition_penalty=1.)
    target=a.output/f'generations_{a.shard}.jsonl'
    if target.exists(): raise FileExistsError(target)
    for start in range(0,len(rows),16):
        batch=rows[start:start+16]
        prompts=[tok.apply_chat_template(r['writer_messages'],tokenize=False,
            add_generation_prompt=True,enable_thinking=False) for r in batch]
        if any(len(tok.encode(p,add_special_tokens=False))+cap>32768 for p in prompts):
            raise ValueError('writer input plus output cap exceeds context; no truncation')
        batch_params=params
        if protocol.get('structured_decoding'):
            from vllm.sampling_params import StructuredOutputsParams
            from .retry_invalid_self_memory import native_schema
            batch_params=[SamplingParams(temperature=0,max_tokens=cap,repetition_penalty=1.,
                structured_outputs=StructuredOutputsParams(json=native_schema(jobs[r['task_id']]['evidence_ids']),
                                                          disable_fallback=True)) for r in batch]
        outputs=llm.generate(prompts,batch_params,lora_request=adapter,use_tqdm=False)
        for row,prompt,output in zip(batch,prompts,outputs,strict=True):
            gen=output.outputs[0]
            record={k:row[k] for k in ['task_id','context_id','split','input_sha256']}
            record.update(origin='self_generated',source_adapter=str(a.checkpoint),
                adapter_sha256=protocol['adapter_sha256'],prompt_sha256=sha(prompt.encode()),
                raw_memory_text=gen.text,finish_reason=gen.finish_reason,tokens=len(gen.token_ids))
            record['decoding']='native_v3_schema' if protocol.get('structured_decoding') else 'unconstrained'
            try:
                if gen.finish_reason!='stop': raise ValueError('length-limited generation')
                value=decode_v3(gen.text,jobs[row['task_id']]['evidence_ids'])
                record.update(schema_valid=True,memory_text=compact(value))
            except (ValueError,TypeError,KeyError) as exc:
                record.update(schema_valid=False,error=str(exc))
            with target.open('a') as f: f.write(compact(record)+'\n')
        print(compact(dict(shard=a.shard,completed=min(start+16,len(rows)),total=len(rows))),flush=True)


def run(a):
    if a.output.exists(): raise FileExistsError('refusing to overwrite extraction')
    rows=inputs(a.prepared)
    gpu_csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'],text=True)
    usage={int(x.split(',')[0]):int(x.split(',')[1]) for x in gpu_csv.splitlines()}
    if any(usage.get(i,10**9)>1000 for i in a.gpus):
        raise RuntimeError('selected GPUs not all available; no jobs displaced')
    a.output.mkdir(parents=True)
    save(a.output/'protocol.json',dict(status='prerequisite_extraction',training_launched=False,
        prepared=str(a.prepared),model=str(a.model),checkpoint=str(a.checkpoint),gpus=a.gpus,
        adapter_sha256=sha((a.checkpoint/'adapter_model.safetensors').read_bytes()),
        prepared_manifest_sha256=sha((a.prepared/'manifest.json').read_bytes()),
        runner_sha256=sha(Path(__file__).read_bytes()),sessions=len(rows),
        max_new_tokens=a.max_new_tokens,structured_decoding=a.structured,
        maximum_sequence=32768,do_sample=False,enable_thinking=False))
    processes=[]; logs=[]
    try:
        for shard,gpu in enumerate(a.gpus):
            log=(a.output/f'worker_{shard}.log').open('w'); logs.append(log)
            command=[sys.executable,'-m','locomo_pipeline.extract_self_memory','worker',
                '--model',str(a.model),'--checkpoint',str(a.checkpoint),
                '--prepared',str(a.prepared),'--output',str(a.output),'--shard',str(shard)]
            processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'TOKENIZERS_PARALLELISM':'false',
                    'TRITON_CACHE_DIR':str(a.output/f'compiler_cache_{shard}'/'triton'),
                    'TORCHINDUCTOR_CACHE_DIR':str(a.output/f'compiler_cache_{shard}'/'inductor'),
                    'VLLM_CACHE_ROOT':str(a.output/f'compiler_cache_{shard}'/'vllm')}))
        codes=[p.wait() for p in processes]
        if any(codes): raise RuntimeError(f'extraction worker exit codes: {codes}')
        records=[r for shard in range(len(a.gpus)) for r in jsonl(a.output/f'generations_{shard}.jsonl')]
        if len(records)!=len(rows) or {r['task_id'] for r in records}!={r['task_id'] for r in rows}:
            raise ValueError('extraction coverage mismatch')
        with (a.output/'self_memories.jsonl').open('w') as f:
            for r in sorted(records,key=lambda r:r['task_id']): f.write(compact(r)+'\n')
        failures=[r['task_id'] for r in records if not r['schema_valid']]
        if failures:
            save(a.output/'result.json',dict(status='blocked_invalid_generations',
                sessions=len(records),invalid_ids=failures,training_launched=False))
            return
        subprocess.run([sys.executable,'-m','locomo_pipeline.prepare_compressor_reconstruction',
            '--prepared',str(a.prepared),'--self-memories',str(a.output/'self_memories.jsonl'),
            '--writer-checkpoint',str(a.checkpoint),'--output',str(a.output/'prepared')],check=True)
        save(a.output/'result.json',dict(status='extraction_and_export_completed',
            sessions=len(records),training_launched=False,reasoning_pretraining_complete=False))
    except Exception as exc:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            if p.poll() is None: p.wait(timeout=30)
        save(a.output/'result.json',dict(status='failed',error=repr(exc),training_launched=False))
        raise
    finally:
        for log in logs: log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run','worker'])
    for name in ['model','checkpoint','prepared','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--shard',type=int,choices=range(4))
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3],choices=range(4))
    p.add_argument('--max-new-tokens',type=int,default=8192)
    p.add_argument('--structured',action='store_true',help='native JSON schema and source evidence IDs only; no QA/gold input')
    a=p.parse_args()
    if len(set(a.gpus))!=len(a.gpus) or not 1<=a.max_new_tokens<32768:
        p.error('unique GPUs and valid positive generation cap required')
    for name in ['model','checkpoint','prepared','output']:
        setattr(a,name,getattr(a,name).resolve())
    if a.mode=='worker' and a.shard is None: p.error('worker requires --shard')
    (run if a.mode=='run' else worker)(a)


if __name__=='__main__': main()
