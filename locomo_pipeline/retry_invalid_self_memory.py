"""One bounded structured-decoding retry of invalid memory-writer initialization sessions, not QA repair."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .compressor_data import read, save, sha
from .prepare import compact
from .extract_self_memory import inputs
from memory_extraction import extract_locomo_memory as schema

ROOT=Path(__file__).resolve().parent.parent
PRIOR=ROOT/'locomo_pipeline/runs/reader_initialization_self_e5_20260907_retry1'
OUT=ROOT/'locomo_pipeline/runs/reader_initialization_self_e5_constrained_v1'
PREP=ROOT/'locomo_pipeline/prepared/full_v3_20260906'
MODEL=ROOT/'models/Qwen3-4B'
ADAPTER=ROOT/'locomo_pipeline/runs/qwen3_4b_v3_e5_20260906/train/epoch_5'


def object_schema(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


def native_schema(evidence):
    string=dict(type='string',minLength=1)
    enum=lambda values:dict(type='string',enum=sorted(values))
    temporal=dict(certainty=enum(schema.CERTAINTIES),end=dict(type='string',pattern=r'^(\d{4}(-\d{2}(-\d{2})?)?)?$'),
        granularity=enum(schema.GRANULARITIES-{'none'}),source_expression=dict(type='string'),
        start=dict(type='string',pattern=r'^(\d{4}(-\d{2}(-\d{2})?)?)?$'))
    absent={**temporal,'granularity':dict(type='string',enum=['none']),
            'start':dict(type='string',enum=['']),'end':dict(type='string',enum=[''])}
    properties=dict(attribute=dict(type='string',pattern=r'^[a-z][a-z0-9_]{0,63}$'),
        confidence=dict(type='number',minimum=0,maximum=1),
        evidence_ids=dict(type='array',items=enum(evidence),minItems=1,maxItems=4),
        explicitness=enum(schema.EXPLICITNESS),facet=enum(schema.FACETS),memory_type=enum(schema.MEMORY_TYPES),
        polarity=enum(schema.POLARITIES),state_status=enum(schema.STATE_STATUSES),statement=string,subject=string,
        temporal=dict(anyOf=[object_schema(temporal),object_schema(absent)]),value=string)
    return object_schema(dict(memories=dict(type='array',items=object_schema(properties))))


def worker(shard):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.lora.request import LoRARequest
    from .evaluate_memory_writer import decode_v3
    torch.set_num_threads(4)
    protocol=json.loads((OUT/'protocol.json').read_text())
    if sha((ADAPTER/'adapter_model.safetensors').read_bytes())!=protocol['adapter_sha256']:
        raise ValueError('adapter changed')
    byid={r['task_id']:r for r in inputs(PREP)}
    jobs={r['job_id']:r for r in read(PREP/'source_sessions.jsonl')}
    selected=protocol['invalid_ids'][shard::4]
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    llm=LLM(model=str(MODEL),dtype='bfloat16',enable_lora=True,max_lora_rank=8,
        max_model_len=32768,gpu_memory_utilization=.65,max_num_seqs=16,
        max_num_batched_tokens=8192,enforce_eager=True,seed=42,enable_prefix_caching=True)
    adapter=LoRARequest('memory_writer_initialization-e5',1,str(ADAPTER))
    prompts=[]; parameters=[]
    for qid in selected:
        prompt=tok.apply_chat_template(byid[qid]['writer_messages'],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        if len(tok.encode(prompt,add_special_tokens=False))+16384>32768:
            raise ValueError('input/output budget overflow; no truncation')
        prompts.append(prompt)
        parameters.append(SamplingParams(temperature=0,max_tokens=16384,repetition_penalty=1.,
            structured_outputs=StructuredOutputsParams(json=native_schema(jobs[qid]['evidence_ids']),disable_fallback=True)))
    outputs=llm.generate(prompts,parameters,lora_request=adapter,use_tqdm=False)
    for qid,prompt,output in zip(selected,prompts,outputs,strict=True):
        src=byid[qid]; gen=output.outputs[0]
        r={k:src[k] for k in ['task_id','context_id','split','input_sha256']}
        r.update(origin='self_generated',source_adapter=str(ADAPTER),adapter_sha256=protocol['adapter_sha256'],
            prompt_sha256=sha(prompt.encode()),raw_memory_text=gen.text,finish_reason=gen.finish_reason,
            tokens=len(gen.token_ids),decoding='native_v3_schema_with_allowed_evidence_16384',retry_round=1)
        try:
            if gen.finish_reason!='stop': raise ValueError('length-limited generation')
            r.update(memory_text=compact(decode_v3(gen.text,jobs[qid]['evidence_ids'])),schema_valid=True)
        except (ValueError,KeyError,TypeError) as exc:
            r.update(schema_valid=False,error=str(exc))
        with (OUT/f'generations_{shard}.jsonl').open('a') as f: f.write(compact(r)+'\n')
        print(compact(dict(id=qid,valid=r['schema_valid'],tokens=r['tokens'])),flush=True)


def supervise():
    previous=read(PRIOR/'self_memories.jsonl')
    processes=[]
    try:
        for shard in range(4):
            with (OUT/f'worker_{shard}.log').open('x') as log:
                processes.append(subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'--shard',str(shard)],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(shard),
                    'TRITON_CACHE_DIR':str(OUT/f'cache_{shard}/triton'),
                    'VLLM_CACHE_ROOT':str(OUT/f'cache_{shard}/vllm'),
                    'TORCHINDUCTOR_CACHE_DIR':str(OUT/f'cache_{shard}/inductor'),
                    'TOKENIZERS_PARALLELISM':'false'}))
        codes=[p.wait() for p in processes]
        if any(codes): raise RuntimeError(f'workers failed: {codes}')
        retries=[r for i in range(4) for r in read(OUT/f'generations_{i}.jsonl')]
        expected={r['task_id'] for r in previous if not r['schema_valid']}
        if len(retries)!=len(expected) or {r['task_id'] for r in retries}!=expected: raise ValueError('retry coverage')
        lookup={r['task_id']:r for r in retries}
        merged=[lookup.get(r['task_id'],r) for r in previous]
        with (OUT/'self_memories.jsonl').open('x') as f:
            for r in merged: f.write(compact(r)+'\n')
        failures=[r['task_id'] for r in merged if not r['schema_valid']]
        if not failures:
            subprocess.run([sys.executable,'-m','locomo_pipeline.prepare_compressor_reconstruction','--prepared',str(PREP),
                '--self-memories',str(OUT/'self_memories.jsonl'),'--writer-checkpoint',str(ADAPTER),
                '--output',str(OUT/'prepared')],cwd=ROOT,check=True)
        save(OUT/'result.json',dict(status='completed' if not failures else 'blocked_invalid_generations',
            sessions=len(merged),retried=len(retries),invalid_ids=failures,training_launched=False))
    except Exception as exc:
        for p in processes:
            if p.poll() is None: p.terminate()
        save(OUT/'result.json',dict(status='failed',error=repr(exc),training_launched=False)); raise


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--shard',type=int,choices=range(4)); p.add_argument('--supervise',action='store_true'); a=p.parse_args()
    if a.shard is not None: worker(a.shard); return
    if a.supervise: supervise(); return
    if OUT.exists(): raise FileExistsError('refusing overwrite')
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    if any(int(s.split(',')[1])>1000 for s in usage.splitlines() if int(s.split(',')[0]) in range(4)):
        raise RuntimeError('GPU 0-3 occupied')
    previous=read(PRIOR/'self_memories.jsonl')
    OUT.mkdir(parents=True)
    save(OUT/'protocol.json',dict(invalid_ids=sorted(r['task_id'] for r in previous if not r['schema_valid']),
        prior_sha256=sha((PRIOR/'self_memories.jsonl').read_bytes()),adapter_sha256=sha((ADAPTER/'adapter_model.safetensors').read_bytes()),
        rules='same original input and memory-writer initialization weights; native JSON schema/evidence constrained decoding; budget 16384; no QA/gold',
        semantic_correctness_verified=False))
    with (OUT/'supervisor.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'--supervise'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(compact(dict(pid=child.pid,output=str(OUT))))


if __name__=='__main__': main()
