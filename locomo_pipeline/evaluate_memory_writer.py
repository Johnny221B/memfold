"""Fixed validation-session generation audit; no training or test-set tuning."""
import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_extraction import extract_locomo_glm_memory_v3 as api
from .prepare import compact, jsonl, sha


def decode_v3(text, evidence):
    fence=re.fullmatch(r'\s*```(?:json)?\s*\n([\s\S]*?)\n```\s*',text)
    value=json.loads(fence.group(1) if fence else text)
    # Stricter than alias-normalizing API ingestion: report emitted schema as-is.
    if isinstance(value,dict) and isinstance(value.get('memories'),list):
        for m in value['memories']:
            if isinstance(m,dict) and m.get('memory_type') not in api.MEMORY_TYPES:
                raise ValueError('noncanonical memory_type')
    memories=api.validate_memory(value,set(evidence))
    return {'memories':[{k:v for k,v in m.items() if k!='memory_key'} for m in memories]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--checkpoint',required=True,help='base or adapter path')
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sessions-per-conversation',type=int,default=2)
    p.add_argument('--max-new-tokens',type=int,default=8192)
    p.add_argument('--disable-thinking',action='store_true')
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError('refusing overwrite')
    rows=jsonl(a.prepared/'validation/writer_sft.jsonl')
    selected=[]
    for context in sorted({r['context_id'] for r in rows}):
        selected.extend(sorted([r for r in rows if r['context_id']==context],
            key=lambda r:int(r['segment_id'].split('_')[1]))[:a.sessions_per_conversation])
    jobs={r['job_id']:r for r in jsonl(a.prepared/'source_sessions.jsonl')}
    a.output.mkdir(parents=True)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0')
    if a.checkpoint!='base':
        model=PeftModel.from_pretrained(model,a.checkpoint,is_trainable=False)
    model.requires_grad_(False).eval()
    results=[]
    for row in selected:
        inputs=tok.apply_chat_template(row['messages'][:2],tokenize=True,
            add_generation_prompt=True,return_tensors='pt',
            **({'enable_thinking':False} if a.disable_thinking else {})).to('cuda:0')
        with torch.no_grad():
            generated=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),
                max_new_tokens=a.max_new_tokens,do_sample=False,repetition_penalty=1.0,
                pad_token_id=tok.pad_token_id,use_cache=True)
        tokens=generated[0,inputs.shape[1]:]
        text=tok.decode(tokens,skip_special_tokens=True)
        eos=model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else eos
        finished=int(tokens[-1]) in eos
        record=dict(task_id=row['task_id'],context_id=row['context_id'],split='validation',
            input_sha256=row['input_sha256'],origin='self_generated',source_adapter=a.checkpoint,
            raw_memory_text=text,finish_reason='stop' if finished else 'length',tokens=len(tokens),
            source_text=row['messages'][1]['content'],target_memory=json.loads(row['messages'][2]['content']))
        try:
            if not finished:
                raise ValueError('length-limited generation')
            value=decode_v3(text,jobs[row['task_id']]['evidence_ids'])
            record.update(schema_valid=True,memory_text=compact(value),memory_count=len(value['memories']))
            gold=set(e for m in record['target_memory']['memories'] for e in m['evidence_ids'])
            predicted=set(e for m in value['memories'] for e in m['evidence_ids'])
            record['api_target_evidence_id_recall_diagnostic']=len(gold&predicted)/max(1,len(gold))
        except (ValueError,TypeError,KeyError) as exc:
            record.update(schema_valid=False,error=str(exc),memory_count=0)
        results.append(record)
        (a.output/'generations.jsonl').write_text(''.join(compact(r)+'\n' for r in results))
        print(compact({k:record[k] for k in ['task_id','schema_valid','tokens','memory_count']}),flush=True)
    report=dict(checkpoint=a.checkpoint,model=str(a.model.resolve()),sessions=len(results),
        thinking_disabled=a.disable_thinking,max_new_tokens=a.max_new_tokens,
        decoding=dict(do_sample=False,repetition_penalty=1.0),
        schema_valid=sum(r['schema_valid'] for r in results),
        schema_valid_rate=sum(r['schema_valid'] for r in results)/len(results),
        memory_count=sum(r['memory_count'] for r in results),
        selected_ids=[r['task_id'] for r in results],
        mean_output_tokens=sum(r['tokens'] for r in results)/len(results),
        input_sha256=[r['input_sha256'] for r in results],
        semantic_evaluated=False,ood_evaluated=False,previously_inspected_development_subset=True,
        runner_sha256=sha(Path(__file__).read_bytes()))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(compact(report),flush=True)


if __name__=='__main__':
    main()
