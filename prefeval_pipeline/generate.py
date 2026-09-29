"""Offline Qwen3-4B inference; resumable, with exact prompts and token counts."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    root = Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    source=root/'prepared/inputs.jsonl'
    rows=[json.loads(s) for s in source.read_text().splitlines()]
    if args.limit: rows=rows[:args.limit*2]
    model=str(root.parent/'models/Qwen3-4B')
    tokenizer=AutoTokenizer.from_pretrained(model)
    for r in rows:
        r['prompt']=tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        r['input_tokens']=len(tokenizer.encode(r['prompt'],add_special_tokens=False))
    assert max(r['input_tokens'] for r in rows)+300<=32768
    manifest={'model':model,'enable_thinking':False,'temperature':0,'max_tokens':300,'seed':42,
              'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rows':len(rows),
              'min_input_tokens':min(r['input_tokens'] for r in rows),'max_input_tokens':max(r['input_tokens'] for r in rows)}
    mp=out/'manifest.json'
    if mp.exists(): assert json.loads(mp.read_text())==manifest
    else: mp.write_text(json.dumps(manifest,indent=2))
    dest=out/'responses.jsonl'
    done={(r['id'],r['method']) for r in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
    pending=[r for r in rows if (r['id'],r['method']) not in done]
    if not pending: return
    llm=LLM(model=model,dtype='bfloat16',max_model_len=32768,gpu_memory_utilization=0.2,enforce_eager=True,
            max_num_seqs=64,seed=42,enable_prefix_caching=True)
    params=SamplingParams(temperature=0,max_tokens=300)
    with dest.open('a') as f:
        for start in range(0,len(pending),64):
            batch=pending[start:start+64]
            outputs=llm.generate([r['prompt'] for r in batch],params)
            for r,output in zip(batch,outputs):
                o=output.outputs[0]
                assert o.text.strip(), 'Empty response'
                record={k:v for k,v in r.items() if k not in ('messages','prompt')}
                record.update(response=o.text,output_tokens=len(o.token_ids),finish_reason=o.finish_reason,
                              prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest())
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
            f.flush()
            print(f'COMPLETE {len(done)+min(start+64,len(pending))}/{len(rows)}',flush=True)

if __name__=='__main__': main()
