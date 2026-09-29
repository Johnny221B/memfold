"""PrefEval generation for downloaded RQ2 Qwen3-4B methods. No training or judge."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parent
METHODS=('vanilla-grpo','opsd','autocompressor','memgen')

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_jsonl(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def inputs(source,limit):
    rows=[r for r in read_jsonl(source) if r['method']=='full_text']
    if len(rows)!=1000 or len({r['id'] for r in rows})!=1000:raise ValueError('Expected 1000 unique full-history HF records')
    for row in rows:
        if row['messages'][0]!={'role':'system','content':'You are a helpful assistant.'}:raise ValueError('Unexpected system prompt')
        if row['messages'][-1]['role']!='user':raise ValueError('Missing final query')
    return rows[:limit] if limit else rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=METHODS,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,default=ROOT/'prepared/inputs.jsonl')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--max-new-tokens',type=int,default=300)
    parser.add_argument('--segment-length',type=int,default=1536)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.limit<0 or args.max_new_tokens<=0 or args.segment_length<=0:raise ValueError('Invalid limits')
    rows=inputs(args.inputs,args.limit)
    from checkpoint_backends import checkpoint,summary_length,BASE
    path=checkpoint(args.method)
    if not path.exists():raise FileNotFoundError(path)
    required=['config.json','pytorch_model.bin'] if args.method=='memgen' else ['adapter_config.json','adapter_model.safetensors']
    ckhash={name:digest(path/name) for name in required}
    config=json.loads((path/required[0]).read_text())
    if args.method!='memgen' and 'qwen3-4b' not in config['base_model_name_or_path'].lower():raise ValueError('Wrong adapter backbone')
    manifest={'method':args.method,'backbone':'Qwen3-4B','base_model':str(BASE),'checkpoint':str(path),
              'checkpoint_sha256':ckhash,'hf_revision':json.loads((ROOT/'checkpoints/hf_inventory.json').read_text())['revision'],
              'input_sha256':digest(args.inputs),'records':len(rows),'max_new_tokens':args.max_new_tokens,
              'temperature':0,'enable_thinking':False,'seed':42,'segment_length':args.segment_length if args.method=='autocompressor' else None,
              'code_sha256':{p.name:digest(p) for p in [Path(__file__),ROOT/'checkpoint_backends.py']},
              'output_token_count':'len(actual generated IDs), includes emitted EOS','device':args.device}
    if args.method=='autocompressor':manifest['summary_length']=summary_length(path)
    if args.method=='memgen':
        manifest['upstream_commit']='970cc95af99b5008610e6b281619d181bc9b5ab9'
        manifest['upstream_sources']={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'vendor/MemGen/memgen').rglob('*.py')}
    if args.method=='autocompressor':
        src=ROOT.parent/'src/memory_opd/baselines/autocompressor'
        manifest['autocompressor_sources']={str(p):digest(p) for p in src.glob('*.py')}
    manifest['base_tokenizer_sha256']={p.name:digest(p) for p in BASE.glob('*.json')}
    import importlib.metadata
    manifest['environment']={k:importlib.metadata.version(k) for k in ['torch','transformers','peft','numpy']}

    args.output.mkdir(parents=True,exist_ok=True)
    if args.dry_run:
        (args.output/'preflight.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2));return
    dest=args.output/'responses.jsonl';mp=args.output/'manifest.json'
    if mp.exists():
        if json.loads(mp.read_text())!=manifest:raise ValueError('Run manifest changed; choose a new output directory')
    else:
        if dest.exists():raise ValueError('Responses without manifest')
        mp.write_text(json.dumps(manifest,indent=2))
    done=read_jsonl(dest) if dest.exists() else []
    ids={r['id'] for r in done}
    if len(ids)!=len(done) or not ids.issubset({r['id'] for r in rows}):raise ValueError('Duplicate/foreign resume IDs')
    if any(r['method']!=args.method for r in done):raise ValueError('Wrong resume method')
    pending=[r for r in rows if r['id'] not in ids]
    if not pending:return
    import torch
    from checkpoint_backends import load,answer_reader,answer_compressed,answer_memgen
    torch.manual_seed(42);random.seed(42);torch.cuda.set_device(torch.device(args.device))
    model,tok,info=load(args.method,args.device)
    (args.output/'loaded_model.json').write_text(json.dumps(info,indent=2))
    with dest.open('a') as f:
        for r in pending:
            started=time.monotonic()
            messages=r['messages']
            context_ids=[]
            if args.method=='autocompressor':
                # Same role-tagged context rendering as the trained PersonaMem compressor, no question in encoder.
                context='\n\n'.join(f'[{m["role"].upper()}]\n{m["content"]}' for m in messages[1:-1])
                context_ids=tok.encode(context,add_special_tokens=False)
                messages=[messages[0],messages[-1]]
            prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            token_ids=tok.encode(prompt,add_special_tokens=False)
            if len(token_ids)+args.max_new_tokens+256>32768:raise ValueError('Reader overflow; no silent truncation')
            tensor=torch.tensor([token_ids],dtype=torch.long,device=args.device);mask=torch.ones_like(tensor)
            detail={}
            if args.method in ('vanilla-grpo','opsd'):
                generated=answer_reader(model,tensor,mask,args.max_new_tokens,tok)
            elif args.method=='autocompressor':
                generated,detail=answer_compressed(model,context_ids,tensor,args.max_new_tokens,tok,args.segment_length)
            else:
                generated,detail=answer_memgen(model,tensor,mask,args.max_new_tokens,tok)
            usage={k:0 for k in ['builder_input','builder_output','document_or_session_embed_input','query_embed_input','compressor_text_input','compressor_effective_positions','weaver_positions','reasoner_positions','soft_input_positions']}
            usage.update(reader_prompt=len(token_ids),reader_output=len(generated))
            usage['reader_only_total']=len(token_ids)+len(generated)
            if args.method=='autocompressor':
                usage['compressor_effective_positions']=detail['compressor_effective_positions']
                usage['soft_input_positions']=detail['soft_input_positions']
                usage['reader_only_total']+=detail['soft_input_positions']
            if args.method=='memgen':
                if detail['reasoner_processed_positions']<=0 or detail['weaver_processed_positions']<=0:raise ValueError('Missing MemGen position hooks')
                # Cached decoding feeds N-1 output IDs; avoid counting those twice with reader_output.
                usage['reasoner_positions']=detail['reasoner_processed_positions']-max(len(generated)-1,0)
                usage['weaver_positions']=detail['weaver_processed_positions']
                usage['reader_only_total']=usage['reasoner_positions']+len(generated)
            usage['strict_end_to_end_total']=usage['reader_only_total']+usage['compressor_effective_positions']+usage['weaver_positions']
            record={'id':r['id'],'method':args.method,'topic':r['topic'],'question':r['question'],'preference':r['preference'],
                    'response':tok.decode(generated,skip_special_tokens=True),'output_token_ids':generated,
                    'input_tokens':len(token_ids),'output_tokens':len(generated),'finish_reason':'stop' if generated and generated[-1]==tok.eos_token_id else 'length',
                    'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'token_usage':usage,'component_details':detail,
                    'seconds':time.monotonic()-started,'context_tokens':len(context_ids)}
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();ids.add(r['id'])
            print(f'{args.method}: {len(ids)}/{len(rows)} id={r["id"]} tokens={len(generated)}',flush=True)
    all_rows=read_jsonl(dest)
    summary={'method':args.method,'records':len(all_rows),'complete':len(all_rows)==len(rows),
             'mean_reader_only_tokens':sum(r['token_usage']['reader_only_total'] for r in all_rows)/len(all_rows),
             'mean_strict_end_to_end_tokens':sum(r['token_usage']['strict_end_to_end_total'] for r in all_rows)/len(all_rows),
             'empty_answers':sum(not r['response'].strip() for r in all_rows),'responses_sha256':digest(dest)}
    (args.output/'generation_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if __name__=='__main__':main()
