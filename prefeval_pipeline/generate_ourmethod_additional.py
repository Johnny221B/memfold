"""RQ2 ourmethod: current adapter writer -> frozen base encoder -> K256 bridge -> adapter reader."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from peft import PeftModel

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from memory_opd.soft_reconstruction import ContextResampler,ContextToSoftTokens,SoftTokenProjector
from memory_opd.rq2_baselines.personamem import render_context
from memory_opd.compressed_opd import serialize_memory
from build_personamem_writer_inputs import SYSTEM as WRITER_SYSTEM
from generate_personamem_self_memory import MEMORY_BUDGET,tolerant_memory
from generate_checkpoints import inputs,digest,read_jsonl

from backbone_config import BASE,BACKBONE,NAMES
CHECKPOINT=ROOT/'checkpoints/ourmethod'/BACKBONE
READER_SYSTEM='Continuous soft-memory tokens precede this conversation. Use only that memory and the visible question. Answer the question in natural language.'

def writer_messages(row):
    return [{'role':'system','content':WRITER_SYSTEM+MEMORY_BUDGET},
            {'role':'user','content':'HISTORY:\n'+render_context(row['messages'][1:-1])+'\n\nCURRENT QUESTION:\n'+row['question']}]

def render(tok,messages):
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return text,tok.encode(text,add_special_tokens=False)

def load(device):
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    policy=PeftModel.from_pretrained(base,CHECKPOINT/'adapter',is_trainable=False).to(device).eval().requires_grad_(False)
    saved=torch.load(CHECKPOINT/'bridge.pt',map_location='cpu',weights_only=True);cfg=saved['config']
    if cfg['lm_dim']!=policy.config.hidden_size or cfg['context_dim']!=policy.config.hidden_size or cfg['token_count']!=256:
        raise ValueError('Wrong backbone or soft-token count in bridge')
    bridge=ContextToSoftTokens(ContextResampler(cfg['context_dim'],latent_dim=cfg['latent_dim'],token_count=cfg['token_count'],
        layers=cfg['layers'],heads=cfg['heads'],context_residual=cfg.get('context_residual',False)),SoftTokenProjector(cfg['latent_dim'],cfg['lm_dim']))
    bridge.load_state_dict(saved['bridge'],strict=True)
    bridge.to(device=device,dtype=torch.bfloat16).eval().requires_grad_(False)
    return policy,tok,bridge,cfg

@torch.inference_mode()
def encode_memory(policy,tok,bridge,text,device):
    ids=tok(text,add_special_tokens=True).input_ids
    pooled=[]
    # Identical frozen-base path to cache_personamem_text_memory_states.py, no writer LoRA here.
    with policy.disable_adapter():
        decoder=policy.get_base_model().model
        for start in range(0,len(ids),2048):
            chunk=ids[start:start+2048];tensor=torch.tensor([chunk],device=device)
            states=decoder(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False,return_dict=True).last_hidden_state[0]
            pooled.extend(states[j:min(len(chunk),j+32)].mean(0) for j in range(0,len(chunk),32))
    states=torch.stack(pooled).to(torch.float16).to(torch.bfloat16).unsqueeze(0)
    soft=bridge(states,torch.ones(states.shape[:2],device=device,dtype=torch.long))
    if soft.shape!=(1,256,policy.config.hidden_size) or not torch.isfinite(soft).all():raise ValueError('Invalid soft memory')
    return soft,len(ids),len(pooled)

@torch.inference_mode()
def generate(policy,tok,ids,max_tokens,soft=None):
    tensor=torch.tensor([ids],device=policy.device,dtype=torch.long)
    gen=GenerationConfig(max_new_tokens=max_tokens,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
    kwargs={'generation_config':gen,'do_sample':False,'temperature':None,'top_p':None,'top_k':None}
    if soft is None:
        output=policy.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),**kwargs)
        return output[0,len(ids):].tolist()
    prefix=torch.cat([soft,policy.get_input_embeddings()(tensor)],dim=1)
    output=policy.generate(inputs_embeds=prefix,attention_mask=torch.ones(prefix.shape[:2],device=prefix.device,dtype=torch.long),**kwargs)
    result=output[0].tolist()
    if len(result)>max_tokens:raise ValueError('Unexpected inputs_embeds generation output layout')
    return result

def main():
    global BASE, CHECKPOINT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--max-new-tokens',type=int,default=300)
    p.add_argument('--max-memory-tokens',type=int,default=2048);p.add_argument('--device',default='cuda:0')
    p.add_argument('--model',type=Path,default=BASE)
    p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    p.add_argument('--inputs',type=Path,default=ROOT/'prepared/inputs.jsonl')
    p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    BASE=args.model;CHECKPOINT=args.checkpoint
    if min(args.max_new_tokens,args.max_memory_tokens)<=0 or args.limit<0:raise ValueError('Invalid generation limits')
    source=args.inputs;rows=inputs(source,args.limit)
    ah=digest(CHECKPOINT/'adapter/adapter_model.safetensors');bh=digest(CHECKPOINT/'bridge.pt')
    files=[Path(__file__),ROOT/'backbone_config.py',ROOT/'generate_checkpoints.py',REPO/'scripts/build_personamem_writer_inputs.py',
           REPO/'scripts/generate_personamem_self_memory.py',REPO/'src/memory_opd/soft_reconstruction.py',
           REPO/'src/memory_opd/compressed_opd.py',REPO/'src/memory_opd/rq2_baselines/personamem.py']
    manifest={'method':'ourmethod','backbone':NAMES[BACKBONE],'checkpoint':str(CHECKPOINT),'adapter_sha256':ah,'bridge_sha256':bh,
              'input_sha256':digest(source),'records':len(rows),'max_new_tokens':args.max_new_tokens,
              'max_memory_tokens':args.max_memory_tokens,'seed':42,'enable_thinking':False,'temperature':0,
              'encoder':f'frozen base {NAMES[BACKBONE]}, LoRA disabled','encoder_chunk_tokens':2048,'pool_tokens':32,'K':256,
              'reader_system':READER_SYSTEM,'code_sha256':{str(f):digest(f) for f in files},
              'self_memory_origin':'same current checkpoint as reader; no API memory','output_ids_include_emitted_eos':True}
    args.output.mkdir(parents=True,exist_ok=True)
    if args.dry_run:(args.output/'preflight.json').write_text(json.dumps(manifest,indent=2));return
    mp=args.output/'manifest.json';dest=args.output/'responses.jsonl'
    if mp.exists():
        if json.loads(mp.read_text())!=manifest:raise ValueError('Manifest mismatch; choose a fresh output directory')
    else:
        if dest.exists():raise ValueError('Responses without manifest')
        mp.write_text(json.dumps(manifest,indent=2))
    old=read_jsonl(dest) if dest.exists() else [];done={r['id'] for r in old}
    if len(old)!=len(done) or not done.issubset({r['id'] for r in rows}):raise ValueError('Invalid resume IDs')
    pending=[r for r in rows if r['id'] not in done]
    if not pending:return
    torch.manual_seed(42);torch.cuda.set_device(torch.device(args.device))
    policy,tok,bridge,cfg=load(args.device)
    (args.output/'bridge_config.json').write_text(json.dumps(cfg,indent=2))
    memories=args.output/'self_memories';memories.mkdir(exist_ok=True)
    with dest.open('a') as f:
        for r in pending:
            start=time.monotonic();wp,wids=render(tok,writer_messages(r))
            if len(wids)+args.max_memory_tokens>32768:raise ValueError('Writer input overflow')
            mh=hashlib.sha256(wp.encode()).hexdigest();cache=memories/f'{r["id"]}.json'
            if cache.exists():
                memory=json.loads(cache.read_text())
                if memory['writer_prompt_sha256']!=mh or memory['adapter_sha256']!=ah:raise ValueError('Stale self memory')
            else:
                mids=generate(policy,tok,wids,args.max_memory_tokens)
                raw=tok.decode(mids,skip_special_tokens=True);canonical,valid=tolerant_memory(raw)
                memory={'id':r['id'],'writer_prompt_sha256':mh,'adapter_sha256':ah,'raw_text':raw,
                        'memory_text':serialize_memory(canonical),'schema_valid':valid,'output_token_ids':mids,
                        'origin':'current_checkpoint_self_generated','writer_input_tokens':len(wids)}
                tmp=cache.with_suffix('.tmp');tmp.write_text(json.dumps(memory,ensure_ascii=False,indent=2));tmp.replace(cache)
            soft,encoded,pooled=encode_memory(policy,tok,bridge,memory['memory_text'],args.device)
            messages=[{'role':'system','content':READER_SYSTEM},{'role':'user','content':r['messages'][-1]['content']}]
            rp,rids=render(tok,messages)
            if len(rids)+256+args.max_new_tokens>32768:raise ValueError('Reader overflow')
            generated=generate(policy,tok,rids,args.max_new_tokens,soft)
            usage={k:0 for k in ['document_or_session_embed_input','query_embed_input','compressor_effective_positions','weaver_positions','reasoner_positions']}
            usage.update(builder_input=len(wids),builder_output=len(memory['output_token_ids']),compressor_text_input=encoded,
                         soft_input_positions=256,reader_prompt=len(rids),reader_output=len(generated),
                         reader_only_total=len(rids)+256+len(generated))
            usage['strict_end_to_end_total']=usage['builder_input']+usage['builder_output']+encoded+usage['reader_only_total']
            record={'id':r['id'],'method':'ourmethod','topic':r['topic'],'question':r['question'],'preference':r['preference'],
                    'response':tok.decode(generated,skip_special_tokens=True),'output_token_ids':generated,
                    'input_tokens':len(rids)+256,'output_tokens':len(generated),'finish_reason':'stop' if generated and generated[-1]==tok.eos_token_id else 'length',
                    'prompt_sha256':hashlib.sha256(rp.encode()).hexdigest(),'writer_prompt_sha256':mh,
                    'self_memory_sha256':hashlib.sha256(memory['memory_text'].encode()).hexdigest(),'self_memory_file':str(cache),
                    'memory_schema_valid':memory['schema_valid'],'memory_pooled_vectors':pooled,'soft_shape':list(soft.shape),
                    'token_usage':usage,'seconds':time.monotonic()-start}
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();done.add(r['id'])
            print(f'ourmethod: {len(done)}/{len(rows)} id={r["id"]} writer={len(memory["output_token_ids"])} reader={len(generated)} valid_memory={memory["schema_valid"]}',flush=True)
    all_rows=read_jsonl(dest)
    summary={'method':'ourmethod','records':len(all_rows),'complete':len(all_rows)==len(rows),
             'mean_reader_only_tokens':sum(r['token_usage']['reader_only_total'] for r in all_rows)/len(all_rows),
             'mean_strict_end_to_end_tokens':sum(r['token_usage']['strict_end_to_end_total'] for r in all_rows)/len(all_rows),
             'invalid_memory_schema':sum(not r['memory_schema_valid'] for r in all_rows)}
    (args.output/'generation_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if __name__=='__main__':main()
