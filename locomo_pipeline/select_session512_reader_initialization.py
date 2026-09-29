"""Select epochs1-3 only by held-out LoCoMo validation answer NLL."""
import argparse,json,os
from pathlib import Path
import subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from .component_alignment_trial import ROOT,append
from .compressor_data import read,save,sha
from .train_session512_reader_initialization import answer_nll,session_prefix


def prepare(run,out):
    cfg=json.loads((run/'protocol.json').read_text())
    memories=ROOT/'locomo_pipeline/runs/persona128k_gpu0123_v1/self_memory_retry_v1/prepared/validation.jsonl'
    questions=ROOT/'locomo_pipeline/prepared/full_v3_20260906/validation/qa.jsonl'
    session_data=Path(cfg.get('session_data',ROOT/'locomo_pipeline/runs/persona128k_gpu0123_v1/shallow_capacity_v1/data.json'))
    rows=read(memories);qa=read(questions);all_data=json.loads(session_data.read_text())
    bank={}
    for row in rows:
        assert row['split']=='validation' and row['memory_text']==all_data['sessions'][row['id']]['memory_text']
        bank.setdefault(row['context_id'],[]).append(row)
    assert set(bank)=={'conv-49','conv-50'} and len(qa)==293
    for q in qa:
        assert q['split']=='validation' and set(q['source_session_ids'])=={r['id'] for r in bank[q['context_id']]}
    data=dict(memories=rows,questions=qa,sessions={r['id']:all_data['sessions'][r['id']] for r in rows})
    checkpoints={str(e):dict(reader=str(run/f'train/epoch_{e}/adapter'),compressor=str(run/f'train/epoch_{e}/compressor.pt')) for e in [1,2,3]}
    sources=[Path(__file__),ROOT/'locomo_pipeline/train_session512_reader_initialization.py',run/'protocol.json',run/'continuation_protocol.json',memories,questions,
        session_data,Path(cfg['mapper']),Path(cfg['cache'])/'manifest.json']
    for ck in checkpoints.values():sources += [Path(ck['reader'])/'adapter_model.safetensors',Path(ck['compressor'])]
    out.mkdir(parents=True);save(out/'data.json',data)
    save(out/'protocol.json',dict(model=cfg['model'],mapper=cfg['mapper'],cache=cfg['cache'],checkpoints=checkpoints,sources={str(p):sha(p.read_bytes()) for p in sources},
        data_hash=sha((out/'data.json').read_bytes()),questions=293,contexts=['conv-49','conv-50'],output_tokens_per_session=512,
        encoder_layer_key=cfg.get('encoder_layer_key','first'),encoder_layer_index=cfg.get('encoder_layer_index',1),
        metric='mean gold answer token NLL over all held-out LoCoMo validation QA; lowest wins',selection_tiebreak='lower epoch number',
        longmemeval_used_for_selection=False,question_generation=False,promotion_allowed=False,on_policy_optimization_launched=False))


def worker(out,epoch):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from .session_memory_trial import build_compressor
    from .train_compressor import prompt_and_target
    cfg=json.loads((out/'protocol.json').read_text());data=json.loads((out/'data.json').read_text())
    for p,digest in cfg['sources'].items():assert sha(Path(p).read_bytes())==digest
    assert sha((out/'data.json').read_bytes())==cfg['data_hash']
    torch.set_num_threads(4);torch.manual_seed(42);device=torch.device('cuda:0')
    tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval().requires_grad_(False);h=base.config.hidden_size
    mapper=torch.nn.Sequential(torch.nn.LayerNorm(h),torch.nn.Linear(h,1024),torch.nn.GELU(),torch.nn.Linear(1024,h)).to(device)
    payload=torch.load(cfg['mapper'],map_location='cpu',weights_only=True);assert payload['encoder_layer']==cfg['encoder_layer_key'];mapper.load_state_dict(payload['state_dict'],strict=True);mapper.eval().requires_grad_(False)
    # The shallow-capacity cache intentionally contains every training session
    # but only a fixed validation probe panel.  Validation checkpoint selection
    # needs all held-out sessions, so reproduce the exact frozen first-block
    # representation for entries that are not present in that diagnostic cache.
    manifest=json.loads((Path(cfg['cache'])/'manifest.json').read_text());banks={};cache_hits=0;live_encodes=0
    with torch.no_grad():
        for row in data['memories']:
            token_ids=tok.encode(row['memory_text'],add_special_tokens=False)
            assert len(token_ids)==data['sessions'][row['id']]['tokens'] and len(token_ids)<=4096
            if row['id'] in manifest:
                path=Path(cfg['cache'])/manifest[row['id']];item=torch.load(path,map_location='cpu',weights_only=True)
                assert item['session']==row['id'] and item['encoder_layer']==cfg['encoder_layer_key'] and item['states'].shape[1]==len(token_ids)
                assert token_ids==item['ids'][0].tolist();states=item['states'].to(device).float();cache_hits+=1
            else:
                ids=torch.tensor([token_ids],device=device)
                encoded=base.model(input_ids=ids,use_cache=False,output_hidden_states=True)
                states=encoded.hidden_states[cfg['encoder_layer_index']].float();live_encodes+=1
            banks.setdefault(row['context_id'],[]).append(mapper(states))
    print(json.dumps(dict(epoch=epoch,cache_hits=cache_hits,live_encodes=live_encodes,encoder_layer=cfg['encoder_layer_key'])),flush=True)
    ck=cfg['checkpoints'][str(epoch)];payload=torch.load(ck['compressor'],map_location='cpu',weights_only=True)
    assert payload['granularity']=='session_json' and payload['output_tokens_per_session']==512 and not payload['smoke_only']
    compressor=build_compressor(h).to(device);compressor.load_state_dict(payload['state_dict'],strict=True);compressor.eval().requires_grad_(False)
    with torch.no_grad():softs={c:session_prefix(compressor,x) for c,x in banks.items()}
    policy=PeftModel.from_pretrained(base,ck['reader'],is_trainable=False).eval().requires_grad_(False)
    totals={c:[0.,0,0] for c in banks};rows=[]
    with torch.no_grad():
        for i,q in enumerate(data['questions']):
            pp,yy=prompt_and_target(tok,'Answer the question concisely and completely.\nQuestion: '+q['question'],q['answer'],device,512)
            loss=float(answer_nll(policy,pp,yy,softs[q['context_id']]))
            n=yy.numel();totals[q['context_id']][0]+=loss*n;totals[q['context_id']][1]+=n;totals[q['context_id']][2]+=1
            rows.append(dict(epoch=epoch,question_id=q['question_id'],context=q['context_id'],answer_tokens=n,nll=loss))
            if (i+1)%25==0:print(json.dumps(dict(epoch=epoch,done=i+1,total=len(data['questions']))),flush=True)
    for row in rows:append(out/f'epoch_{epoch}.jsonl',row)
    total_loss=sum(v[0] for v in totals.values());tokens=sum(v[1] for v in totals.values())
    result=dict(epoch=epoch,questions=len(rows),answer_tokens=tokens,mean_token_nll=total_loss/tokens,
        by_context={c:dict(questions=v[2],answer_tokens=v[1],mean_token_nll=v[0]/v[1]) for c,v in totals.items()},
        cache_hits=cache_hits,live_encodes=live_encodes,encoder_layer=cfg['encoder_layer_key'])
    save(out/f'epoch_{epoch}.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',type=int,choices=[1,2,3]);p.add_argument('--wait',action='store_true');a=p.parse_args();run=a.run.resolve();out=a.output.resolve()
    if a.worker:worker(out,a.worker);return
    if out.exists():raise FileExistsError(out)
    if a.wait:
        deadline=time.time()+3*3600
        while not (run/'continuation_result.json').exists():
            if json.loads((run/'status.json').read_text())['status']=='failed':raise RuntimeError('training failed')
            if time.time()>deadline:raise TimeoutError('training deadline')
            time.sleep(15)
    from .run_persona128k_alignment import gpu_free
    if not gpu_free([0,1,2]):raise RuntimeError('GPU0-2 busy')
    prepare(run,out)
    def launch(epoch,gpu):
        with (out/f'epoch_{epoch}.log').open('x') as log:
            rc=subprocess.call([sys.executable,'-u','-m',__spec__.name,'--run',str(run),'--output',str(out),'--worker',str(epoch)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'TOKENIZERS_PARALLELISM':'false'})
        if rc:raise RuntimeError(f'epoch{epoch} validation failed')
    try:
        save(out/'status.json',dict(status='running',updated=time.time()))
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(launch,e,e-1) for e in [1,2,3]]
            for job in jobs:job.result()
        results=[json.loads((out/f'epoch_{e}.json').read_text()) for e in [1,2,3]];best=min(results,key=lambda r:(r['mean_token_nll'],r['epoch']))
        save(out/'selection.json',dict(status='completed',metric='mean_token_nll',results=results,selected_epoch=best['epoch'],selected_checkpoint=json.loads((out/'protocol.json').read_text())['checkpoints'][str(best['epoch'])],longmemeval_used=False,on_policy_optimization_launched=False))
        save(out/'status.json',dict(status='completed',selected_epoch=best['epoch'],updated=time.time()))
    except Exception as exc:save(out/'status.json',dict(status='failed',error=repr(exc),updated=time.time()));raise


if __name__=='__main__':main()
