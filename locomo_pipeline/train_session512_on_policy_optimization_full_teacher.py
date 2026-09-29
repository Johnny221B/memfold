"""Strict on-policy optimization variant: frozen memory-writer initialization teacher receives every text-memory record."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .component_alignment_trial import ROOT
from .compressor_data import save, sha
from . import train_session512_on_policy_optimization as base_on_policy_optimization


TEACHER_MAX_SEQUENCE = 49152


def full_teacher_prompt(tok, memories, question, device):
    """Render all records in source order, with no retrieval, pooling, or truncation."""
    import torch
    records = []
    for row in memories:
        records.extend(json.loads(row['memory_text'])['memories'])
    text = '\n'.join(json.dumps(m,ensure_ascii=False,separators=(',',':')) for m in records)
    memory_tokens = len(tok.encode(text,add_special_tokens=False))
    rendered = tok.apply_chat_template([
        {'role':'system','content':base_on_policy_optimization.SYSTEM},
        {'role':'user','content':'Memories:\n'+text+'\n\nQuestion: '+question}],
        tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if len(rendered)+64>TEACHER_MAX_SEQUENCE:
        raise ValueError('complete teacher text exceeds49152; no retrieval or truncation allowed')
    return torch.tensor([rendered],device=device),len(records),len(records),memory_tokens


def full_context_response_logps(model,prompt,responses,response_mask,soft=None):
    """Same answer-token scorer, allowing only the full-text teacher up to49152."""
    import torch
    batch,width=responses.shape
    p=prompt.expand(batch,-1);ids=torch.cat([p,responses],1)
    core=model.module if hasattr(model,'module') else model
    emb=core.get_input_embeddings()(ids)
    if soft is not None:emb=torch.cat([soft.expand(batch,-1,-1).to(emb.dtype),emb],1)
    limit=32768 if soft is not None else TEACHER_MAX_SEQUENCE
    if emb.shape[1]>limit:raise ValueError('on-policy optimization score context overflow; no truncation')
    logits=model(inputs_embeds=emb,attention_mask=torch.ones(emb.shape[:2],device=emb.device,dtype=torch.long),
        use_cache=False,logits_to_keep=width+1).logits[:,:-1]
    if logits.shape[:2]!=responses.shape:raise RuntimeError('unexpected answer-logit slice')
    values=logits.float().log_softmax(-1).gather(-1,responses.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(values[response_mask.bool()]).all():raise FloatingPointError('nonfinite sampled-token log probabilities')
    return values


def patch_base():
    base_on_policy_optimization.teacher_prompt=full_teacher_prompt
    base_on_policy_optimization.response_logps=full_context_response_logps


def prepare(reader_initialization,selection,out,reference_kl_weight=0.0,teacher=None):
    base_on_policy_optimization.prepare(reader_initialization,selection,out,reference_kl_weight,teacher)
    cfg=json.loads((out/'protocol.json').read_text())
    cfg.update(
        variant='strict_full_text_teacher_no_retrieval',
        teacher_memory='ALL question-blind memory-writer initialization-extracted native-v3 records in source order; no BM25, retrieval, pooling, truncation, or record dropping',
        teacher_text_max_sequence=TEACHER_MAX_SEQUENCE,
        teacher_model_declared_max_position_embeddings=40960,
        teacher_longest_preflight_prompt_tokens=44197,
        teacher_longest_with_response_tokens=44261,
        rope_note='default RoPE extrapolation only for full prompts above40960, at most7.9%; no rope scaling and no content selection',
        controlled_difference_from_bm25_run='teacher receives complete text memory; every other training hyperparameter, initialization, seed, dataset, student soft input, and loss is unchanged')
    cfg['sources'][str(Path(__file__).resolve())]=sha(Path(__file__).read_bytes())
    save(out/'protocol.json',cfg)


def worker(out,max_updates=0):
    patch_base();base_on_policy_optimization.worker(out,max_updates)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reader-initialization',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--max-updates',type=int,default=0)
    p.add_argument('--reference-kl-weight',type=float,default=0.0)
    p.add_argument('--teacher',type=Path,default=None)
    a=p.parse_args();reader_initialization=a.reader_initialization.resolve();selection=a.selection.resolve();out=a.output.resolve()
    if a.worker:worker(out,a.max_updates);return
    from .run_persona128k_alignment import gpu_free
    if out.exists():raise FileExistsError(out)
    if not gpu_free([0,1,2,3]):raise RuntimeError('GPU0-3 busy')
    prepare(reader_initialization,selection,out,a.reference_kl_weight,a.teacher)
    try:
        for phase,extra in [('smoke',['--max-updates','2']),('train',[])]:
            save(out/'status.json',dict(status='running',phase=phase,updated=time.time(),on_policy_optimization_launched=True,no_retrieval=True))
            with (out/(phase+'.log')).open('x') as log:
                rc=subprocess.call([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m',__spec__.name,
                    '--reader-initialization',str(reader_initialization),'--selection',str(selection),'--output',str(out),'--worker',*extra],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3','TOKENIZERS_PARALLELISM':'false','PYTHONPATH':str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH','')})
            if rc:raise RuntimeError(phase+' failed')
        save(out/'status.json',dict(status='completed',phase='training',updated=time.time(),on_policy_optimization_launched=True,no_retrieval=True,evaluation_pending=True))
    except Exception as exc:
        save(out/'status.json',dict(status='failed',error=repr(exc),updated=time.time(),on_policy_optimization_launched=True,no_retrieval=True));raise


if __name__=='__main__':main()
