"""Full-epoch question-blind native-v3 SFT; train backbone LoRA only.

Run standalone or with torchrun. Each training row is visited exactly once per
epoch; partial distributed batches use zero-weight dummy forwards, not duplicates.
Validation reports token-weighted teacher-forced CE against API pseudo-labels.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prepare import compact, jsonl, sha
from .training_scopes import check_trainable_scope
from .persona128k_recipe import RECIPES, lora_options


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--max-sequence', type=int, default=32768)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--recipe', choices=RECIPES, default='legacy')
    p.add_argument('--allow-partial', action='store_true', help='test only')
    p.add_argument('--disable-thinking', action='store_true', help='use matching non-thinking chat prefix for Qwen3')
    a = p.parse_args()
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    exists = [a.output.exists() if rank==0 else None]
    if world>1:
        dist.broadcast_object_list(exists,src=0,device=device)
    if exists[0]:
        raise FileExistsError('refusing to overwrite training output')
    manifest = json.loads((a.prepared/'manifest.json').read_text())
    if not manifest['complete_data'] and not a.allow_partial:
        raise ValueError('full data required')
    if a.epochs < 1:
        raise ValueError('epochs must be positive')
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tok.pad_token = tok.eos_token
    rows = {s:jsonl(a.prepared/s/'writer_sft.jsonl') for s in ['train','validation']}
    if not all(rows.values()):
        raise ValueError('empty split')
    if {r['context_id'] for r in rows['train']} & {r['context_id'] for r in rows['validation']}:
        raise ValueError('conversation split overlap')
    encoded = {}
    for split, rr in rows.items():
        encoded[split] = []
        for row in rr:
            if row['split'] != split or len(row['messages']) != 3:
                raise ValueError('invalid SFT row')
            prefix = tok.apply_chat_template(row['messages'][:2], tokenize=True, add_generation_prompt=True,
                **({'enable_thinking':False} if a.disable_thinking else {}))
            target = tok(row['messages'][2]['content']+tok.eos_token, add_special_tokens=False).input_ids
            if len(prefix)+len(target) > a.max_sequence:
                raise ValueError(f'{row["id"]} exceeds sequence limit; refusing truncation')
            encoded[split].append((prefix, target))
    if rank == 0:
        a.output.mkdir(parents=True)
    if world > 1:
        dist.barrier()
    report = dict(status='running', task='memory_writer_initialization_full_epoch_sft',
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        model=str(a.model.resolve()), world_size=world,
        train_rows=len(rows['train']), validation_rows=len(rows['validation']),
        train_contexts=sorted({r['context_id'] for r in rows['train']}),
        validation_contexts=sorted({r['context_id'] for r in rows['validation']}),
        prepared_manifest_sha256=sha((a.prepared/'manifest.json').read_bytes()),
        runner_sha256=sha(Path(__file__).read_bytes()),
        max_observed_sequence=max(len(x)+len(y) for rr in encoded.values() for x,y in rr),
        lora=(lora_options(a.recipe) if a.recipe=='persona128k' else
              dict(rank=8,alpha=16,target_modules=['q_proj','v_proj'],dropout=0)),
        qa_in_writer_prompt=False, semantic_quality_verified=False, events=[])
    start = time.time()

    def emit(event, **values):
        if rank != 0:
            return
        record = dict(event=event,elapsed_seconds=time.time()-start,**values)
        report['events'].append(record)
        temp = a.output/'report.tmp'
        temp.write_text(json.dumps(report,indent=2)+'\n')
        temp.replace(a.output/'report.json')
        print(compact(record),flush=True)

    emit('data_ready', **{k:report[k] for k in ['train_rows','validation_rows','max_observed_sequence']})
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
    model = get_peft_model(model,LoraConfig(**lora_options(a.recipe),task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    active = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    versions = [p._version for p in frozen]
    optimizer = torch.optim.AdamW(active,lr=a.learning_rate,weight_decay=0)
    roles = {'backbone_base':[], 'backbone_lora':[]}
    for name, par in model.named_parameters():
        roles['backbone_lora' if 'lora_' in name else 'backbone_base'].append((name,par))
    emit('scope', **check_trainable_scope('memory_writer_initialization',roles,optimizer))
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if world>1 else model
    versions = [p._version for p in frozen]

    def nll(engine, example):
        pre, target = example
        tokens = torch.tensor([pre+target],device=device)
        gold = torch.tensor([target],device=device)
        logits = engine(input_ids=tokens,use_cache=False,logits_to_keep=len(target)+1).logits[:,:-1]
        sums = [F.cross_entropy(block.float().reshape(-1,block.shape[-1]), ids.reshape(-1),reduction='sum')
                for block,ids in zip(logits.split(128,1),gold.split(128,1))]
        return torch.stack(sums).sum(),len(target)

    def validate(epoch):
        model.eval()
        totals = torch.zeros(2,device=device,dtype=torch.float64)
        with torch.no_grad():
            for example in encoded['validation'][rank::world]:
                loss,count = nll(model,example)
                totals += torch.tensor([float(loss),count],device=device,dtype=torch.float64)
        if world>1:
            dist.all_reduce(totals)
        value = float(totals[0]/totals[1])
        if not math.isfinite(value):
            raise ValueError('nonfinite validation loss')
        emit('validation',epoch=epoch,token_ce=value,target_tokens=int(totals[1]))
        return value

    try:
        validate(0)
        steps_per_epoch = math.ceil(len(encoded['train'])/world)
        total_steps = a.epochs*steps_per_epoch
        step = 0
        best = float('inf')
        for epoch in range(1,a.epochs+1):
            model.train()
            order = list(range(len(encoded['train'])))
            random.Random(a.seed+epoch).shuffle(order)
            totals = torch.zeros(3,device=device,dtype=torch.float64)
            before = [p.detach().clone() for p in active]
            for offset in range(0,len(order),world):
                batch = order[offset:offset+world]
                present = rank<len(batch)
                index = batch[rank] if present else order[0]
                optimizer.zero_grad(set_to_none=True)
                loss,count = nll(wrapped,encoded['train'][index])
                scaled = loss/count * (world/len(batch) if present else 0.0)
                scaled.backward()
                grad = torch.nn.utils.clip_grad_norm_(active,1.0)
                if not torch.isfinite(grad) or not torch.isfinite(loss):
                    raise ValueError('nonfinite gradient/loss')
                # Linear warmup then cosine decay, recorded at every log point.
                warmup = max(1,math.ceil(total_steps*0.05))
                factor = (step+1)/warmup if step<warmup else 0.5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
                lr = a.learning_rate*factor
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.step()
                if present:
                    totals += torch.tensor([float(loss.detach()),count,1],device=device,dtype=torch.float64)
                step += 1
                if step%10==0 or offset+world>=len(order):
                    emit('train_step',epoch=epoch,step=step,total_steps=total_steps,
                         local_sequence_ce=float(loss.detach()/count),grad_norm=float(grad),learning_rate=lr)
            if versions != [p._version for p in frozen] or any(p.grad is not None for p in frozen):
                raise ValueError('frozen backbone changed or received gradients')
            changed = sum(not torch.equal(x,p) for x,p in zip(before,active))
            del before
            if changed==0:
                raise ValueError('LoRA did not update')
            if world>1:
                dist.all_reduce(totals)
            if int(totals[2])!=len(order):
                raise ValueError('epoch did not visit every row exactly once')
            emit('epoch_train',epoch=epoch,rows=int(totals[2]),token_ce=float(totals[0]/totals[1]),
                 changed_lora_tensors=changed,frozen_unchanged=True)
            val = validate(epoch)
            if rank==0:
                checkpoint = a.output/f'epoch_{epoch}'
                model.save_pretrained(checkpoint)
                torch.save(dict(optimizer=optimizer.state_dict(),epoch=epoch,step=step),checkpoint/'optimizer.pt')
                tok.save_pretrained(checkpoint)
                if val<best:
                    best=val
                    report['best_epoch']=epoch
                emit('checkpoint',epoch=epoch,path=str(checkpoint.resolve()),validation_token_ce=val)
            if world>1:
                dist.barrier()
        report['status']='completed'
        emit('completed',epochs=a.epochs,optimizer_steps=step)
    except Exception as exc:
        report['status']='failed'
        emit('failure',error=repr(exc))
        raise
    finally:
        if world>1:
            dist.destroy_process_group()


if __name__=='__main__':
    main()
