"""Joint reader initialization DDP pilot: memory-writer initialization LoRA + last bridge layer/norm/projector.

Answer-only CE + 0.1 relative soft-output anchor. Frozen base and encoder.
Explicit LoCoMo API-pretrain -> same-session self-memory provenance transfer.
"""
import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .compressor_core import load, configure_scope, audit_scope, target_nll, soft_anchor_loss
from .compressor_data import read, save, sha, memory_bank, qa_rows, source_identity
from .train_compressor import fingerprint, state_bank, prompt_and_target
from .persona128k_recipe import RECIPES, check_adapter, assert_fp32_optimizer

ROOT=Path(__file__).resolve().parent.parent


def transfer_gate(parent, current_fingerprint, bank, writer):
    if parent.get('procedure')!='auxiliary_reasoning_adaptation' or parent.get('smoke_only') or parent.get('external_initialization'):
        raise ValueError('requires completed native LoCoMo reasoning pretraining')
    if not parent.get('backbone_lora_trainable') or parent.get('memory_origin')!='api_teacher':
        raise ValueError('unexpected pretraining recipe/source')
    old=parent['cache_fingerprint']
    if {k:v for k,v in old.items() if k!='memories_sha256'}!={k:v for k,v in current_fingerprint.items() if k!='memories_sha256'}:
        raise ValueError('encoder/model/cache settings changed')
    if set(parent['training_contexts'])!=set(bank): raise ValueError('training conversations changed')
    if any(Path(source_identity(r)).resolve()!=Path(writer).resolve() for rows in bank.values() for r in rows):
        raise ValueError('self-memory writer does not match memory-writer initialization initialization')
    return dict(type='explicit_api_to_memory_writer_initialization_self_memory',source=old,target=current_fingerprint,
                sessions_and_source_dialogue_checked=True)


def check_session_transfer(old_rows, input_rows, subset_manifest=None):
    if subset_manifest is None:
        if set(old_rows) != set(input_rows):
            raise ValueError('pretraining/self session coverage changed')
    else:
        if subset_manifest.get('subset_policy') != 'schema_valid_nonempty_memory_and_all_qa_evidence_sessions_v1':
            raise ValueError('unrecognized subset policy')
        if subset_manifest.get('complete_data') is not False:
            raise ValueError('filtered data must declare incomplete original coverage')
        if not set(input_rows) <= set(old_rows) or not input_rows:
            raise ValueError('subset must contain only original training sessions')
    for jid, r in input_rows.items():
        if old_rows[jid]['input_sha256'] != r['input_sha256']:
            raise ValueError('source conversation changed')


class JointLoss(torch.nn.Module):
    def __init__(self, policy, bridge, anchor_weight=.1, maximum_sequence=32768):
        super().__init__()
        self.policy=policy; self.bridge=bridge
        self.anchor=copy.deepcopy(bridge).eval().requires_grad_(False)
        self.anchor_weight=anchor_weight; self.maximum_sequence=maximum_sequence

    def forward(self, states, prompt, target):
        mask=torch.ones(states.shape[:2],device=states.device,dtype=torch.long)
        soft=self.bridge(states,mask)
        self.anchor.eval()
        with torch.no_grad(): reference=self.anchor(states,mask)
        ce=target_nll(self.policy,prompt,target,soft,self.maximum_sequence)
        anchor=soft_anchor_loss(soft,reference)
        return ce+self.anchor_weight*anchor,ce.detach(),anchor.detach()


def scale_for_rank(rank, real_batch, world):
    if not 1<=real_batch<=world: raise ValueError('invalid distributed batch size')
    return world/real_batch if rank<real_batch else 0.


def main():
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','memories','cache','compressor-checkpoint','writer-adapter','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepared',type=Path,default=ROOT/'locomo_pipeline/prepared/full_v3_20260906')
    p.add_argument('--source-self-memories',type=Path,required=True)
    p.add_argument('--max-updates',type=int,default=0)
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--recipe',choices=RECIPES,default='legacy')
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--compressor-learning-rate',type=float,default=1e-5)
    p.add_argument('--anchor-weight',type=float,default=.1)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--allow-valid-session-subset',action='store_true')
    p.add_argument('--resume-from',type=Path,help='Completed joint reader initialization epoch directory; epochs is total target epoch count')
    a=p.parse_args()
    check_adapter(json.loads((a.writer_adapter/'adapter_config.json').read_text()),a.recipe)
    if a.recipe=='persona128k' and a.resume_from:
        check_adapter(json.loads((a.resume_from/'adapter/adapter_config.json').read_text()),a.recipe)
    if a.epochs<1 or a.max_updates<0 or min(a.learning_rate,a.compressor_learning_rate)<=0 or a.anchor_weight<0:
        raise ValueError('invalid hyperparameters')
    a.chunk_tokens=2048; a.pool_tokens=32
    if a.output.exists(): raise FileExistsError('refusing existing output')
    rows=read(a.memories); bank=memory_bank(rows)
    questions=qa_rows(read(a.prepared/'train/qa.jsonl'),bank)
    # Uses all eligible QA in the selected prepared dataset, not reasoning exclusions.
    source={r['task_id']:r for r in read(a.source_self_memories)}
    input_rows={r['task_id']:r for r in read(a.prepared/'train/writer_inputs.jsonl')}
    if set(input_rows)!={r['id'] for r in rows}: raise ValueError('train session coverage mismatch')
    adapter_hash=sha((a.writer_adapter/'adapter_model.safetensors').read_bytes())
    for r in rows:
        s=source[r['id']]
        if not s.get('schema_valid') or s['finish_reason']!='stop' or s['origin']!='self_generated':
            raise ValueError('unvalidated self memory')
        if s['adapter_sha256']!=adapter_hash or s['input_sha256']!=input_rows[r['id']]['input_sha256']:
            raise ValueError('self memory source/adapter hash mismatch')
        if json.loads(s['memory_text'])!=json.loads(r['memory_text']): raise ValueError('memory content changed')
    hidden=AutoConfig.from_pretrained(a.model,local_files_only=True).hidden_size
    bridge,config,parent=load(ROOT,a.compressor_checkpoint,hidden)
    transfer=transfer_gate(parent,fingerprint(a),bank,a.writer_adapter)
    # Verify pretraining used the exact same original session inputs, not only context names.
    old_mem=ROOT/'locomo_pipeline/prepared/api_v3_compressor_v1/train.jsonl'
    if sha(old_mem.read_bytes())!=parent['cache_fingerprint']['memories_sha256']:
        raise ValueError('pretraining memory provenance unavailable')
    old_rows={r['id']:r for r in read(old_mem)}
    subset = None
    if a.allow_valid_session_subset:
        subset=json.loads((a.prepared/'subset_manifest.json').read_text())
        subset_root=a.prepared.parent
        for name, digest in subset['file_hashes'].items():
            if sha((subset_root/name).read_bytes()) != digest:
                raise ValueError('subset artifact hash mismatch')
        if a.memories.resolve() != (a.prepared/'train.jsonl').resolve():
            raise ValueError('subset memory path mismatch')
        if a.source_self_memories.resolve() != (subset_root/'self_memories.jsonl').resolve():
            raise ValueError('subset source path mismatch')
        transfer.update(valid_session_subset=True, subset_manifest_sha256=sha((a.prepared/'subset_manifest.json').read_bytes()),
                        selected_train_sessions=len(input_rows), original_train_sessions=len(old_rows),
                        selected_train_qa=len(questions), qa_filter=subset['subset_policy'])
    check_session_transfer(old_rows,input_rows,subset)
    resume_payload=None; resume_optimizer=None
    if a.resume_from:
        resume_payload=torch.load(a.resume_from/'bridge.pt',map_location='cpu',weights_only=True)
        resume_optimizer=torch.load(a.resume_from/'optimizer.pt',map_location='cpu',weights_only=True)
        rp=resume_payload['provenance']
        if not rp.get('joint_reader_initialization') or rp.get('smoke_only') or rp.get('procedure')!='reader_initialization':
            raise ValueError('resume requires completed formal joint reader initialization')
        if rp['writer_adapter_sha256']!=adapter_hash or rp['cache_fingerprint']!=fingerprint(a):
            raise ValueError('resume writer/data mismatch')
        if rp['parent_bridge_sha256']!=sha(a.compressor_checkpoint.read_bytes()) or resume_payload['config']!=config:
            raise ValueError('resume anchor/bridge configuration mismatch')
        if not 0<resume_optimizer['epoch']<a.epochs:
            raise ValueError('resume epoch must precede total target epochs')
        if resume_optimizer['step']!=resume_optimizer['epoch']*math.ceil(len(questions)/4):
            raise ValueError('resume requires complete epochs')
    cache=json.loads((a.cache/'manifest.json').read_text())
    if not cache.get('complete') or cache['fingerprint']!=fingerprint(a): raise ValueError('cache mismatch')
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    encoded={q['question_id']:prompt_and_target(tok,'Answer the question concisely and completely.\nQuestion: '+q['question'],
        q['answer'],torch.device('cpu'),512) for q in questions}
    if a.validate_only:
        print(json.dumps(dict(status='preflight_passed',questions=len(questions),sessions=len(rows),transfer=transfer)),flush=True)
        return
    world=int(os.environ.get('WORLD_SIZE','1')); rank=int(os.environ.get('RANK','0')); local=int(os.environ.get('LOCAL_RANK','0'))
    if world!=4: raise ValueError('this pilot requires torchrun with exactly four workers')
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    torch.set_num_threads(4); torch.manual_seed(42)
    dist.init_process_group('nccl',device_id=device)
    try:
        if rank==0: a.output.mkdir(parents=True)
        dist.barrier()
        bridge_dtype=torch.float32 if a.recipe=='persona128k' else torch.bfloat16
        bridge=bridge.to(device=device,dtype=bridge_dtype)
        states,_=state_bank(a,bank,hidden,device,bridge_dtype)
        base=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
        policy=PeftModel.from_pretrained(base,a.resume_from/'adapter' if a.resume_from else a.writer_adapter,is_trainable=True)
        policy.config.use_cache=False
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        configure_scope(policy,bridge,'reader_initialization',True)
        if a.recipe=='persona128k':
            for par in policy.parameters():
                if par.requires_grad: par.data=par.data.float()
        objective=JointLoss(policy,bridge,a.anchor_weight)
        # Anchor remains the ORIGINAL reasoning-pretrained bridge, not the resumed bridge.
        if resume_payload:
            bridge.load_state_dict(resume_payload['bridge'],strict=True)
        groups=[dict(params=[p for p in policy.parameters() if p.requires_grad],lr=a.learning_rate),
                dict(params=[p for p in bridge.parameters() if p.requires_grad],lr=a.compressor_learning_rate)]
        opt=torch.optim.AdamW(groups,weight_decay=.01)
        if resume_optimizer:
            opt.load_state_dict(resume_optimizer['optimizer'])
            if [g['lr'] for g in opt.param_groups]!=[a.learning_rate,a.compressor_learning_rate]:
                raise ValueError('resume learning rate differs from saved optimizer')
        scope=audit_scope(policy,bridge,opt,'reader_initialization',True)
        wrapped=DDP(objective,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False)
        frozen=[p for p in objective.parameters() if not p.requires_grad]; versions=[p._version for p in frozen]
        active=[p for g in groups for p in g['params']]
        before=[p.detach().clone() for p in active]
        provenance=dict(procedure='reader_initialization',joint_reader_initialization=True,decoder_policy='writer',smoke_only=bool(a.max_updates),
            recipe=a.recipe,compressor_precision='float32' if a.recipe=='persona128k' else 'historical',
            parent_bridge_sha256=sha(a.compressor_checkpoint.read_bytes()),writer_adapter_sha256=adapter_hash,
            cache_fingerprint=fingerprint(a),training_contexts=sorted(bank),transfer=transfer,
            auxiliary_lora_transferred_to_reader_initialization=False,anchor_definition='relative_soft_output_mse',**scope)
        if a.resume_from:
            provenance.update(resumed_from=str(a.resume_from.resolve()),
                resumed_bridge_sha256=sha((a.resume_from/'bridge.pt').read_bytes()),
                resumed_adapter_sha256=sha((a.resume_from/'adapter/adapter_model.safetensors').read_bytes()),
                resumed_optimizer_sha256=sha((a.resume_from/'optimizer.pt').read_bytes()),
                optimizer_restored=True,anchor_reset=False,
                rng_note='Original epoch1 checkpoint omitted RNG state; seed42 restart, not bitwise uninterrupted continuation')
        if rank==0:
            save(a.output/'run_config.json',dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                world_size=world,effective_global_batch=4,training_questions=len(questions),
                trainable_policy_parameters=sum(p.numel() for p in groups[0]['params']),
                **scope,provenance=provenance,runner_sha256=sha(Path(__file__).read_bytes())))
        step=resume_optimizer['step'] if resume_optimizer else 0; epoch_means=[]
        start_epoch=resume_optimizer['epoch']+1 if resume_optimizer else 1
        for epoch in range(start_epoch,a.epochs+1):
            objective.train(); order=list(questions); random.Random(42+epoch).shuffle(order)
            totals=torch.zeros(4,device=device,dtype=torch.float64)
            for offset in range(0,len(order),world):
                batch=order[offset:offset+world]; present=rank<len(batch); q=batch[rank] if present else order[0]
                prompt,target=(x.to(device) for x in encoded[q['question_id']])
                opt.zero_grad(set_to_none=True)
                loss,ce,anchor=wrapped(states[q['context_id']],prompt,target)
                (loss*scale_for_rank(rank,len(batch),world)).backward()
                grad=torch.nn.utils.clip_grad_norm_(active,1.)
                ok=torch.tensor(int(bool(torch.isfinite(loss)) and bool(torch.isfinite(grad)) and float(grad)>0),device=device)
                dist.all_reduce(ok,op=dist.ReduceOp.MIN)
                if not ok.item(): raise FloatingPointError('nonfinite loss/gradient or zero gradient')
                opt.step(); step+=1
                if a.recipe=='persona128k': assert_fp32_optimizer(opt)
                if versions!=[p._version for p in frozen] or any(p.grad is not None for p in frozen):
                    raise ValueError('frozen parameter changed')
                stat=torch.tensor([float(loss.detach()),float(ce),float(anchor),1.],device=device,dtype=torch.float64) if present else torch.zeros(4,device=device,dtype=torch.float64)
                totals+=stat
                dist.all_reduce(stat)
                if rank==0:
                    metric=dict(epoch=epoch,step=step,total_steps=a.epochs*math.ceil(len(order)/world),
                        loss=float(stat[0]/stat[3]),answer_ce=float(stat[1]/stat[3]),
                        anchor=float(stat[2]/stat[3]),gradient_norm=float(grad),examples=int(stat[3]))
                    with (a.output/'metrics.jsonl').open('a') as f: f.write(json.dumps(metric)+'\n')
                    if step%10==0 or a.max_updates: print(json.dumps(metric),flush=True)
                if a.max_updates and step>=a.max_updates: break
            dist.all_reduce(totals)
            if not a.max_updates and int(totals[3])!=len(questions): raise ValueError('epoch coverage mismatch')
            changed=sum(not torch.equal(b,p.detach()) for b,p in zip(before,active))
            if changed==0: raise ValueError('no trainable parameter update')
            if rank==0:
                out=a.output/f'epoch_{epoch}'; out.mkdir()
                policy.save_pretrained(out/'adapter')
                torch.save(dict(bridge=bridge.state_dict(),config=config,provenance=provenance),out/'bridge.pt')
                torch.save(dict(optimizer=opt.state_dict(),step=step,epoch=epoch),out/'optimizer.pt')
                epoch_means.append(dict(epoch=epoch,examples=int(totals[3]),loss=float(totals[0]/totals[3]),answer_ce=float(totals[1]/totals[3])))
            dist.barrier()
            if a.max_updates and step>=a.max_updates: break
        if rank==0:
            save(a.output/'result.json',dict(status='smoke_completed' if a.max_updates else 'completed',
                updates=step,epoch_means=epoch_means,bridge=str(out/'bridge.pt'),adapter=str(out/'adapter'),
                parameters_changed=changed,frozen_parameters_unchanged=True,provenance=provenance,
                on_policy_optimization_launched=False,quality_evaluated=False))
    finally: dist.destroy_process_group()


if __name__=='__main__': main()
