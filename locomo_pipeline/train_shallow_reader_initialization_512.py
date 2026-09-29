"""Full-conversation fixed512 reader initialization using readable first-block mappings."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from .component_alignment_trial import ROOT, RUN, append
from .compressor_data import read, save, sha, memory_bank, qa_rows


def prepare(out, source=None):
    src = source or RUN / 'shallow_capacity_v1'
    old = json.loads((src / 'protocol.json').read_text())
    memories = RUN / 'self_memory_retry_v1/prepared/train.jsonl'
    questions = ROOT / 'locomo_pipeline/prepared/full_v3_20260906/train/qa.jsonl'
    rows = read(memories); bank = memory_bank(rows); qa = qa_rows(read(questions), bank)
    data = json.loads((src / 'data.json').read_text())
    manifest = json.loads((src / 'cache/manifest.json').read_text())
    for r in rows:
        assert r['memory_text'] == data['sessions'][r['id']]['memory_text'] and r['id'] in manifest
    sources = [Path(__file__), ROOT / 'locomo_pipeline/session_memory_trial.py', ROOT / 'locomo_pipeline/compressor_core.py',
               ROOT / 'locomo_pipeline/train_compressor.py', ROOT / 'locomo_pipeline/persona128k_recipe.py',
               memories, questions, src / 'data.json', src / 'cache/manifest.json', src / 'mapper.pt', src / 'k512/compressor.pt',
               Path(old['adapter']) / 'adapter_model.safetensors', Path(old['model']) / 'config.json']
    out.mkdir(parents=True)
    save(out / 'data.json', dict(memories=rows, questions=qa, sessions={r['id']: data['sessions'][r['id']] for r in rows}))
    layer_key=old.get('encoder_layer_key','first');layer_index=old.get('encoder_layer_index',1)
    save(out / 'protocol.json', dict(model=old['model'], initial_reader=old['adapter'], mapper=str(src / 'mapper.pt'),
         initial_compressor=str(src / 'k512/compressor.pt'), cache=str(src / 'cache'),
         session_data=str(src/'data.json'),encoder_layer_key=layer_key,encoder_layer_index=layer_index,
         sources={str(p): sha(p.read_bytes()) for p in sources}, data_hash=sha((out / 'data.json').read_bytes()),
         epochs=10, global_batch=4, seed=801, reader_lr=1e-5, compressor_lr=1e-4,
         output_tokens=512, encoder_layer=f'transformer block hidden_states[{layer_index}], no LoRA, no pooling',
         input='all original memory JSON tokens from all sessions in the conversation, in source order; no retrieval or truncation',
         objective='answer-only CE + .2 relu(.5 + own CE - different-conversation wrong CE)',
         trainable=['all K-independent compressor parameters', 'reader LoRA'], frozen=['base encoder', 'base reader weights', f'block-{layer_index} mapper', 'memory writer'],
         optimizer='AdamW FP32 parameters/moments, weight_decay .01, global gradient clipping1.0',
         checkpoints=[1, 5, 10], selection='fixed final epoch10; epoch5 and initialization are prespecified LongMemEval comparisons, not test-selected checkpoints',
         train_contexts=sorted(bank), train_questions=len(qa), train_sessions=len(rows),
         evaluation='LongMemEval-S existing fixed50 exploratory panel only, no LongMemEval training',
         initialization_note='new matched512 reader initialization adaptation from prior epoch5 reader plus shallow512 field-trained compressor; NOT a256 checkpoint relabeled512',
         on_policy_optimization_launched=False, promotion_allowed=False))


def worker(out, max_updates=0):
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from .session_memory_trial import build_compressor
    from .train_compressor import prompt_and_target
    from .compressor_core import target_nll
    from .persona128k_recipe import assert_fp32_optimizer
    cfg = json.loads((out / 'protocol.json').read_text()); data = json.loads((out / 'data.json').read_text())
    for p, digest in cfg['sources'].items(): assert sha(Path(p).read_bytes()) == digest
    assert sha((out / 'data.json').read_bytes()) == cfg['data_hash']
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK']); world = int(os.environ['WORLD_SIZE'])
    assert world == cfg['global_batch'] == 4
    torch.cuda.set_device(local); device = torch.device('cuda', local)
    torch.set_num_threads(4); torch.manual_seed(cfg['seed'])
    dist.init_process_group('nccl', device_id=device)
    dest = out / ('smoke' if max_updates else 'train')
    try:
        if rank == 0: dest.mkdir()
        dist.barrier()
        tok = AutoTokenizer.from_pretrained(cfg['model'], local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(cfg['model'], local_files_only=True, torch_dtype=torch.bfloat16,
                   attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        h = base.config.hidden_size
        mapper = torch.nn.Sequential(torch.nn.LayerNorm(h), torch.nn.Linear(h,1024), torch.nn.GELU(), torch.nn.Linear(1024,h)).to(device)
        ck = torch.load(cfg['mapper'], map_location='cpu', weights_only=True); assert ck['encoder_layer'] == 'first'
        mapper.load_state_dict(ck['state_dict'], strict=True); mapper.eval().requires_grad_(False)
        manifest = json.loads((Path(cfg['cache']) / 'manifest.json').read_text())
        bank = memory_bank(data['memories']); states = {}; hashes = {}
        with torch.no_grad():
            for c, rows in bank.items():
                parts = []
                for r in rows:
                    path = Path(cfg['cache']) / manifest[r['id']]
                    item = torch.load(path, map_location='cpu', weights_only=True)
                    assert item['encoder_layer'] == 'first' and item['session'] == r['id']
                    assert item['states'].shape[1] == data['sessions'][r['id']]['tokens']
                    assert tok.encode(r['memory_text'], add_special_tokens=False) == item['ids'][0].tolist()
                    parts.append(mapper(item['states'].to(device).float()))
                    hashes[r['id']] = sha(path.read_bytes())
                states[c] = torch.cat(parts, 1)
                assert states[c].shape[1] == sum(data['sessions'][r['id']]['tokens'] for r in rows)
        policy = PeftModel.from_pretrained(base, cfg['initial_reader'], is_trainable=True)
        for name, par in policy.named_parameters():
            par.requires_grad_('lora_' in name)
            if par.requires_grad: par.data = par.data.float()
        policy.config.use_cache = False
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        compressor = build_compressor(h).to(device)
        ck = torch.load(cfg['initial_compressor'], map_location='cpu', weights_only=True)
        assert ck['output_tokens'] == 512 and ck['mapper_hash'] == sha(Path(cfg['mapper']).read_bytes())
        compressor.load_state_dict(ck['state_dict'], strict=True)
        class Objective(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.reader = policy; self.compressor = compressor
            def forward(self, own_states, wrong_states, prompt, target):
                own_soft = self.compressor(own_states, 512); wrong_soft = self.compressor(wrong_states, 512)
                assert own_soft.shape[1] == wrong_soft.shape[1] == 512
                own = target_nll(self.reader, prompt, target, own_soft, 32768)
                wrong = target_nll(self.reader, prompt, target, wrong_soft, 32768)
                margin = F.relu(.5 + own - wrong)
                return own + .2 * margin, own.detach(), wrong.detach(), margin.detach()
        objective = Objective()
        wrapped = DDP(objective, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
        reader_params = [p for p in policy.parameters() if p.requires_grad]
        comp_params = list(compressor.parameters()); active = reader_params + comp_params
        frozen = [p for p in policy.parameters() if not p.requires_grad] + list(mapper.parameters())
        versions = [p._version for p in frozen]
        before = [p.detach().clone() for p in active]
        opt = torch.optim.AdamW([dict(params=reader_params,lr=cfg['reader_lr']), dict(params=comp_params,lr=cfg['compressor_lr'])], weight_decay=.01)
        encoded = {q['question_id']: prompt_and_target(tok, 'Answer the question concisely and completely.\nQuestion: ' + q['question'], q['answer'], device, 512) for q in data['questions']}
        if rank == 0:
            save(dest / 'cache_audit.json', dict(session_hashes=hashes, context_tokens={c:x.shape[1] for c,x in states.items()}, no_input_pooling=True, no_truncation=True))
            save(dest / 'scope.json', dict(reader_parameters=sum(p.numel() for p in reader_params), compressor_parameters=sum(p.numel() for p in comp_params), output_tokens=512, frozen_mapper=True))
        step = 0; epoch_summaries = []; contexts = sorted(states)
        for epoch in range(1, cfg['epochs'] + 1):
            objective.train(); order = list(data['questions']); random.Random(cfg['seed'] + epoch).shuffle(order)
            totals = torch.zeros(4,device=device,dtype=torch.float64)
            for offset in range(0,len(order),world):
                batch = order[offset:offset+world]; present = rank < len(batch); q = batch[rank] if present else order[0]
                donor_rng = random.Random(f"{cfg['seed']}|{epoch}|{q['question_id']}")
                donor = donor_rng.choice([c for c in contexts if c != q['context_id']])
                pp, yy = encoded[q['question_id']]
                opt.zero_grad(set_to_none=True)
                loss, ce, wrong, margin = wrapped(states[q['context_id']], states[donor], pp, yy)
                (loss * (world / len(batch) if present else 0.)).backward()
                grad = torch.nn.utils.clip_grad_norm_(active,1.)
                ok = torch.tensor(int(bool(torch.isfinite(loss)) and bool(torch.isfinite(grad))),device=device)
                dist.all_reduce(ok,op=dist.ReduceOp.MIN)
                if not ok.item(): raise FloatingPointError('nonfinite training')
                opt.step(); step += 1; assert_fp32_optimizer(opt)
                assert versions == [p._version for p in frozen] and not any(p.grad is not None for p in frozen)
                stat = torch.tensor([float(ce),float(wrong),float(margin),1.],device=device,dtype=torch.float64) if present else torch.zeros(4,device=device,dtype=torch.float64)
                totals += stat; dist.all_reduce(stat)
                if rank == 0:
                    row = dict(epoch=epoch,step=step,total_steps=cfg['epochs']*math.ceil(len(order)/world), own_nll=float(stat[0]/stat[3]), wrong_nll=float(stat[1]/stat[3]), margin=float(stat[2]/stat[3]),grad=float(grad),examples=int(stat[3]))
                    append(dest / 'metrics.jsonl',row)
                    if step % 20 == 0 or max_updates: print(json.dumps(row),flush=True)
                if max_updates and step >= max_updates: break
            dist.all_reduce(totals)
            if not max_updates: assert int(totals[3]) == len(data['questions'])
            reader_changed = sum(not torch.equal(a,p) for a,p in zip(before[:len(reader_params)],reader_params))
            comp_changed = sum(not torch.equal(a,p) for a,p in zip(before[len(reader_params):],comp_params))
            assert reader_changed > 0 and comp_changed > 0
            if rank == 0:
                epoch_summaries.append(dict(epoch=epoch,examples=int(totals[3]),own_nll=float(totals[0]/totals[3]),wrong_nll=float(totals[1]/totals[3])))
                save(dest / 'progress.json',dict(step=step,epoch=epoch,epoch_summaries=epoch_summaries))
                if epoch in cfg['checkpoints'] or max_updates:
                    ep = dest / f'epoch_{epoch}'; ep.mkdir(); policy.save_pretrained(ep / 'adapter')
                    torch.save(dict(state_dict=compressor.state_dict(),hidden=h,output_tokens=512,procedure='reader_initialization',smoke_only=bool(max_updates),
                        mapper_hash=sha(Path(cfg['mapper']).read_bytes()),protocol_hash=sha((out/'protocol.json').read_bytes()),promotion_allowed=False),ep/'compressor.pt')
                    torch.save(dict(optimizer=opt.state_dict(),step=step,epoch=epoch),ep/'optimizer.pt')
            dist.barrier()
            if max_updates and step >= max_updates: break
        if rank == 0:
            save(dest/'result.json',dict(status='smoke_completed' if max_updates else 'completed',updates=step,epochs=epoch_summaries,
                 final_checkpoint=str(ep),reader_changed_tensors=reader_changed,compressor_changed_tensors=comp_changed,
                 frozen_unchanged=True,on_policy_optimization_launched=False,promotion_allowed=False))
    finally: dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true'); p.add_argument('--max-updates',type=int,default=0)
    a=p.parse_args(); out=a.output.resolve()
    if a.worker: worker(out,a.max_updates); return
    from .run_persona128k_alignment import gpu_free
    if out.exists(): raise FileExistsError(out)
    if not gpu_free([0,1,2,3]): raise RuntimeError('GPU0-3 busy')
    prepare(out)
    try:
        for name, extra in [('smoke',['--max-updates','2']),('train',[])]:
            save(out/'status.json',dict(status='running',phase=name,updated=time.time()))
            with (out/(name+'.log')).open('x') as log:
                rc=subprocess.call([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                    '-m',__spec__.name,'--output',str(out),'--worker',*extra],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3','TOKENIZERS_PARALLELISM':'false'})
            if rc: raise RuntimeError(name+' failed')
        save(out/'status.json',dict(status='completed',updated=time.time(),on_policy_optimization_launched=False,evaluation_pending=True))
    except Exception as exc:
        save(out/'status.json',dict(status='failed',error=repr(exc),updated=time.time())); raise


if __name__ == '__main__': main()
