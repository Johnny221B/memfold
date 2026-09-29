"""Single-GPU LoCoMo bridge cache/warmup/reasoning/reader initialization entrypoints.

No import-time jobs, API calls, Persona loaders, MCQ reward, or on-policy optimization training.
Explicit input files and fresh output directories are required.
"""
import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .compressor_data import memory_bank, qa_rows, reasoning_rows, negative_context, read, save, sha, source_identity
from .compressor_core import build, load, configure_scope, audit_scope, warmup_loss, separation, target_nll, soft_anchor_loss
from .persona128k_recipe import RECIPES, pool_hidden, paired_example, check_adapter

ROOT=Path(__file__).resolve().parent.parent


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('procedure',choices=['cache','compressor_reconstruction','representation_warmup','auxiliary_reasoning_adaptation','reader_initialization'])
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--memories',type=Path,required=True,help='validated train.jsonl from prepare_compressor_reconstruction or prepare_api_compressor')
    p.add_argument('--questions',type=Path,help='LoCoMo prepared train/qa.jsonl')
    p.add_argument('--traces',type=Path,help='audited reasoning-only train targets')
    p.add_argument('--cache',type=Path)
    p.add_argument('--compressor-checkpoint',type=Path)
    p.add_argument('--writer-adapter',type=Path)
    p.add_argument('--decoder-policy',choices=['writer','base'],default='writer',help='reader initialization frozen decoder; base preserves the prior text-reader control')
    p.add_argument('--pretraining-recipe',choices=['reasoning','representation'],default='reasoning',help='explicit representation-only pilot or historical reasoning chain')
    p.add_argument('--legacy-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validate-only',action='store_true',help='read-only data/checkpoint/cache preflight; no decoder/GPU training')
    p.add_argument('--allow-external-init',action='store_true',help='explicitly allow historical bridge without LoCoMo provenance')
    p.add_argument('--objective',choices=['reconstruction','qa'],default=None)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--warmup-steps',type=int,default=100)
    p.add_argument('--max-updates',type=int,default=0,help='nonzero produces smoke-only artifacts, not formal initialization')
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--compressor-learning-rate',type=float,default=3e-5)
    p.add_argument('--anchor-weight',type=float,default=.1)
    p.add_argument('--token-count',type=int,default=256)
    p.add_argument('--chunk-tokens',type=int,default=2048)
    p.add_argument('--pool-tokens',type=int,default=32)
    p.add_argument('--maximum-target-tokens',type=int,default=8192)
    p.add_argument('--maximum-sequence',type=int,default=32768)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--recipe',choices=RECIPES,default='legacy')
    p.add_argument('--compressor-precision',choices=['historical','float32'],default='historical',
                   help='historical follows base dtype; float32 keeps compressor and Adam moments FP32')
    p.add_argument('--save-initial',action='store_true',help='save initial bridge for controlled update audits')
    return p.parse_args()


def fingerprint(a):
    result=dict(model=str(a.model.resolve()),config_sha256=sha((a.model/'config.json').read_bytes()),
        tokenizer_sha256=sha((a.model/'tokenizer.json').read_bytes()),
        memories_sha256=sha(a.memories.read_bytes()),chunk_tokens=a.chunk_tokens,pool_tokens=a.pool_tokens,
        encoder='frozen base, no LoRA',split='train')
    if getattr(a,'recipe','legacy')=='persona128k':
        result.update(recipe='persona128k',pooling='consecutive_window_mean_v1',cache_dtype='float16')
    return result


def encode_cache(a, rows, model, tok, device):
    state_map={}
    for row in rows:
        ids=tok.encode(row['memory_text'],add_special_tokens=False)
        if not ids: raise ValueError('empty encoded memory')
        pieces=[]
        with torch.no_grad():
            for start in range(0,len(ids),a.chunk_tokens):
                tokens=torch.tensor([ids[start:start+a.chunk_tokens]],device=device)
                h=model.model(input_ids=tokens,use_cache=False).last_hidden_state
                recipe=getattr(a,'recipe','legacy')
                pooled=pool_hidden(h,a.pool_tokens,recipe)[0]
                pieces.append(pooled.cpu().to(torch.float16 if recipe=='persona128k' else torch.bfloat16))
        name=row['state_id']+'.pt'
        torch.save({'states':torch.cat(pieces),'state_id':row['state_id']},a.output/name)
        state_map[row['id']]=name
        print(json.dumps(dict(event='cached',done=len(state_map),total=len(rows))),flush=True)
    save(a.output/'manifest.json',dict(fingerprint=fingerprint(a),states=state_map,
        hidden_size=model.config.hidden_size,complete=True))


def state_bank(a, bank, hidden, device, dtype):
    manifest=json.loads((a.cache/'manifest.json').read_text())
    if not manifest.get('complete') or manifest['fingerprint']!=fingerprint(a) or manifest['hidden_size']!=hidden:
        raise ValueError('cache provenance/config mismatch')
    if set(manifest['states'])!={r['id'] for rows in bank.values() for r in rows}:
        raise ValueError('cache ID coverage mismatch')
    result={}
    sessions={}
    for context, rows in bank.items():
        parts=[]
        for r in rows:
            path=a.cache/manifest['states'][r['id']]
            if path.resolve().parent!=a.cache.resolve(): raise ValueError('cache path outside cache directory')
            payload=torch.load(path,map_location='cpu',weights_only=True)
            x=payload['states']
            if getattr(a,'recipe','legacy')=='persona128k' and x.dtype!=torch.float16:
                raise ValueError('persona128k cache must contain FP16 states')
            if payload['state_id']!=r['state_id'] or x.ndim!=2 or x.shape[1]!=hidden or not torch.isfinite(x).all():
                raise ValueError('invalid cached states')
            sessions[r['id']]=x.to(device=device,dtype=dtype).unsqueeze(0)
            parts.append(x)
        result[context]=torch.cat(parts).to(device=device,dtype=dtype).unsqueeze(0)
    return result,sessions


def prompt_and_target(tok, text, target, device, limit):
    prompt=tok.apply_chat_template([dict(role='system',content='Memory is provided before this conversation. Use its information; do not invent facts.'),
        dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    target_ids=tok.encode(target,add_special_tokens=False)+[tok.eos_token_id]
    if len(target_ids)>limit:
        raise ValueError('target exceeds configured cap; no silent target truncation')
    return torch.tensor([prompt],device=device),torch.tensor([target_ids],device=device)


def main():
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model
    a=arguments()
    a.joint_reader_initialization = a.procedure == "reader_initialization"
    if a.objective is None:
        a.objective = "qa" if a.joint_reader_initialization else "reconstruction"
    if a.joint_reader_initialization and a.objective != "qa":
        raise ValueError("reader_initialization requires the QA objective")
    aligned=a.recipe=='persona128k'
    if aligned and a.procedure=='reader_initialization':
        raise ValueError('persona128k reader initialization uses train_joint_reader_initialization --recipe persona128k (four-worker global batch4)')
    if aligned:
        a.compressor_precision='float32'
        if a.procedure in ('compressor_reconstruction','representation_warmup'): a.compressor_learning_rate=a.learning_rate
    if a.output.exists(): raise FileExistsError('refusing to overwrite output')
    if min(a.epochs,a.warmup_steps,a.token_count,a.chunk_tokens,a.pool_tokens,a.maximum_target_tokens,a.maximum_sequence)<=0:
        raise ValueError('positive counts/caps required')
    if a.max_updates<0 or min(a.learning_rate,a.compressor_learning_rate)<=0 or a.anchor_weight<0:
        raise ValueError('invalid update count, learning rate or anchor')
    if a.joint_reader_initialization and a.procedure!='reader_initialization': raise ValueError('joint option only applies to reader initialization')
    if a.joint_reader_initialization and a.decoder_policy=='base': raise ValueError('base decoder control cannot train LoRA')
    if a.objective=='qa' and a.procedure!='reader_initialization': raise ValueError('QA objective only applies to reader initialization')
    torch.manual_seed(a.seed); torch.set_num_threads(4)
    rng=random.Random(a.seed)
    rows=read(a.memories); bank=memory_bank(rows)
    hidden=AutoConfig.from_pretrained(a.model,local_files_only=True).hidden_size
    device=torch.device(a.device)
    dtype=torch.bfloat16 if device.type=='cuda' else torch.float32
    bridge_dtype=torch.float32 if a.compressor_precision=='float32' else dtype
    if a.procedure!='cache' and a.cache is None: raise ValueError('--cache is required')
    if a.procedure in ('auxiliary_reasoning_adaptation','reader_initialization') and a.compressor_checkpoint is None:
        raise ValueError('pretrained --compressor-checkpoint required; random reader initialization initialization prohibited')
    if a.procedure=='reader_initialization' and a.writer_adapter is None: raise ValueError('memory-writer initialization --writer-adapter required')
    questions=[]; traces={}
    if a.procedure=='auxiliary_reasoning_adaptation' or (a.procedure=='reader_initialization' and a.objective=='qa'):
        if a.questions is None: raise ValueError('--questions required')
        questions=qa_rows(read(a.questions),bank)
    if a.procedure=='auxiliary_reasoning_adaptation':
        if a.traces is None: raise ValueError('--traces required; gold answers are not reasoning targets')
        traces=reasoning_rows(read(a.traces),questions,sha(a.memories.read_bytes()))
    config=dict(context_dim=hidden,lm_dim=hidden,latent_dim=768,token_count=a.token_count,
                layers=2,heads=12,context_residual=True)
    parent={}
    bridge=None
    if a.procedure!='cache':
        if a.compressor_checkpoint:
            bridge,config,parent=load(a.legacy_root,a.compressor_checkpoint,hidden)
            if not parent and not a.allow_external_init:
                raise ValueError('external/Persona initialization requires --allow-external-init')
            if parent.get('smoke_only'): raise ValueError('smoke checkpoint cannot initialize formal training')
            if parent and parent.get('cache_fingerprint')!=fingerprint(a):
                raise ValueError('bridge training data/encoder mismatch; create a separately reviewed transfer initialization')
            required_parent='auxiliary_reasoning_adaptation' if a.pretraining_recipe=='reasoning' else 'representation_warmup'
            if a.procedure=='reader_initialization' and parent and parent.get('procedure')!=required_parent:
                raise ValueError('reader initialization bridge does not match the explicit pretraining recipe')
            if a.procedure=='auxiliary_reasoning_adaptation' and parent and parent.get('procedure')!='representation_warmup':
                raise ValueError('reasoning pretraining requires representation warmup checkpoint')
            if config['token_count']!=a.token_count: raise ValueError('checkpoint K differs from requested K')
        else:
            bridge=build(a.legacy_root,config)
    if a.procedure in ('representation_warmup','auxiliary_reasoning_adaptation') and len(bank)<2:
        raise ValueError('pretraining requires at least two train contexts')
    if aligned and a.procedure in ('representation_warmup','compressor_reconstruction') and (len(bank)<2 or any(len(v)<2 for v in bank.values())):
        raise ValueError('persona128k contrastive views require two contexts, each with two sessions')
    if a.procedure=='reader_initialization':
        if rows[0].get('origin')!='api_teacher' and a.writer_adapter.resolve()!=Path(source_identity(rows[0])).resolve():
            raise ValueError('reader initialization adapter must match the self-memory writer')
        if not (a.writer_adapter/'adapter_config.json').is_file():
            raise ValueError('writer adapter configuration missing')
        check_adapter(json.loads((a.writer_adapter/'adapter_config.json').read_text()),a.recipe)
    if a.procedure!='cache':
        manifest=json.loads((a.cache/'manifest.json').read_text())
        if not manifest.get('complete') or manifest['fingerprint']!=fingerprint(a) or manifest['hidden_size']!=hidden:
            raise ValueError('cache provenance/config mismatch')
        if set(manifest['states'])!={r['id'] for r in rows}:
            raise ValueError('cache ID coverage mismatch')
        for name in manifest['states'].values():
            path=a.cache/name
            if path.resolve().parent!=a.cache.resolve() or not path.is_file():
                raise ValueError('missing or unsafe cache state path')
    if a.validate_only:
        print(json.dumps(dict(status='input_preflight_passed',procedure=a.procedure,memories=len(rows),
            contexts=len(bank),questions=len(questions),gpu_training_launched=False,
            note='metadata/file coverage only; tensor contents checked on actual training load')),flush=True)
        return
    if a.procedure!='cache':
        bridge=bridge.to(device=device,dtype=bridge_dtype)
        states,sessions=state_bank(a,bank,hidden,device,bridge_dtype)
    if a.procedure=='representation_warmup' and any(x.shape[1]<2 for x in states.values()):
        raise ValueError('two nonempty warmup views required')
    tok=None
    model=torch.nn.Identity()
    if a.procedure!='representation_warmup':
        tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,
            torch_dtype=dtype,attn_implementation='sdpa').to(device)
        model.requires_grad_(False).eval()
    if a.procedure=='reader_initialization' and a.decoder_policy=='writer':
        model=PeftModel.from_pretrained(model,a.writer_adapter,is_trainable=a.joint_reader_initialization)
    elif a.procedure=='auxiliary_reasoning_adaptation':
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,
            target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM'))
    a.output.mkdir(parents=True)
    run_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    save(a.output/'run_config.json',run_config)
    if a.procedure=='cache':
        encode_cache(a,rows,model,tok,device)
        return
    configure_scope(model,bridge,a.procedure,a.joint_reader_initialization)
    if a.compressor_precision=='float32':
        for p in model.parameters():
            if p.requires_grad: p.data=p.data.float()
    model.eval(); bridge.train()
    groups=[dict(params=[p for p in bridge.parameters() if p.requires_grad],lr=a.compressor_learning_rate)]
    policy=[p for p in model.parameters() if p.requires_grad]
    if policy:
        model.train()
        model.config.use_cache=False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if policy: groups.append(dict(params=policy,lr=a.learning_rate))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    scope=audit_scope(model,bridge,optimizer,a.procedure,a.joint_reader_initialization)
    frozen=[p for m in (model,bridge) for p in m.parameters() if not p.requires_grad]
    versions=[p._version for p in frozen]
    anchor_bridge=copy.deepcopy(bridge).requires_grad_(False).eval() if a.joint_reader_initialization else None
    if anchor_bridge is not None:
        frozen.extend(anchor_bridge.parameters())
        versions=[p._version for p in frozen]
    provenance=dict(procedure=a.procedure,smoke_only=bool(a.max_updates),cache_fingerprint=fingerprint(a),
        memory_source_id=source_identity(rows[0]),memory_origin=rows[0].get('origin','self_generated'),
        source_writer=rows[0].get('source_adapter'),parent_bridge_sha256=sha(a.compressor_checkpoint.read_bytes()) if a.compressor_checkpoint else None,
        external_initialization=bool(a.compressor_checkpoint and not parent) or parent.get('external_initialization',False),
        auxiliary_lora_transferred_to_reader_initialization=False,training_contexts=sorted(bank),**scope)
    provenance['decoder_policy']=a.decoder_policy if a.procedure=='reader_initialization' else 'base'
    provenance['pretraining_recipe']=a.pretraining_recipe
    provenance['compressor_precision']=a.compressor_precision
    provenance['recipe']=a.recipe
    if aligned:
        provenance['implementation_sha256']={name:sha((Path(__file__).parent/name).read_bytes())
            for name in ('train_compressor.py','compressor_core.py','persona128k_recipe.py')}
    provenance['effective_batch']=4 if aligned and a.procedure=='auxiliary_reasoning_adaptation' else 1
    provenance['view_definition']='distinct question-blind sessions, not question-conditioned Persona memories' if aligned else 'legacy'
    provenance['anchor_definition']='relative_soft_output_mse' if a.joint_reader_initialization else None
    if a.traces: provenance['traces_sha256']=sha(a.traces.read_bytes())
    if a.questions: provenance['questions_sha256']=sha(a.questions.read_bytes())
    if a.save_initial:
        torch.save(dict(bridge=bridge.state_dict(),config=config,provenance=provenance),a.output/'initial_bridge.pt')
    def soft(x):
        return bridge(x,torch.ones(x.shape[:2],device=device,dtype=torch.long))
    step=0
    epochs=1 if a.procedure=='representation_warmup' else a.epochs
    for epoch in range(1,epochs+1):
        # Historical reasoning schedule: reset optimizer at the 1+2 epoch boundary.
        if a.procedure=='auxiliary_reasoning_adaptation' and epoch==2:
            rng.seed(a.seed+1)
            torch.manual_seed(a.seed+1)
            groups[-1]['lr']=a.learning_rate*.5
            optimizer=torch.optim.AdamW(groups,weight_decay=.01)
        examples=list(range(a.warmup_steps)) if a.procedure=='representation_warmup' else list(questions if a.procedure=='auxiliary_reasoning_adaptation' or a.objective=='qa' else rows)
        if a.procedure!='representation_warmup': rng.shuffle(examples)
        accumulation=4 if aligned and a.procedure=='auxiliary_reasoning_adaptation' else 1
        optimizer.zero_grad(set_to_none=True)
        batch_loss=0.; batch_ids=[]
        for item_index,item in enumerate(examples):
            loss_components={}
            if a.procedure=='representation_warmup':
                views=[]; labels=[]
                for index, context in enumerate(sorted(states)):
                    if aligned:
                        rr=bank[context]
                        xs=[sessions[rr[(2*item+offset)%len(rr)]['id']] for offset in (0,1)]
                    else:
                        x=states[context]; xs=[x[:,offset::2] for offset in (0,1)]
                    for x in xs:
                        views.append(soft(x)); labels.append(index)
                loss=warmup_loss(torch.cat(views),labels,off_diagonal_gram=aligned)
            else:
                is_recon=a.procedure=='compressor_reconstruction' or (a.procedure=='reader_initialization' and a.objective=='reconstruction')
                context=item['context_id']
                own=soft(sessions[item['id']] if is_recon else states[context])
                if is_recon:
                    instruction='Reconstruct the source memory. Return only its native JSON.'
                    target=item['memory_text']
                elif a.procedure=='auxiliary_reasoning_adaptation':
                    instruction='Give concise evidence-grounded reasoning. Do not state a final answer.\nQuestion: '+item['question']
                    target=traces[item['question_id']]['reasoning']
                else:
                    instruction='Answer the question concisely and completely.\nQuestion: '+item['question']
                    target=item['answer']
                prompt,target_ids=prompt_and_target(tok,instruction,target,device,a.maximum_target_tokens)
                loss=target_nll(model,prompt,target_ids,own.to(dtype),a.maximum_sequence)
                if aligned and a.procedure=='compressor_reconstruction':
                    negative=paired_example(examples,item_index,False)
                    positive=paired_example(examples,item_index,True)
                    other=soft(sessions[negative['id']])
                    positive_soft=soft(sessions[positive['id']])
                    wrong=target_nll(model,prompt,target_ids,other.to(dtype),a.maximum_sequence)
                    ranking=F.relu(.05+loss-wrong)
                    sep=separation(own,other)
                    attraction=(1-F.cosine_similarity(own.float().mean(1),positive_soft.float().mean(1))).mean()
                    loss_components=dict(own_nll=float(loss.detach()),wrong_nll=float(wrong.detach()),
                        ranking_loss=float(ranking.detach()),separation_loss=float(sep.detach()),
                        alignment_loss=float(attraction.detach()),ranking_weight=.2,separation_weight=1.,alignment_weight=2.,
                        negative_id=negative['id'],positive_id=positive['id'])
                    loss=loss+.2*ranking+sep+2*attraction
                if a.procedure=='auxiliary_reasoning_adaptation':
                    donor=paired_example(examples,item_index,False)['context_id'] if aligned else negative_context(context,bank)
                    other=soft(states[donor])
                    wrong=target_nll(model,prompt,target_ids,other.to(dtype),a.maximum_sequence)
                    margin=.05 if epoch==1 else .1
                    weight=.2 if epoch==1 else 1.0
                    ranking=F.relu(margin+loss-wrong)
                    sep=separation(own,other)
                    loss_components=dict(own_nll=float(loss.detach()),wrong_nll=float(wrong.detach()),
                        ranking_loss=float(ranking.detach()),separation_loss=float(sep.detach()),
                        ranking_weight=weight,ranking_margin=margin,separation_weight=.1)
                    loss=loss+weight*ranking+.1*sep
                if anchor_bridge is not None:
                    anchor_states=sessions[item['id']] if is_recon else states[context]
                    with torch.no_grad():
                        initial_soft=anchor_bridge(anchor_states,torch.ones(anchor_states.shape[:2],device=device,dtype=torch.long))
                    reg=soft_anchor_loss(own,initial_soft)
                    loss=loss+a.anchor_weight*reg
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            real_batch=min(accumulation,len(examples)-(item_index//accumulation)*accumulation)
            (loss/real_batch).backward()
            batch_loss+=float(loss.detach())/real_batch
            if isinstance(item,dict): batch_ids.append(item.get('question_id',item.get('id')))
            if (item_index+1)%accumulation and item_index+1<len(examples): continue
            active=[p for group in optimizer.param_groups for p in group['params']]
            norm=torch.nn.utils.clip_grad_norm_(active,1.)
            if not torch.isfinite(norm) or float(norm)==0: raise FloatingPointError('invalid/zero gradient')
            if any(p.grad is not None for p in frozen): raise ValueError('frozen parameter received gradient')
            optimizer.step(); step+=1
            if a.compressor_precision=='float32':
                if any(p.dtype!=torch.float32 for p in active): raise ValueError('FP32 master parameter contract')
                if any(s[k].dtype!=torch.float32 or not torch.isfinite(s[k]).all()
                       for s in optimizer.state.values() for k in ('exp_avg','exp_avg_sq')):
                    raise ValueError('FP32 finite optimizer moment contract')
            if any(p._version!=v for p,v in zip(frozen,versions)): raise ValueError('frozen parameter changed')
            metric=dict(step=step,epoch=epoch,loss=batch_loss,gradient_norm=float(norm),**loss_components)
            if accumulation>1:
                metric.update(example_ids=batch_ids,microbatch_count=real_batch,components_scope='last_microbatch')
            if isinstance(item,dict): metric['example_id']=item.get('question_id',item.get('id'))
            with (a.output/'metrics.jsonl').open('a') as f: f.write(json.dumps(metric)+'\n')
            print(json.dumps(metric),flush=True)
            optimizer.zero_grad(set_to_none=True)
            batch_loss=0.; batch_ids=[]
            if a.max_updates and step>=a.max_updates: break
        out=a.output/f'epoch_{epoch}'; out.mkdir()
        torch.save(dict(bridge=bridge.state_dict(),config=config,provenance=provenance),out/'bridge.pt')
        if policy: model.save_pretrained(out/('auxiliary_lora' if a.procedure=='auxiliary_reasoning_adaptation' else 'adapter'))
        torch.save(dict(optimizer=optimizer.state_dict(),step=step,epoch=epoch,
            torch_rng=torch.get_rng_state(),python_rng=rng.getstate()),out/'optimizer.pt')
        if a.max_updates and step>=a.max_updates: break
    save(a.output/'result.json',dict(status='smoke_completed' if a.max_updates else 'completed',updates=step,
        bridge=str((out/'bridge.pt').resolve()),scope=scope,provenance=provenance,
        automatic_resume_supported=False,quality_evaluated=False))


if __name__=='__main__': main()
