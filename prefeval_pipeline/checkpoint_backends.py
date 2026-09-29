"""Inference-only loaders for the four RQ2 Qwen3-4B checkpoint families."""
import json
import sys
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'models/Qwen3-4B'
CHECKPOINTS=ROOT/'checkpoints'
METHODS=('vanilla-grpo','opsd','autocompressor','memgen')

def checkpoint(method):
    if method not in METHODS: raise ValueError(method)
    return CHECKPOINTS/method/'qwen3-4b/last'

def summary_length(path):
    from safetensors import safe_open
    with safe_open(path/'adapter_model.safetensors',framework='pt') as f:
        matches=[k for k in f.keys() if k.endswith('embed_summary.weight')]
        if len(matches)!=1: raise ValueError('Missing or ambiguous trained summary embedding')
        shape=f.get_slice(matches[0]).get_shape()
    if shape[1]!=2560: raise ValueError('Not a Qwen3-4B summary checkpoint')
    return shape[0]

def load(method,device):
    path=checkpoint(method)
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    if method in ('vanilla-grpo','opsd'):
        base=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        model=PeftModel.from_pretrained(base,path,is_trainable=False).to(device).eval()
        return model,tok,{'kind':'reader_lora','rank':json.loads((path/'adapter_config.json').read_text())['r']}
    if method=='autocompressor':
        sys.path.insert(0,str(ROOT.parent/'src'))
        from memory_opd.baselines.autocompressor import autocompressor_class
        config=AutoConfig.from_pretrained(BASE,local_files_only=True)
        config.summary_length=summary_length(path);config.accumulate_summary=True
        base=autocompressor_class(config).from_pretrained(BASE,config=config,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
        adapted=PeftModel.from_pretrained(base,path,is_trainable=False)
        from safetensors import safe_open
        with safe_open(path/'adapter_model.safetensors',framework='pt') as f:
            key=next(k for k in f.keys() if k.endswith('embed_summary.weight'))
            trained_summary=f.get_tensor(key)
        model=adapted.merge_and_unload().to(device).eval()
        if not torch.equal(model.embed_summary.weight.detach().cpu(),trained_summary.to(model.embed_summary.weight.dtype)):
            raise ValueError('Trained summary embedding failed to restore')
        # Do not use inherited generate(): custom forward intentionally disallows past_key_values.
        return model,tok,{'kind':'recurrent_summary','summary_length':config.summary_length,'accumulate_summary':True}
    sys.path.insert(0,str(ROOT/'vendor/MemGen'))
    from memgen.model.modeling_memgen import MemGenModel
    from memgen.model.configuration_memgen import MemGenConfig
    cfg=MemGenConfig.from_pretrained(path,local_files_only=True)
    if cfg.trigger_active: raise ValueError('This loader expects the published inactive-trigger Weaver checkpoint')
    def base():
        return AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    reasoner=base();weaver=base()
    reasoner.generation_config=GenerationConfig.from_model_config(reasoner.config)
    native_template=tok.chat_template
    model=MemGenModel(cfg,tok,reasoner,weaver,reasoner)
    model.trigger.model.disable_adapter_layers()
    state=torch.load(path/'pytorch_model.bin',map_location='cpu',weights_only=True)
    expected={k for k in model.state_dict() if k.startswith(('reasoner_to_weaver.','weaver_to_reasoner.')) or
              (k.startswith('weaver.') and ('.lora_' in k or not k.startswith('weaver.model.')))}
    if set(state)!=expected:
        raise ValueError(f'MemGen trainable state mismatch: missing {sorted(expected-set(state))[:5]}, extra {sorted(set(state)-expected)[:5]}')
    incompatible=model.load_state_dict(state,strict=False)
    if incompatible.unexpected_keys: raise ValueError(incompatible.unexpected_keys)
    # Same native non-thinking prompt policy as the full-text PrefEval baseline.
    tok.chat_template=native_template
    model.to(device).eval()
    return model,tok,{'kind':'memgen_weaver','trigger_active':False,'loaded_trainable_tensors':len(state),
                      'prompt_latents':cfg.prompt_latents_len,'inference_latents':cfg.inference_latents_len,
                      'max_inference_augmentations':cfg.max_inference_aug_num}

@torch.inference_mode()
def answer_reader(model,ids,mask,max_tokens,tok):
    gen=GenerationConfig(max_new_tokens=max_tokens,do_sample=False,pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id,use_cache=True)
    output=model.generate(input_ids=ids,attention_mask=mask,generation_config=gen,do_sample=False,temperature=None,top_p=None,top_k=None)
    return output[0,ids.shape[1]:].tolist()

@torch.inference_mode()
def answer_compressed(model,context_ids,query_ids,max_tokens,tok,segment_length):
    device=query_ids.device
    soft=None;segments=[]
    for start in range(0,len(context_ids),segment_length):
        chunk=context_ids[start:start+segment_length]
        carry=0 if soft is None else soft.shape[1]
        # Only hidden summary is needed; avoid materializing full-vocabulary logits for every context token.
        embeds=model.get_input_embeddings()(torch.tensor([chunk],device=device))
        if soft is None:soft=embeds[:,:0]
        _,new,_=model._forward_segment(embeds,torch.ones((1,len(chunk)),dtype=torch.long,device=device),soft,True,False,False)
        soft=torch.cat([soft,new],dim=1)
        segments.append({'text_positions':len(chunk),'carry_positions':carry,'new_summary_positions':new.shape[1],
                         'effective_positions':len(chunk)+carry+new.shape[1]})
    if soft is None: raise ValueError('Empty history')
    composed=torch.cat([soft,model.get_input_embeddings()(query_ids)],dim=1)
    past=None;outputs=[];length=composed.shape[1]
    for step in range(max_tokens):
        mask=torch.ones((1,length+step),dtype=torch.long,device=device)
        out=model.model(inputs_embeds=composed,attention_mask=mask,past_key_values=past,use_cache=True,return_dict=True)
        token=int(model.lm_head(out.last_hidden_state[:,-1]).argmax(dim=-1).item())
        outputs.append(token);past=out.past_key_values
        if token==tok.eos_token_id:break
        composed=model.get_input_embeddings()(torch.tensor([[token]],device=device))
    return outputs,{'encoder_segments':segments,'compressor_effective_positions':sum(s['effective_positions'] for s in segments),
                    'soft_input_positions':soft.shape[1],'reader_prompt':query_ids.shape[1]}

@torch.inference_mode()
def answer_memgen(model,ids,mask,max_tokens,tok):
    calls=[];handles=[]
    def hook(component):
        def before(module,args,kwargs):
            value=kwargs.get('inputs_embeds',kwargs.get('input_ids'))
            if value is None and args:value=args[0]
            if value is not None:calls.append({'component':component,'input_positions':int(value.shape[0]*value.shape[1])})
        return before
    # At the transformer level each hook counts only actually processed positions, including recomputation.
    handles.append(model.reasoner.model.register_forward_pre_hook(hook('reasoner'),with_kwargs=True))
    handles.append(model.weaver.model.base_model.model.model.register_forward_pre_hook(hook('weaver'),with_kwargs=True))
    gen=GenerationConfig(max_new_tokens=max_tokens,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
    gen.trigger_do_sample=False;gen.weaver_do_sample=False;gen.temperature=0.0
    try:
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            output,augmentation=model.generate(ids,mask,generation_config=gen,return_augmentation_mask=True)
    finally:
        for handle in handles:handle.remove()
    result=output[0,ids.shape[1]:].tolist()
    return result,{'model_forward_calls':calls,'augmentation_mask':augmentation[0].tolist(),
                   'weaver_processed_positions':sum(c['input_positions'] for c in calls if c['component']=='weaver'),
                   'reasoner_processed_positions':sum(c['input_positions'] for c in calls if c['component']=='reasoner')}
