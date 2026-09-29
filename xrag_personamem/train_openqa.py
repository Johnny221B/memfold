#!/usr/bin/env python3
"""Train an xRAG bridge on LoCoMo open QA and select on its validation split."""

from __future__ import annotations

import argparse, json, math, random, re, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from run import Projector, inject, pretrain_batch


def messages(row, k):
    tokens = " ".join(["<xRAG>"] * k)
    return [
        {"role":"system","content":"Answer the question concisely using the compressed retrieved conversation memories. Preserve names, dates, quantities, negation, and updates. If evidence is insufficient, say so."},
        {"role":"user","content":f"Compressed retrieved memories: {tokens}\n\nQuestion: {row['question']}\n\nGive only the concise answer."},
    ]


def encode(tok, row, k, answer=True):
    prompt = tok.apply_chat_template(messages(row,k),tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if not answer: return torch.tensor(prompt)
    target = tok.encode(str(row["answer"]),add_special_tokens=False)+[tok.eos_token_id]
    return torch.tensor(prompt+target), torch.tensor([-100]*len(prompt)+target)


def qa_batch(model, bridge, tok, rows, k, device):
    enc=[encode(tok,r,k,True) for r in rows]; width=max(len(x[0]) for x in enc); es=[];ls=[];ms=[]
    xid=tok.convert_tokens_to_ids("<xRAG>")
    for r,(ids,labels) in zip(rows,enc):
        ids=ids.to(device); labels=labels.to(device)
        e=inject(model,bridge,ids,r["retrieved_vectors"][:k],xid); pad=width-len(ids)
        es.append(F.pad(e,(0,0,0,pad)));ls.append(F.pad(labels,(0,pad),value=-100));ms.append(F.pad(torch.ones(len(ids),device=device,dtype=torch.long),(0,pad)))
    return model(inputs_embeds=torch.stack(es),attention_mask=torch.stack(ms),labels=torch.stack(ls),use_cache=False).loss


def norm(x): return re.sub(r"[^a-z0-9]+"," ",x.lower()).strip()


@torch.inference_mode()
def evaluate(model,bridge,tok,rows,k,device,path):
    out=[];xid=tok.convert_tokens_to_ids("<xRAG>")
    for i,r in enumerate(rows,1):
        ids=encode(tok,r,k,False).to(device);e=inject(model,bridge,ids,r["retrieved_vectors"][:k],xid).unsqueeze(0)
        y=model.generate(inputs_embeds=e,attention_mask=torch.ones(1,len(ids),device=device,dtype=torch.long),max_new_tokens=96,do_sample=False,pad_token_id=tok.eos_token_id)
        text=tok.decode(y[0],skip_special_tokens=True).strip(); gold=norm(str(r["answer"])); hyp=norm(text)
        out.append({"question_id":r["question_id"],"question":r["question"],"reference":r["answer"],"hypothesis":text,"contains_match":gold in hyp or hyp in gold,"retrieved_texts":r["retrieved_texts"][:k]})
        if i%50==0: print(json.dumps({"eval":i,"total":len(rows),"matches":sum(x['contains_match'] for x in out)}),flush=True)
    path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in out))
    return {"n":len(out),"contains_correct":sum(x["contains_match"] for x in out),"contains_accuracy":sum(x["contains_match"] for x in out)/len(out)}


def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",type=Path,required=True);p.add_argument("--model",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--top-k",type=int,default=8);p.add_argument("--pretrain-lr",type=float,default=6e-3);p.add_argument("--qa-lr",type=float,default=1e-4);p.add_argument("--qa-epochs",type=int,default=3);p.add_argument("--seed",type=int,default=42);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True); random.seed(a.seed);torch.manual_seed(a.seed)
    cache=torch.load(a.cache,map_location="cpu",weights_only=False);tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True);tok.add_special_tokens({"additional_special_tokens":["<xRAG>"]})
    model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2").to(a.device);model.resize_token_embeddings(len(tok));model.eval();model.config.use_cache=False
    for p0 in model.parameters():p0.requires_grad_(False)
    bridge=Projector(cache["retriever_hidden_size"],model.config.hidden_size).to(a.device);docs=cache["pretrain_docs"]
    opt=torch.optim.AdamW(bridge.parameters(),lr=a.pretrain_lr);steps=math.ceil(len(docs)/12);sched=get_linear_schedule_with_warmup(opt,max(1,int(.03*steps)),steps);order=list(range(len(docs)));random.shuffle(order)
    for step,start in enumerate(range(0,len(order),12),1):
        loss=pretrain_batch(model,bridge,tok,[docs[i] for i in order[start:start+12]],a.device,180);loss.backward();opt.step();sched.step();opt.zero_grad(set_to_none=True)
        if step%50==0:print(json.dumps({"phase":"pretrain","step":step,"steps":steps,"loss":float(loss.detach())}),flush=True)
    torch.save(bridge.state_dict(),a.output/"projector-pretrain.pt")
    train=[x for x in cache["questions"] if x["split"]=="train"]; opt=torch.optim.AdamW(bridge.parameters(),lr=a.qa_lr);steps=math.ceil(len(train)/8)*a.qa_epochs;sched=get_linear_schedule_with_warmup(opt,max(1,int(.03*steps)),steps);step=0;rng=random.Random(a.seed)
    for epoch in range(a.qa_epochs):
        order=list(range(len(train)));rng.shuffle(order)
        for start in range(0,len(order),8):
            loss=qa_batch(model,bridge,tok,[train[i] for i in order[start:start+8]],a.top_k,a.device);loss.backward();opt.step();sched.step();opt.zero_grad(set_to_none=True);step+=1
            if step%50==0:print(json.dumps({"phase":"qa","step":step,"steps":steps,"loss":float(loss.detach())}),flush=True)
    torch.save(bridge.state_dict(),a.output/"projector-final.pt");val=evaluate(model,bridge,tok,[x for x in cache["questions"] if x["split"]=="validation"],a.top_k,a.device,a.output/"validation.jsonl")
    summary={"top_k":a.top_k,"qa_lr":a.qa_lr,"validation":val};(a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
