#!/usr/bin/env python3
"""Generate PrefEval or LongMemEval answers from a frozen xRAG bridge."""

from __future__ import annotations

import argparse, copy, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from run import Projector, inject


def system(dataset):
    if dataset=="prefeval": return "Answer helpfully and personalize the response using compressed memories of this target user. Respect relevant preferences without mechanically restating them. Do not invent preferences."
    return "Answer concisely and completely using compressed target-conversation memories. Preserve names, dates, quantities, negation, and updates. If evidence is insufficient, say so."


def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",type=Path,required=True);p.add_argument("--model",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--top-k",type=int,default=8);p.add_argument("--control",choices=["own","shuffled","null","no_memory"],default="own");p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=16);p.add_argument("--num-shards",type=int,default=1);p.add_argument("--shard-index",type=int,default=0);a=p.parse_args()
    if not 0 <= a.shard_index < a.num_shards: raise ValueError("invalid shard index")
    cache=torch.load(a.cache,map_location="cpu",weights_only=False);all_rows=[copy.deepcopy(x) for x in cache["questions"] if x["split"]=="test"]
    # Deterministic modulo sharding permits multi-GPU evaluation without
    # changing the generated sample order after shards are merged.
    rows=[row for i,row in enumerate(all_rows) if i%a.num_shards==a.shard_index]
    if a.control=="shuffled":
        source=list(rows);random.Random(17).shuffle(source)
        for row,other in zip(rows,source):
            row["retrieved_vectors"]=other["retrieved_vectors"]
            row["retrieved_texts"]=other["retrieved_texts"]
    elif a.control=="null":
        for row in rows:row["retrieved_vectors"]=torch.zeros_like(row["retrieved_vectors"])
    elif a.control=="no_memory":a.top_k=0
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True);tok.add_special_tokens({"additional_special_tokens":["<xRAG>"]});model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2").to(a.device);model.resize_token_embeddings(len(tok));model.eval()
    bridge=Projector(cache["retriever_hidden_size"],model.config.hidden_size).to(a.device);bridge.load_state_dict(torch.load(a.checkpoint,map_location=a.device,weights_only=True));bridge.eval();xid=tok.convert_tokens_to_ids("<xRAG>");a.output.parent.mkdir(parents=True,exist_ok=True);done=[]
    with torch.inference_mode(),a.output.open("w") as f:
        for start in range(0,len(rows),a.batch_size):
            batch=rows[start:start+a.batch_size]; embeddings=[];lengths=[];actual_ks=[]
            for row in batch:
                actual_k=min(a.top_k,len(row["retrieved_vectors"]));actual_ks.append(actual_k);tokens=" ".join(["<xRAG>"]*actual_k);date=f"\nQuestion date: {row['question_date']}" if row.get("question_date") else ""
                user=f"Compressed memories: {tokens}{date}\n\nQuestion: {row['question']}"
                ids=torch.tensor(tok.apply_chat_template([{"role":"system","content":system(cache["dataset"])},{"role":"user","content":user}],tokenize=True,add_generation_prompt=True,enable_thinking=False),device=a.device)
                embeddings.append(inject(model,bridge,ids,row["retrieved_vectors"][:actual_k],xid));lengths.append(len(ids))
            width=max(lengths); padded=[];masks=[]
            for value,length in zip(embeddings,lengths):
                pad=width-length;padded.append(F.pad(value,(0,0,pad,0)));masks.append(F.pad(torch.ones(length,device=a.device,dtype=torch.long),(pad,0)))
            limit=300 if cache["dataset"]=="prefeval" else 128
            ys=model.generate(inputs_embeds=torch.stack(padded),attention_mask=torch.stack(masks),max_new_tokens=limit,do_sample=False,pad_token_id=tok.eos_token_id)
            for row,y,length,actual_k in zip(batch,ys,lengths,actual_ks):
                values=y.tolist(); eos=values.index(tok.eos_token_id)+1 if tok.eos_token_id in values else len(values);values=values[:eos];text=tok.decode(values,skip_special_tokens=True).strip()
                query_cost=0 if a.control=="no_memory" else row["query_encoder_tokens"]
                result={"question_id":row["question_id"],"question":row["question"],"hypothesis":text,"input_tokens":length+query_cost,"output_tokens":len(values),"control":a.control,"retrieved_texts":row["retrieved_texts"][:actual_k]}
                if cache["dataset"]=="prefeval":result.update({"id":int(row["question_id"]),"method":"xrag_"+a.control,"topic":row["topic"],"preference":row["preference"],"response":text})
                else:result.update({"reference":row["answer"],"question_type":row["question_type"]})
                f.write(json.dumps(result,ensure_ascii=False)+"\n");done.append(result)
            f.flush();print(json.dumps({"complete":len(done),"total":len(rows)}),flush=True)
    summary={"dataset":cache["dataset"],"control":a.control,"questions":len(done),"mean_online_tokens":sum(x["input_tokens"]+x["output_tokens"] for x in done)/len(done)};(a.output.parent/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
