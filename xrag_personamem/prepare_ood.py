#!/usr/bin/env python3
"""Prepare frozen-GTE retrieval caches for PrefEval, LoCoMo, and LongMemEval."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(x) for x in handle if x.strip()]


def chunks(text: str, tokenizer, limit: int = 480, prefix: str = "") -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    return [(prefix + tokenizer.decode(ids[i : i + limit])).strip() for i in range(0, len(ids), limit)]


def embed(model, tokenizer, texts: list[str], device: str, batch_size: int = 192):
    vectors, lengths = [], []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        lengths.extend(int(x) for x in encoded["attention_mask"].sum(1))
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = F.normalize(((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).float(), dim=-1)
        vectors.append(pooled.cpu().half())
    return torch.cat(vectors), lengths


def retrieve(docs: list[str], vectors: torch.Tensor, query: str, model, tokenizer, device: str, top_k: int):
    qvec, qlens = embed(model, tokenizer, [query], device, 1)
    scores = vectors.float() @ qvec[0].float()
    k = min(top_k, len(docs))
    idx = torch.topk(scores, k).indices.tolist()
    return {
        "retrieved_vectors": vectors[idx],
        "retrieved_texts": [docs[i] for i in idx],
        "retrieved_scores": [float(scores[i]) for i in idx],
        "query_encoder_tokens": int(qlens[0]),
    }


def prefeval(tokenizer) -> tuple[list[dict], dict[str, list[str]], set[str]]:
    rows = [x for x in read_jsonl(ROOT / "prefeval_pipeline/prepared/inputs.jsonl") if x["method"] == "full_text"]
    docs = {}
    questions = []
    for row in rows:
        qid = str(row["id"])
        values = []
        for message in row["messages"][1:-1]:
            role = message.get("role", "unknown").upper()
            values.extend(chunks(message.get("content", ""), tokenizer, prefix=f"[{role}]\n"))
        docs[qid] = values
        questions.append({"question_id": qid, "question": row["question"], "topic": row["topic"], "preference": row["preference"], "split": "test", "context_id": qid})
    return questions, docs, set()


def locomo(tokenizer) -> tuple[list[dict], dict[str, list[str]], set[str]]:
    sessions = read_jsonl(ROOT / "locomo_pipeline/prepared/full_v3_20260906/source_sessions.jsonl")
    docs: dict[str, list[str]] = defaultdict(list)
    train_docs = set()
    for session in sessions:
        values = chunks(session["input"], tokenizer)
        docs[session["context_id"]].extend(values)
        if session["split"] == "train":
            train_docs.update(values)
    questions = []
    for split in ("train", "validation"):
        path = ROOT / f"locomo_pipeline/prepared/full_v3_20260906/{split}/qa.jsonl"
        for row in read_jsonl(path):
            questions.append({
                "question_id": row["question_id"], "question": row["question"], "answer": row["answer"],
                "question_type": row.get("question_type"), "split": split, "context_id": row["context_id"],
            })
    return questions, dict(docs), train_docs


def longmemeval(tokenizer) -> tuple[list[dict], dict[str, list[str]], set[str]]:
    source = json.loads((ROOT / "datasets/longmemeval/longmemeval_s_cleaned.json").read_text())
    docs, questions = {}, []
    for row in source:
        values = []
        for date, session in zip(row["haystack_dates"], row["haystack_sessions"], strict=True):
            text = "\n\n".join(f"[{m.get('role','unknown').upper()}]\n{m.get('content','')}" for m in session)
            values.extend(chunks(text, tokenizer, prefix=f"[SESSION_TIME] {date}\n"))
        docs[row["question_id"]] = values
        questions.append({
            "question_id": row["question_id"], "question": row["question"], "question_date": row["question_date"],
            "question_type": row["question_type"], "answer": row["answer"], "split": "test", "context_id": row["question_id"],
        })
    return questions, docs, set()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["prefeval", "locomo", "longmemeval"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--retriever", type=Path, default=ROOT / "models/gte-large")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top-k", type=int, default=16)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.retriever, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.retriever, trust_remote_code=True, torch_dtype=torch.bfloat16).to(args.device).eval()
    loader = {"prefeval": prefeval, "locomo": locomo, "longmemeval": longmemeval}[args.dataset]
    questions, docs_by_context, train_doc_set = loader(tokenizer)
    context_cache = {}
    for number, (context_id, docs) in enumerate(docs_by_context.items(), 1):
        vectors, lengths = embed(model, tokenizer, docs, args.device)
        context_cache[context_id] = (docs, vectors, lengths)
        if number % 50 == 0:
            print(json.dumps({"contexts": number, "total": len(docs_by_context)}), flush=True)
    prepared = []
    for number, row in enumerate(questions, 1):
        docs, vectors, lengths = context_cache[row["context_id"]]
        query = row["question"]
        if row.get("question_date"):
            query += "\nQuestion date: " + row["question_date"]
        item = dict(row); item.update(retrieve(docs, vectors, query, model, tokenizer, args.device, args.top_k))
        item["context_encoder_tokens"] = sum(int(x) for x in lengths)
        prepared.append(item)
        if number % 100 == 0:
            print(json.dumps({"retrieved": number, "total": len(questions)}), flush=True)
    pretrain_docs = []
    if train_doc_set:
        for context_id, (docs, vectors, lengths) in context_cache.items():
            for text, vector, length in zip(docs, vectors, lengths):
                if text in train_doc_set:
                    pretrain_docs.append({"text": text, "vector": vector, "tokens": int(length)})
    payload = {"schema": "xrag-ood-cache-v1", "dataset": args.dataset, "retriever_hidden_size": 1024,
               "questions": prepared, "pretrain_docs": pretrain_docs,
               "counts": {s: sum(x["split"] == s for x in prepared) for s in ("train", "validation", "test")}}
    args.output.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, args.output)
    print(json.dumps({"output": str(args.output), "questions": len(prepared), "pretrain_docs": len(pretrain_docs), **payload["counts"]}, indent=2))


if __name__ == "__main__": main()
