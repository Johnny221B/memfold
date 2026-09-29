#!/usr/bin/env python3
"""Prepare leakage-safe PersonaMem retrieval caches for an xRAG-style baseline."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_contexts(path: Path) -> dict[str, list[dict]]:
    merged: dict[str, list[dict]] = {}
    for row in jsonl(path):
        merged.update(row)
    return merged


def normalize_csv(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        options = ast.literal_eval(row["all_options"])
        options = [x[4:] if len(x) > 4 and x[:3] in {"(a)", "(b)", "(c)", "(d)"} else x for x in options]
        result[row["question_id"]] = {
            "question_id": row["question_id"],
            "shared_context_id": row["shared_context_id"],
            "end_index_in_shared_context": int(row["end_index_in_shared_context"]),
            "question_type": row["question_type"],
            "question": row["user_question_or_message"],
            "options": options,
            "answer": row["correct_answer"],
        }
    return result


def split_ids_32k() -> dict[str, set[str]]:
    base = ROOT / "soft_recon_poc_runs/inputs/personamem32k_official_memories_existing_v1"
    return {
        split: {str(x["question_id"]) for x in jsonl(base / f"personamem_32k_{split}_memories.jsonl")}
        for split in ("train", "validation", "test")
    }


def load_128k() -> list[dict]:
    base = ROOT / "soft_recon_poc_runs/personamem128k_reader_initialization_from_full128k_e3_v1/inputs"
    values = []
    for split, count in (("train", 8), ("validation", 4)):
        for shard in range(count):
            values.extend(jsonl(base / split / f"questions-{shard}.jsonl"))
    test = ROOT / "soft_recon_poc_runs/personamem128k_two_stage_sft_v1/data/test-questions.jsonl"
    values.extend(jsonl(test))
    return values


def embed_texts(model, tokenizer, texts: list[str], device: str, batch_size: int) -> tuple[torch.Tensor, list[int]]:
    vectors, lengths = [], []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        lengths.extend(encoded["attention_mask"].sum(1).tolist())
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = F.normalize(pooled.float(), dim=-1)
        vectors.append(pooled.cpu().half())
    return torch.cat(vectors), [int(x) for x in lengths]


def message_text(message: dict) -> str:
    return f"{message.get('role', 'unknown')}: {message.get('content', '')}".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["32k", "128k"], required=True)
    parser.add_argument("--retriever", type=Path, default=ROOT / "models/gte-large")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=16)
    args = parser.parse_args()

    if args.dataset == "32k":
        by_id = normalize_csv(ROOT / "personamem_v1_raw/questions_32k.csv")
        ids = split_ids_32k()
        questions = []
        for split in ("train", "validation", "test"):
            for qid in ids[split]:
                row = dict(by_id[qid]); row["split"] = split; questions.append(row)
        contexts = load_contexts(ROOT / "personamem_v1_raw/shared_contexts_32k.jsonl")
    else:
        questions = load_128k()
        contexts = load_contexts(ROOT / "personamem_v1_raw/shared_contexts_128k.jsonl")

    assert len({x["question_id"] for x in questions}) == len(questions)
    split_contexts = {
        split: {x["shared_context_id"] for x in questions if x["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert not (split_contexts["train"] & split_contexts["validation"])
    assert not (split_contexts["train"] & split_contexts["test"])

    tokenizer = AutoTokenizer.from_pretrained(args.retriever, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.retriever, trust_remote_code=True, torch_dtype=torch.bfloat16).to(args.device).eval()

    context_cache = {}
    for number, context_id in enumerate(sorted({x["shared_context_id"] for x in questions}), 1):
        texts = [message_text(x) for x in contexts[context_id]]
        vectors, token_lengths = embed_texts(model, tokenizer, texts, args.device, args.batch_size)
        context_cache[context_id] = {"texts": texts, "vectors": vectors, "token_lengths": token_lengths}
        if number % 10 == 0:
            print(json.dumps({"contexts": number, "total": len(contexts)}), flush=True)

    q_texts = [x["question"] for x in questions]
    qo_texts = [x["question"] + "\n" + "\n".join(x["options"]) for x in questions]
    q_vectors, q_lengths = embed_texts(model, tokenizer, q_texts, args.device, args.batch_size)
    qo_vectors, qo_lengths = embed_texts(model, tokenizer, qo_texts, args.device, args.batch_size)

    prepared = []
    for row, qvec, qovec, qlen, qolen in zip(questions, q_vectors, qo_vectors, q_lengths, qo_lengths):
        cache = context_cache[row["shared_context_id"]]
        # PersonaMem's audited visibility contract is the Python slice
        # messages[:end_index_in_shared_context].  In particular, -1 means all
        # but the final message, rather than an empty prefix.
        raw_end = int(row["end_index_in_shared_context"])
        end = len(cache["texts"][:raw_end])
        k = min(args.top_k, end)
        item = dict(row)
        for name, vector in (("question", qvec), ("question_options", qovec)):
            scores = cache["vectors"][:end].float() @ vector.float()
            indices = torch.topk(scores, k=k).indices.tolist()
            item[f"retrieved_{name}_vectors"] = cache["vectors"][indices]
            item[f"retrieved_{name}_texts"] = [cache["texts"][i] for i in indices]
            item[f"retrieved_{name}_scores"] = [float(scores[i]) for i in indices]
        item["query_tokens_question"] = qlen
        item["query_tokens_question_options"] = qolen
        item["context_encoder_tokens"] = sum(cache["token_lengths"][:end])
        prepared.append(item)

    train_docs = []
    for context_id in sorted(split_contexts["train"]):
        cache = context_cache[context_id]
        for text, vector, length in zip(cache["texts"], cache["vectors"], cache["token_lengths"]):
            train_docs.append({"text": text, "vector": vector, "tokens": length})

    payload = {
        "schema": "xrag-personamem-cache-v1",
        "dataset": args.dataset,
        "retriever": str(args.retriever),
        "retriever_hidden_size": int(q_vectors.shape[-1]),
        "questions": prepared,
        "pretrain_docs": train_docs,
        "counts": {split: sum(x["split"] == split for x in questions) for split in ("train", "validation", "test")},
        "context_counts": {split: len(split_contexts[split]) for split in split_contexts},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"output": str(args.output), "questions": len(prepared), "pretrain_docs": len(train_docs), **payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()
