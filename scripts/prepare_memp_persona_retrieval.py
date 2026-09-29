#!/usr/bin/env python3
"""CPU-only frozen embedding/retrieval stage; never imports Torch or labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from memory_opd.baselines.memp.local_embedding import FastEmbedLangChainEmbeddings
from memory_opd.baselines.memp.persona import load_questions


def normalize(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("zero embedding")
    return [value / norm for value in vector]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    test_ids = json.loads(args.split.read_text())["question_ids"]["test"]
    by_id = {q.question_id: q for q in load_questions(args.questions, include_labels=False)}
    records = [json.loads(path.read_text()) for path in sorted(args.memories.glob("*.json"))]
    embedder = FastEmbedLangChainEmbeddings("thenlper/gte-large", args.embedding_cache)
    vectors = [normalize(v) for v in embedder.embed_documents([r["memory"] for r in records])]
    with args.output.open("w", encoding="utf-8") as handle:
        for question_id in test_ids:
            question = by_id[question_id]
            query = normalize(embedder.embed_query(question.question))
            candidates = []
            for record, vector in zip(records, vectors, strict=True):
                if record["context_id"] == question.context_id and record["end_index"] <= question.end_index:
                    candidates.append((sum(a*b for a,b in zip(query, vector, strict=True)), record))
            selected = sorted(candidates, key=lambda x: (-x[0], x[1]["end_index"]))[:args.top_k]
            selected.sort(key=lambda x: x[1]["end_index"])
            memory = "\n\n".join(
                f"Memory {i}: {record['memory']}" for i, (_score, record) in enumerate(selected, 1)
            )
            handle.write(json.dumps({"question_id": question_id, "memory": memory,
                "retrieved": [{"end_index": r["end_index"], "score": s} for s,r in selected]}) + "\n")


if __name__ == "__main__":
    main()
