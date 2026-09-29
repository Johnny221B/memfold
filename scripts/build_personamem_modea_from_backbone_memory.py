#!/usr/bin/env python3
"""Replace Mode-A extraction targets with backbone-specific text memories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=489)
    args = parser.parse_args()

    records = read_jsonl(args.template)
    memories = read_jsonl(args.memories)
    memory_by_id = {row["question_id"]: row["memory_text"] for row in memories}
    record_ids = [row["metadata"]["question_id"] for row in records]
    if len(records) != args.expected_records or len(memory_by_id) != args.expected_records:
        raise ValueError("expected complete one-to-one Mode-A and memory inputs")
    if set(record_ids) != set(memory_by_id):
        missing = sorted(set(record_ids) - set(memory_by_id))[:5]
        extra = sorted(set(memory_by_id) - set(record_ids))[:5]
        raise ValueError(f"question-id mismatch: missing={missing}, extra={extra}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    output_rows = []
    for record in records:
        question_id = record["metadata"]["question_id"]
        memory_text = memory_by_id[question_id]
        updated = {**record, "messages": [dict(message) for message in record["messages"]]}
        assistant_indices = [
            index for index, message in enumerate(updated["messages"])
            if message["role"] == "assistant"
        ]
        if len(assistant_indices) != 1:
            raise ValueError(f"Mode A must have one assistant target: {question_id}")
        updated["messages"][assistant_indices[0]]["content"] = memory_text
        updated["metadata"] = {
            **record["metadata"],
            "memory_variant": "gpt51-backbone-specific-v1",
            "memory_tokens": len(tokenizer.encode(memory_text, add_special_tokens=False)),
            "scheme": "A",
            "assistant_loss_weights": [1.0],
        }
        output_rows.append(updated)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "records": len(output_rows)}))


if __name__ == "__main__":
    main()
