#!/usr/bin/env python3
"""Stream one-question writer inputs into extraction-only Mode-A SFT rows."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from transformers import AutoTokenizer

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--writer-inputs", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    args = parser.parse_args()
    memories = read_jsonl(args.memories)
    memory_by_id = {str(row["question_id"]): row["memory_text"] for row in memories}
    if len(memories) != args.expected_records or len(memory_by_id) != len(memories):
        raise ValueError("memory count/uniqueness mismatch")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    seen = set()
    with args.writer_inputs.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip(): continue
            row = json.loads(line); question_id = str(row["task_id"])
            if question_id in seen or question_id not in memory_by_id:
                raise ValueError(f"duplicate or missing memory: {question_id}")
            seen.add(question_id); memory_text = memory_by_id[question_id]
            messages = [dict(message) for message in row["writer_messages"]]
            messages.append({"role":"assistant","content":memory_text,"reasoning_content":""})
            output = {"id":question_id,"split":"train","messages":messages,"metadata":{
                "question_id":question_id,"memory_variant":"gpt51-backbone-specific-128k-v1",
                "memory_tokens":len(tokenizer.encode(memory_text,add_special_tokens=False)),
                "assistant_loss_weights":[1.0],"scheme":"A"}}
            target.write(json.dumps(output,ensure_ascii=False)+"\n")
    if len(seen) != args.expected_records or seen != set(memory_by_id):
        raise ValueError(f"writer/memory ID mismatch: {len(seen)}")
    temporary.replace(args.output)
    print(json.dumps({"output":str(args.output),"records":len(seen)}))

if __name__ == "__main__": main()
