#!/usr/bin/env python3
"""Generate answer-blind full-text-memory reasoning traces for PersonaMem."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_opd.soft_reconstruction import serialize_memory


FINAL_ANSWER = re.compile(r"(?i)(?:final\s+answer|answer)\s*:\s*(\([a-d]\))")
ANY_OPTION = re.compile(r"\([a-d]\)", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_user(question: dict[str, Any], memory: dict[str, Any]) -> str:
    options = "\n".join(
        f"({chr(ord('a') + index)}) {text}"
        for index, text in enumerate(question["options"])
    )
    return (
        "Use the supplied memory to reason through the multiple-choice question. "
        "Give 2 to 4 concise, visible reasoning steps: identify the relevant memory evidence, connect it to the question, and eliminate inconsistent alternatives. "
        "End with `Final answer: (a)`, `(b)`, `(c)`, or `(d)`.\n\n"
        f"MEMORY:\n{serialize_memory(memory)}\n\n"
        f"QUESTION:\n{question['question']}\n\nOPTIONS:\n{options}"
    )


def split_reasoning_and_answer(generation: str) -> tuple[str, str | None]:
    matches = list(FINAL_ANSWER.finditer(generation))
    if matches:
        match = matches[-1]
        return generation[: match.start()].rstrip(), match.group(1).lower()
    options = list(ANY_OPTION.finditer(generation))
    answer = options[-1].group(0).lower() if options else None
    return generation.strip(), answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.shards <= 0 or not 0 <= args.shard_index < args.shards:
        parser.error("--shard-index must be in [0, --shards)")
    if min(args.batch_size, args.max_new_tokens) <= 0:
        parser.error("batch size and max new tokens must be positive")

    questions = read_jsonl(args.questions)
    memory_rows = read_jsonl(args.memories)
    memories = {str(row["question_id"]): row["memory"] for row in memory_rows}
    if len(memories) != len(memory_rows):
        raise ValueError("duplicate memory question_id")
    selected = questions[args.shard_index :: args.shards]
    if any(str(row["question_id"]) not in memories for row in selected):
        raise ValueError("missing memory for selected question")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.output.exists():
        completed = {str(row["question_id"]) for row in read_jsonl(args.output)}
    selected = [row for row in selected if str(row["question_id"]) not in completed]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model = model.to(args.device).eval().requires_grad_(False)

    for offset in range(0, len(selected), args.batch_size):
        batch = selected[offset : offset + args.batch_size]
        prompts = []
        for row in batch:
            question_id = str(row["question_id"])
            messages = [
                {
                    "role": "system",
                    "content": "You are a careful memory-grounded reasoning assistant.",
                },
                {"role": "user", "content": render_user(row, memories[question_id])},
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[:, encoded.input_ids.shape[1] :]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=False)
        with args.output.open("a", encoding="utf-8") as handle:
            for row, token_ids, text in zip(batch, generated, texts, strict=True):
                eos = tokenizer.eos_token or ""
                cleaned = text.split(eos, 1)[0].strip() if eos else text.strip()
                reasoning, predicted = split_reasoning_and_answer(cleaned)
                record = {
                    "question_id": str(row["question_id"]),
                    "reasoning": reasoning,
                    "generation": cleaned,
                    "predicted_answer": predicted,
                    "gold_answer_for_audit_only": row.get("answer"),
                    "answer_correct_for_audit_only": predicted == row.get("answer"),
                    "generated_tokens": int(token_ids.ne(tokenizer.pad_token_id).sum()),
                    "finish_reason": (
                        "eos" if tokenizer.eos_token_id in token_ids.tolist() else "length"
                    ),
                    "teacher_condition": "question_options_and_full_text_memory",
                    "gold_answer_sent_to_teacher": False,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    "shard": args.shard_index,
                    "completed": min(offset + len(batch), len(selected)),
                    "selected": len(selected),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
