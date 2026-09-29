#!/usr/bin/env python3
"""Reconstruct per-trial token use for a final PersonaMem soft-memory run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from memory_opd.compressed_opd import permute_options, soft_reader_messages


MEMORY_BUDGET = (
    "\n\nOUTPUT BUDGET: Return no more than 8 evidence items, 4 temporal_relations "
    "items, and 4 derived_facts items. Keep every item at most 160 characters. "
    "Finish the complete JSON object within 1200 tokens."
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--writer-inputs", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--eval-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=50)
    parser.add_argument("--expected-trials", type=int, default=5)
    parser.add_argument("--soft-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    questions = {str(row["question_id"]): row for row in read_jsonl(args.questions)}
    writers = {str(row["task_id"]): row for row in read_jsonl(args.writer_inputs)}
    memories = {str(row["question_id"]): row for row in read_jsonl(args.memories)}
    eval_rows = read_jsonl(args.eval_rows)
    expected_ids = set(questions)
    if len(expected_ids) != args.expected_questions:
        raise ValueError(f"expected {args.expected_questions} questions, found {len(expected_ids)}")
    if set(writers) != expected_ids or set(memories) != expected_ids:
        raise ValueError("question/writer/memory IDs differ")

    static: dict[str, dict[str, int]] = {}
    for question_id in sorted(expected_ids):
        writer_messages = [dict(message) for message in writers[question_id]["writer_messages"]]
        writer_messages[0]["content"] = str(writer_messages[0]["content"]) + MEMORY_BUDGET
        writer_tokens = tokenizer.apply_chat_template(
            writer_messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        memory = memories[question_id]
        memory_text = str(memory["memory_text"])
        static[question_id] = {
            "writer_text_input_tokens": len(writer_tokens),
            "writer_generated_memory_tokens": int(memory["memory_tokens"]),
            "compressor_text_input_tokens": len(
                tokenizer(memory_text, add_special_tokens=True).input_ids
            ),
        }

    audited = []
    for row in eval_rows:
        question_id = str(row["question_id"])
        question = questions[question_id]
        options, expected = permute_options(
            question["options"], question["answer"], row["option_order"]
        )
        if expected != row["expected"]:
            raise ValueError(f"permutation mismatch: {question_id} trial {row['trial']}")
        answer_prompt = tokenizer.apply_chat_template(
            soft_reader_messages(str(question["question"]), options),
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        generated_visible = len(
            tokenizer(str(row["own"]["output"]), add_special_tokens=False).input_ids
        )
        item = {
            "question_id": question_id,
            "trial": int(row["trial"]),
            **static[question_id],
            "answer_text_input_tokens": len(answer_prompt),
            "soft_memory_tokens": args.soft_tokens,
            "answer_effective_input_positions": len(answer_prompt) + args.soft_tokens,
            "answer_generated_visible_tokens": generated_visible,
            "answer_total_effective_tokens": len(answer_prompt) + args.soft_tokens + generated_visible,
        }
        item["end_to_end_token_equivalents"] = (
            item["writer_text_input_tokens"]
            + item["writer_generated_memory_tokens"]
            + item["compressor_text_input_tokens"]
            + item["answer_total_effective_tokens"]
        )
        audited.append(item)

    trials = sorted({row["trial"] for row in audited})
    if trials != list(range(args.expected_trials)) or any(
        sum(row["trial"] == trial for row in audited) != args.expected_questions
        for trial in trials
    ):
        raise ValueError(
            f"expected {args.expected_trials} complete, separate trials"
        )
    fields = [
        "writer_text_input_tokens", "writer_generated_memory_tokens",
        "compressor_text_input_tokens", "answer_text_input_tokens",
        "soft_memory_tokens", "answer_effective_input_positions",
        "answer_generated_visible_tokens", "answer_total_effective_tokens",
        "end_to_end_token_equivalents",
    ]
    by_trial = []
    for trial in trials:
        selected = [row for row in audited if row["trial"] == trial]
        by_trial.append({
            "trial": trial, "questions": len(selected),
            **{f"mean_{field}": mean(selected, field) for field in fields},
        })
    result = {
        "schema_version": "personamem-soft-token-usage-v1",
        "model": str(args.model.resolve()),
        "questions": args.expected_questions,
        "trials_reported_separately": True,
        "soft_memory_tokens_per_answer": args.soft_tokens,
        "generated_answer_tokens_exclude_unobservable_eos": True,
        "online_answer_formula": "answer_text_input_tokens + soft_memory_tokens + answer_generated_visible_tokens",
        "end_to_end_formula": "writer_text_input_tokens + writer_generated_memory_tokens + compressor_text_input_tokens + answer_total_effective_tokens",
        "by_trial": by_trial,
        "rows": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({**{k: v for k, v in result.items() if k != "rows"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
