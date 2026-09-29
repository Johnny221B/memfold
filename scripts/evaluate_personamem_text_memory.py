#!/usr/bin/env python3
"""Evaluate one Qwen checkpoint using explicit PersonaMem text memory only."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from memory_opd.compressed_opd import (
    deterministic_option_order,
    load_compressed_opd_examples,
    permute_options,
    text_reader_messages,
)
from memory_opd.rq2_baselines.rewards import parse_choice
from on_policy_optimization_utils import chat_ids, load_policy, teacher_token_log_probs


FINAL_ANSWER = re.compile(r"(?i)final\s+answer\s*:\s*(\([a-d]\))")
LEADING_OPTION = re.compile(r"(?i)^\s*(\([a-d]\))")


def parsed_choice(text: str) -> str | None:
    matches = FINAL_ANSWER.findall(text)
    if matches:
        return matches[-1].lower()
    leading = LEADING_OPTION.match(text)
    return leading.group(1).lower() if leading else parse_choice(text)


@torch.inference_mode()
def generate(model: Any, tokenizer: Any, prompt: torch.Tensor, maximum_tokens: int) -> str:
    output = model.generate(
        input_ids=prompt,
        attention_mask=torch.ones_like(prompt),
        max_new_tokens=maximum_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    return tokenizer.decode(output[0, prompt.shape[1] :], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", default="test")
    parser.add_argument("--expected-examples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--maximum-response-tokens", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="resume from output/rows.jsonl")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    if min(args.trials, args.maximum_response_tokens) <= 0:
        parser.error("trials and maximum response tokens must be positive")

    examples = load_compressed_opd_examples(
        args.questions, args.memories, expected_split=args.expected_split
    )
    if args.expected_examples and len(examples) != args.expected_examples:
        raise ValueError(f"expected {args.expected_examples} examples, found {len(examples)}")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_policy(args.model, args.adapter, device, trainable=False)

    rows: list[dict[str, Any]] = []
    rows_path = args.output / "rows.jsonl"
    if args.resume and rows_path.exists():
        with rows_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    done_trials = {(str(row["question_id"]), int(row["trial"])) for row in rows}
    args.output.mkdir(parents=True, exist_ok=True)
    append_handle = rows_path.open("a", encoding="utf-8")
    try:
      for number, example in enumerate(examples, start=1):
        for trial in range(args.trials):
            if (example.question_id, trial) in done_trials:
                continue
            order = deterministic_option_order(example.question_id, trial)
            options, expected = permute_options(example.options, example.gold_label, order)
            prompt = chat_ids(
                tokenizer,
                text_reader_messages(example.memory_text, example.question, options),
                device,
            )
            output = generate(model, tokenizer, prompt, args.maximum_response_tokens)
            parsed = parsed_choice(output)
            gold = torch.tensor(
                [tokenizer(expected, add_special_tokens=False).input_ids],
                device=device,
                dtype=torch.long,
            )
            gold_mask = torch.ones_like(gold)
            logp = teacher_token_log_probs(model, prompt, gold, gold_mask)
            row = {
                "question_id": example.question_id,
                "question_type": example.question_type,
                "trial": trial,
                "expected": expected,
                "option_order": order,
                "output": output,
                "parsed": parsed,
                "correct": parsed == expected,
                "gold_nll": float(-logp.mean()),
            }
            rows.append(row)
            append_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            append_handle.flush()
            done_trials.add((example.question_id, trial))
        print(json.dumps({"completed_question": number, "total": len(examples)}), flush=True)
    finally:
      append_handle.close()
    summary = {
        "examples": len(examples),
        "trials": args.trials,
        "model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "memories": str(args.memories.resolve()),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "parsed_rate": sum(row["parsed"] is not None for row in rows) / len(rows),
        "gold_nll": sum(row["gold_nll"] for row in rows) / len(rows),
        "maximum_response_tokens": args.maximum_response_tokens,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
