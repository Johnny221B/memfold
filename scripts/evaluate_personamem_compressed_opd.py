#!/usr/bin/env python3
"""Evaluate text teacher and own/shuffled/null soft-memory readers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

from memory_opd.compressed_opd import (
    CompressedOPDExample,
    deterministic_option_order,
    load_compressed_opd_examples,
    permute_options,
    soft_reader_messages,
    text_reader_messages,
)
from memory_opd.rq2_baselines.rewards import parse_choice
from on_policy_optimization_utils import (
    cached_soft_memory,
    chat_ids,
    load_bridge,
    load_policy,
    rollout_student,
    student_token_log_probs,
    teacher_token_log_probs,
)


FINAL_ANSWER = re.compile(r"(?i)final\s+answer\s*:\s*(\([a-d]\))")
LEADING_OPTION = re.compile(r"(?i)^\s*(\([a-d]\))")


def parsed_choice(text: str) -> str | None:
    matches = FINAL_ANSWER.findall(text)
    if matches:
        return matches[-1].lower()
    leading = LEADING_OPTION.match(text)
    return leading.group(1).lower() if leading else parse_choice(text)


def negative_example(
    examples: list[CompressedOPDExample], anchor: CompressedOPDExample, trial: int
) -> CompressedOPDExample:
    candidates = sorted(
        (item for item in examples if item.shared_context_id != anchor.shared_context_id),
        key=lambda item: item.question_id,
    )
    if not candidates:
        raise ValueError("evaluation requires at least two distinct contexts")
    digest = hashlib.sha256(f"{anchor.question_id}\0{trial}".encode()).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def soft_prefix(
    model: Any,
    tokenizer: Any,
    soft: torch.Tensor,
    question: str,
    options: tuple[str, str, str, str],
    device: torch.device,
    response_format: str,
) -> torch.Tensor:
    messages = soft_reader_messages(question, options)
    if response_format == "reasoning-answer":
        rendered = "\n".join(
            "({}) {}".format(chr(97 + index), text)
            for index, text in enumerate(options)
        )
        messages = [
            {"role": "system", "content": (
                "Continuous soft-memory tokens precede this conversation. "
                "Use them as the only memory evidence."
            )},
            {"role": "user", "content": (
                "Give concise evidence-grounded reasoning and end with "
                "`Final answer: (a)`, `(b)`, `(c)`, or `(d)`.\n\n"
                f"QUESTION:\n{question}\n\nOPTIONS:\n{rendered}"
            )},
        ]
    prompt = chat_ids(tokenizer, messages, device)
    return torch.cat((soft, model.get_input_embeddings()(prompt)), dim=1)


def decode_response(tokenizer: Any, response: torch.Tensor, mask: torch.Tensor) -> str:
    return tokenizer.decode(response[0][mask[0].bool()], skip_special_tokens=True).strip()


@torch.no_grad()
def teacher_generate(
    model: Any,
    tokenizer: Any,
    prompt: torch.Tensor,
    maximum_tokens: int,
) -> str:
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
    parser.add_argument("--teacher-adapter", type=Path, required=True)
    parser.add_argument("--student-adapter", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--self-memories", type=Path, required=True)
    parser.add_argument(
        "--teacher-memories",
        type=Path,
        help="Optional privileged text memories; defaults to --self-memories.",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--compressor-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", default="validation")
    parser.add_argument("--expected-examples", type=int, default=50)
    parser.add_argument("--student-device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--maximum-response-tokens", type=int, default=5)
    parser.add_argument("--expected-token-count", type=int, default=256)
    parser.add_argument("--response-format", choices=("label", "reasoning-answer"), default="label")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    if min(args.trials, args.maximum_response_tokens) <= 0:
        parser.error("trials and response token budget must be positive")

    examples = load_compressed_opd_examples(
        args.questions, args.self_memories, expected_split=args.expected_split
    )
    teacher_examples = load_compressed_opd_examples(
        args.questions,
        args.teacher_memories or args.self_memories,
        expected_split=args.expected_split,
    )
    teacher_memory_by_id = {
        item.question_id: item.memory_text for item in teacher_examples
    }
    if args.expected_examples and len(examples) != args.expected_examples:
        raise ValueError(f"expected {args.expected_examples} examples, found {len(examples)}")
    student_device = torch.device(args.student_device)
    teacher_device = torch.device(args.teacher_device)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    student = load_policy(args.model, args.student_adapter, student_device, trainable=False)
    teacher = load_policy(args.model, args.teacher_adapter, teacher_device, trainable=False)
    bridge, config = load_bridge(args.compressor_checkpoint, student_device)
    if int(config["token_count"]) != args.expected_token_count:
        raise ValueError(
            f"expected K={args.expected_token_count}, bridge has K={config['token_count']}"
        )

    rows: list[dict[str, Any]] = []
    for number, example in enumerate(examples, start=1):
        own = cached_soft_memory(args.cache_dir, example, bridge, student_device)
        for trial in range(args.trials):
            order = deterministic_option_order(example.question_id, trial)
            options, expected = permute_options(example.options, example.gold_label, order)
            negative = negative_example(examples, example, trial)
            shuffled = cached_soft_memory(args.cache_dir, negative, bridge, student_device)
            conditions = {"own": own, "shuffled": shuffled, "null": torch.zeros_like(own)}
            condition_results: dict[str, Any] = {}
            gold = torch.tensor(
                [tokenizer(expected, add_special_tokens=False).input_ids],
                device=student_device,
                dtype=torch.long,
            )
            gold_mask = torch.ones_like(gold)
            for name, soft in conditions.items():
                prefix = soft_prefix(
                    student, tokenizer, soft, example.question, options, student_device,
                    args.response_format,
                )
                response, response_mask = rollout_student(
                    student, prefix, tokenizer, args.maximum_response_tokens
                )
                text = decode_response(tokenizer, response, response_mask)
                with torch.no_grad():
                    logp = student_token_log_probs(student, prefix, gold)
                condition_results[name] = {
                    "output": text,
                    "parsed": parsed_choice(text),
                    "correct": parsed_choice(text) == expected,
                    "gold_nll": float(-(logp * gold_mask).sum() / gold_mask.sum()),
                }

            teacher_prompt = chat_ids(
                tokenizer,
                text_reader_messages(
                    teacher_memory_by_id[example.question_id],
                    example.question,
                    options,
                ),
                teacher_device,
            )
            teacher_text = teacher_generate(
                teacher, tokenizer, teacher_prompt, args.maximum_response_tokens
            )
            teacher_gold = gold.to(teacher_device)
            teacher_mask = torch.ones_like(teacher_gold)
            teacher_logp = teacher_token_log_probs(
                teacher, teacher_prompt, teacher_gold, teacher_mask
            )
            row = {
                "question_id": example.question_id,
                "question_type": example.question_type,
                "trial": trial,
                "expected": expected,
                "option_order": order,
                "negative_question_id": negative.question_id,
                "teacher": {
                    "output": teacher_text,
                    "parsed": parsed_choice(teacher_text),
                    "correct": parsed_choice(teacher_text) == expected,
                    "gold_nll": float(-teacher_logp.mean()),
                },
                **condition_results,
            }
            rows.append(row)
            print(json.dumps({"completed_question": number, "total": len(examples), **row}), flush=True)

    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary: dict[str, Any] = {
        "examples": len(examples),
        "trials": args.trials,
        "token_count": int(config["token_count"]),
        "teacher_and_student_memory_content_identical": all(
            teacher_memory_by_id[item.question_id] == item.memory_text
            for item in examples
        ),
        "teacher_memories": str(
            (args.teacher_memories or args.self_memories).resolve()
        ),
        "student_memories": str(args.self_memories.resolve()),
    }
    for condition in ("teacher", "own", "shuffled", "null"):
        summary[condition] = {
            "accuracy": sum(row[condition]["correct"] for row in rows) / len(rows),
            "parsed_rate": sum(row[condition]["parsed"] is not None for row in rows) / len(rows),
            "gold_nll": sum(row[condition]["gold_nll"] for row in rows) / len(rows),
        }
    summary["own_beats_shuffled_nll_rate"] = sum(
        row["own"]["gold_nll"] < row["shuffled"]["gold_nll"] for row in rows
    ) / len(rows)
    summary["own_beats_null_nll_rate"] = sum(
        row["own"]["gold_nll"] < row["null"]["gold_nll"] for row in rows
    ) / len(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
