#!/usr/bin/env python3
"""Resumable greedy PersonaMem evaluation with exact reader token accounting."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

from memory_opd.rq2_baselines.rewards import parse_choice
from memory_opd.rq2_baselines.trl_adapter import load_prepared_records


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize empty token values")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row["question_id"])
            if question_id in completed:
                raise ValueError(f"duplicate prediction {question_id!r} at line {line_number}")
            usage = row.get("token_usage", {})
            required = {
                "reader_prompt",
                "reader_output",
                "reader_only_total",
                "strict_end_to_end_total",
            }
            if set(usage) != required:
                raise ValueError("existing prediction lacks exact token usage; use a new output directory")
            completed[question_id] = row
    return completed


def summarize(records: list[dict[str, Any]], adapter: Path) -> dict[str, Any]:
    totals = [int(row["token_usage"]["strict_end_to_end_total"]) for row in records]
    prompts = [int(row["token_usage"]["reader_prompt"]) for row in records]
    outputs = [int(row["token_usage"]["reader_output"]) for row in records]
    correct = sum(bool(row["correct"]) for row in records)
    return {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records),
        "token_metric": "strict_end_to_end_reader_tokens_per_question",
        "mean_token_usage": statistics.fmean(totals),
        "median_token_usage": statistics.median(totals),
        "p95_token_usage": percentile(totals, 0.95),
        "mean_prompt_tokens": statistics.fmean(prompts),
        "mean_output_tokens": statistics.fmean(outputs),
        "max_prompt_tokens": max(prompts),
        "total_token_usage": sum(totals),
        "adapter": str(adapter),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1")

    from transformers import AutoTokenizer

    rows = load_prepared_records(args.test_jsonl, allowed_splits={"test"})
    expected_ids = [str(row["question_id"]) for row in rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("test question IDs must be unique")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]
    lengths = [
        len(tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"])
        for prompt in prompts
    ]
    if max(lengths) + 5 > args.max_model_len:
        raise ValueError(
            f"test prompt requires {max(lengths) + 5} tokens, limit is {args.max_model_len}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / "predictions.jsonl"
    completed = load_completed(predictions_path)
    unexpected = set(completed) - set(expected_ids)
    if unexpected:
        raise ValueError(f"prediction file contains unexpected IDs: {sorted(unexpected)[:3]}")

    pending = [
        (row, prompt, prompt_length)
        for row, prompt, prompt_length in zip(rows, prompts, lengths)
        if str(row["question_id"]) not in completed
    ]
    if pending:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=True,
            max_lora_rank=64,
            dtype="bfloat16",
            disable_custom_all_reduce=True,
            enforce_eager=True,
        )
        sampling = SamplingParams(temperature=0.0, max_tokens=5)
        adapter = LoRARequest("rq2", 1, str(args.adapter))
        with predictions_path.open("a", encoding="utf-8") as handle:
            for row, prompt, prompt_length in pending:
                generated = llm.generate(
                    [prompt], sampling, lora_request=adapter, use_tqdm=False
                )[0]
                candidate = generated.outputs[0]
                text = candidate.text
                parsed = parse_choice(text)
                output_tokens = len(candidate.token_ids)
                total_tokens = prompt_length + output_tokens
                record = {
                    "question_id": row["question_id"],
                    "answer": row["answer"],
                    "output": text,
                    "parsed": parsed,
                    "correct": parsed == row["answer"],
                    "token_usage": {
                        "reader_prompt": prompt_length,
                        "reader_output": output_tokens,
                        "reader_only_total": total_tokens,
                        "strict_end_to_end_total": total_tokens,
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed[str(row["question_id"])] = record

    if set(completed) != set(expected_ids):
        raise RuntimeError("evaluation did not complete the expected test ID set")
    ordered = [completed[question_id] for question_id in expected_ids]
    summary = summarize(ordered, args.adapter)
    (args.output / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
