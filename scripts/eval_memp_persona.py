#!/usr/bin/env python3
"""Test-only MemP-style persona retrieval and strict MC generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_opd.baselines.memp.persona import load_questions, render_qa_prompt, strict_option
from memory_opd.baselines.memp.policy import QwenMemPPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--retrievals", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    test_ids = json.loads(args.split.read_text())["question_ids"]["test"]
    by_id = {q.question_id: q for q in load_questions(args.questions, include_labels=False)}
    retrievals = {row["question_id"]: row for row in (
        json.loads(line) for line in args.retrievals.read_text().splitlines()
    )}
    policy = QwenMemPPolicy(args.model, args.device, max_new_tokens=5, do_sample=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for ordinal, question_id in enumerate(test_ids, 1):
            question = by_id[question_id]
            retrieval = retrievals[question_id]
            memory = retrieval["memory"]
            prompt = render_qa_prompt(question, memory)
            prediction = policy("You answer PersonaMem multiple-choice questions using only legal prior memory.", prompt)
            input_tokens = len(policy.tokenizer.apply_chat_template(
                [{"role": "system", "content": "You answer PersonaMem multiple-choice questions using only legal prior memory."},
                 {"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True,
                enable_thinking=False,
            ))
            row = {"question_id": question_id, "prediction": prediction,
                   "parsed_prediction": strict_option(prediction), "input_tokens": input_tokens,
                   "generated_tokens": len(policy.tokenizer(prediction, add_special_tokens=False).input_ids),
                   "retrieved": retrieval["retrieved"]}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush()
            print(f"{ordinal}/{len(test_ids)} {question_id}", flush=True)


if __name__ == "__main__":
    main()
