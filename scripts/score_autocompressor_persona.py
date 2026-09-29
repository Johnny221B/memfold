#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_opd.shared.personamem.data import load_questions
from memory_opd.shared.personamem.evaluation import canonical_gold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    expected = set(json.loads(args.split.read_text())["question_ids"]["test"])
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    if len(rows) != len(expected) or {row["question_id"] for row in rows} != expected:
        raise ValueError("label firewall: prediction set is not complete")
    labeled = {x.question_id: x for x in load_questions(args.questions, include_labels=True)}
    correct = sum(
        row["parsed_prediction"] == canonical_gold(
            labeled[row["question_id"]].correct_answer or "", labeled[row["question_id"]].options
        ) for row in rows
    )
    result = {
        "method": "AutoCompressor-Qwen-adapted", "correct": correct, "total": len(rows),
        "accuracy": correct / len(rows),
        "strict_invalid": sum(row["parsed_prediction"] is None for row in rows),
        "mean_effective_tokens": sum(
            row["input_tokens"] + row["latent_memory_tokens"] + row["generated_tokens"] for row in rows
        ) / len(rows),
        "mean_source_context_tokens": sum(row["source_context_tokens"] for row in rows) / len(rows),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
