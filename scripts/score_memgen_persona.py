#!/usr/bin/env python3
"""Open PersonaMem labels only after a sealed MemGen prediction file is complete."""

import argparse
import ast
import csv
import json
from pathlib import Path

from memory_opd.baselines.memgen.persona_data import gold_option


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output path must be new")
    expected = json.loads(args.split.read_text(encoding="utf-8"))["question_ids"]["test"]
    if args.allow_limit is not None:
        expected = expected[: args.allow_limit]
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()]
    if [row["question_id"] for row in rows] != expected:
        raise ValueError("sealed predictions are incomplete or out of frozen order")
    # Completeness gate passed; labels may now be opened.
    labels = {}
    with args.questions.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["question_id"] in set(expected):
                labels[row["question_id"]] = gold_option(
                    row["correct_answer"], list(ast.literal_eval(row["all_options"]))
                )
    correct = sum(row["parsed_prediction"] == labels[row["question_id"]] for row in rows)
    invalid = sum(row["parsed_prediction"] is None for row in rows)
    overflow = sum(row["status"] == "overflow" for row in rows)
    tokens = sum(row["input_tokens"] + row.get("generated_tokens", 0) for row in rows)
    result = {
        "correct": correct, "total": len(rows), "accuracy": correct / len(rows),
        "strict_invalid": invalid, "overflow": overflow,
        "mean_total_tokens": tokens / len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
