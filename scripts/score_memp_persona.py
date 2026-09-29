#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_opd.baselines.memp.persona import load_questions


def gold_label(answer: str, options: tuple[str, str, str, str]) -> str:
    value = answer.strip()
    if value.lower() in {"(a)", "(b)", "(c)", "(d)"}:
        return value.lower()
    if value.lower() in {"a", "b", "c", "d"}:
        return f"({value.lower()})"
    matches = [index for index, option in enumerate(options) if option.strip() == value]
    if len(matches) != 1:
        raise ValueError("gold answer does not map uniquely to an option")
    return f"({chr(ord('a') + matches[0])})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = set(json.loads(args.split.read_text())["question_ids"]["test"])
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    if len(rows) != len(expected) or {row["question_id"] for row in rows} != expected:
        raise ValueError("refusing to open labels before the complete test prediction set exists")
    # Label firewall opens only after the completeness check above.
    labeled = {q.question_id: q for q in load_questions(args.questions, include_labels=True)}
    correct = sum(row["parsed_prediction"] == gold_label(
        labeled[row["question_id"]].answer or "", labeled[row["question_id"]].options
    ) for row in rows)
    result = {
        "method": "MemP-style Persona Memory", "correct": correct, "total": len(rows),
        "accuracy": correct / len(rows), "strict_invalid": sum(row["parsed_prediction"] is None for row in rows),
        "mean_total_tokens": sum(row["input_tokens"] + row["generated_tokens"] for row in rows) / len(rows),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
