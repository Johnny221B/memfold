#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from memory_opd.shared.personamem.data import load_questions
from memory_opd.shared.personamem.splits import build_group_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=("PersonaMem-32K", "PersonaMem-128K"), default="PersonaMem-32K")
    args = parser.parse_args()
    examples = load_questions(args.questions, include_labels=False)
    counts = {
        "PersonaMem-32K": (489, 50, 50),
        "PersonaMem-128K": (2221, 273, 233),
    }[args.dataset]
    split = build_group_split(examples, train_count=counts[0], val_count=counts[1], test_count=counts[2], seed=args.seed)
    context_by_question = {item.question_id: item.shared_context_id for item in examples}
    payload = {
        "dataset": args.dataset,
        "split_unit": "shared_context_id",
        "seed": args.seed,
        "source_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "counts": {name: len(ids) for name, ids in split.items()},
        "question_ids": split,
        "shared_context_ids": {
            name: sorted({context_by_question[qid] for qid in ids}) for name, ids in split.items()
        },
        "label_fields_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": payload["counts"]}))


if __name__ == "__main__":
    main()
