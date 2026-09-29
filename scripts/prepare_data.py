#!/usr/bin/env python3
"""Prepare leakage-safe PersonaMem JSONL records for VERL adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from memory_opd.data.personamem import (
    load_contexts,
    load_questions,
    render_context,
    split_by_context,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--benchmark-size", choices=("32k", "128k"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = load_questions(args.questions)
    contexts = load_contexts(args.contexts)
    splits = split_by_context(examples, benchmark_size=args.benchmark_size, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    split_ids: dict[str, list[str]] = {}
    for split, rows in splits.items():
        split_ids[split] = sorted({row.shared_context_id for row in rows})
        output_path = args.output / f"personamem_{args.benchmark_size}_{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                if row.shared_context_id not in contexts:
                    raise KeyError(f"missing context {row.shared_context_id!r}")
                messages = contexts[row.shared_context_id]
                if not -len(messages) <= row.end_index_in_shared_context <= len(messages):
                    raise ValueError(
                        f"invalid end_index_in_shared_context for {row.question_id}: "
                        f"{row.end_index_in_shared_context} outside valid Python slice bounds "
                        f"[-{len(messages)}, {len(messages)}]"
                    )
                visible = messages[: row.end_index_in_shared_context]
                record = row.as_record(split=split, memory=render_context(visible))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": "1.0",
        "dataset": f"PersonaMem-v1-{args.benchmark_size}",
        "seed": args.seed,
        "questions_sha256": _sha256(args.questions),
        "contexts_sha256": _sha256(args.contexts),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "shared_context_ids": split_ids,
    }
    (args.output / f"personamem_{args.benchmark_size}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
