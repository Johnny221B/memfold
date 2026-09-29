#!/usr/bin/env python3
"""Build answer-blind, one-question-per-row PersonaMem writer inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from memory_opd.rq2_baselines.personamem import load_contexts, render_context


SYSTEM = """Extract compact question-conditioned evidence memory from the chronological conversation.
Use only information available up to the supplied endpoint. Do not answer the question and do not
infer options, labels, question type, or future dialogue. Output exactly one JSON object with keys
evidence, temporal_relations, and derived_facts. Preserve decisive entities, states, changes, order,
and multi-hop relations while remaining concise."""


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.questions)
    if len(rows) != args.expected_records:
        raise ValueError(f"expected {args.expected_records} questions, found {len(rows)}")
    if len({str(row["question_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate question_id")
    contexts = load_contexts(args.contexts)

    # Context/endpoint ordering improves prefix-cache locality for very long histories.
    rows.sort(key=lambda row: (
        str(row["shared_context_id"]), int(row["end_index_in_shared_context"]),
        str(row["question_id"]),
    ))
    outputs = []
    for row in rows:
        context_id = str(row["shared_context_id"])
        endpoint = int(row["end_index_in_shared_context"])
        messages = contexts[context_id]
        visible = messages[:endpoint]
        if not visible or len(visible) > len(messages):
            raise ValueError(f"invalid endpoint for {row['question_id']}: {endpoint}")
        history = render_context(visible)
        user = "HISTORY:\n" + history + "\n\nCURRENT QUESTION:\n" + str(row["question"])
        outputs.append({
            "schema_version": "personamem-text-seed-v1",
            "task_id": str(row["question_id"]),
            "split": str(row["split"]),
            "question_type": str(row.get("question_type", "")),
            "question": str(row["question"]),
            # Kept for downstream evaluation, but extractor serializes only writer_messages.
            "options": list(row["options"]),
            "gold_label": str(row["answer"]),
            "writer_messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            "glm_memory": None,
            "self_memory": "",
            "history_sha256": hashlib.sha256(history.encode()).hexdigest(),
            "shared_context_id": context_id,
            "history_end_index": endpoint,
        })

    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output), "records": len(outputs),
        "contexts": len({row["shared_context_id"] for row in outputs}),
        "answer_and_options_excluded_from_writer_messages": True,
    }))


if __name__ == "__main__":
    main()
