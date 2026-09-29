"""Leakage-safe PersonaMem records for the MemGen conversation path."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

SYSTEM = (
    "Use the user's prior conversation as long-term persona memory. "
    "Answer with exactly one option: (a), (b), (c), or (d)."
)


def load_contexts(path: Path) -> dict[str, list[dict[str, str]]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or len(row) != 1:
            raise ValueError("one context ID is required per JSONL row")
        overlap = set(result).intersection(row)
        if overlap:
            raise ValueError(f"duplicate context IDs: {sorted(overlap)}")
        result.update(row)
    return result


def gold_option(answer: str, options: list[str]) -> str:
    value = answer.strip()
    if value.lower() in {"(a)", "(b)", "(c)", "(d)"}:
        return value.lower()
    if value.lower() in {"a", "b", "c", "d"}:
        return f"({value.lower()})"
    matches = [i for i, option in enumerate(options) if str(option).strip() == value]
    if len(matches) != 1:
        raise ValueError("answer does not uniquely map to one choice")
    return f"({chr(ord('a') + matches[0])})"


def render_history(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{str(message.get('role', 'unknown')).upper()}]\n{message.get('content', '')}"
        for message in messages
    )


def build_records(
    questions_path: Path, contexts_path: Path, split_path: Path, partition: str,
) -> list[dict]:
    if partition not in {"train", "val"}:
        raise ValueError("MemGen training records may only be built from train or val")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    allowed = set(split["question_ids"][partition])
    contexts = load_contexts(contexts_path)
    rows = list(csv.DictReader(questions_path.open(newline="", encoding="utf-8")))
    records = []
    for row in rows:
        if row["question_id"] not in allowed:
            continue
        options = list(ast.literal_eval(row["all_options"]))
        if len(options) != 4:
            raise ValueError("PersonaMem must have exactly four options")
        end = int(row["end_index_in_shared_context"])
        messages = contexts[row["shared_context_id"]]
        # PersonaMem explicitly uses -1 for the legal ``context[:-1]`` prefix.
        if not -len(messages) <= end <= len(messages):
            raise ValueError("invalid legal context prefix")
        choices = "\n".join(
            f"({chr(ord('a') + i)}) {option}" for i, option in enumerate(options)
        )
        user = (
            f"PRIOR CONVERSATION:\n{render_history(messages[:end])}\n\n"
            f"QUESTION:\n{row['user_question_or_message']}\n\n"
            f"OPTIONS:\n{choices}\n\nANSWER:"
        )
        records.append({
            "schema_version": "1.0",
            "trajectory_id": row["question_id"],
            "shared_context_id": row["shared_context_id"],
            "partition": partition,
            "legal_end_index": end,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": gold_option(row["correct_answer"], options)},
            ],
        })
    if len(records) != len(allowed):
        raise ValueError(f"expected {len(allowed)} {partition} records, found {len(records)}")
    if {row["trajectory_id"] for row in records} != allowed:
        raise ValueError("partition question IDs do not match the frozen split")
    return records
