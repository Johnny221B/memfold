from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PersonaMemExample:
    question_id: str
    shared_context_id: str
    question: str
    options: tuple[str, str, str, str]
    end_index: int
    correct_answer: str | None = None


def _parse_options(raw: str) -> tuple[str, str, str, str]:
    value = ast.literal_eval(raw)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("all_options must contain exactly four choices")
    return tuple(str(item) for item in value)  # type: ignore[return-value]


def load_questions(path: str | Path, *, include_labels: bool = False) -> list[PersonaMemExample]:
    """Load questions; labels stay inaccessible unless evaluation explicitly opts in."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "question_id",
        "shared_context_id",
        "user_question_or_message",
        "all_options",
        "end_index_in_shared_context",
        "correct_answer",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"missing PersonaMem columns: {sorted(required - set(rows[0] if rows else []))}")
    return [
        PersonaMemExample(
            question_id=row["question_id"],
            shared_context_id=row["shared_context_id"],
            question=row["user_question_or_message"],
            options=_parse_options(row["all_options"]),
            end_index=int(row["end_index_in_shared_context"]),
            correct_answer=row["correct_answer"] if include_labels else None,
        )
        for row in rows
    ]


def load_contexts(path: str | Path) -> dict[str, list[dict[str, str]]]:
    contexts: dict[str, list[dict[str, str]]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if not isinstance(record, dict) or len(record) != 1:
                raise ValueError(f"context line {line_number} must contain one id")
            context_id, messages = next(iter(record.items()))
            if context_id in contexts:
                raise ValueError(f"duplicate shared_context_id: {context_id}")
            if not isinstance(messages, list):
                raise ValueError(f"context {context_id} is not a message list")
            contexts[str(context_id)] = messages
    return contexts


def prior_context(
    example: PersonaMemExample, contexts: dict[str, list[dict[str, str]]]
) -> list[dict[str, str]]:
    messages = contexts[example.shared_context_id]
    # Upstream PersonaMem uses -1 for a legal ``context[:-1]`` prefix.
    if not -len(messages) <= example.end_index <= len(messages):
        raise ValueError(f"invalid end_index for question {example.question_id}")
    # The official benchmark explicitly defines this prefix as the legal history.
    return messages[: example.end_index]
