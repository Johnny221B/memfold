"""PersonaMem adapter for the MemP direct-build/retrieve/use skeleton."""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PersonaQuestion:
    question_id: str
    context_id: str
    question: str
    options: tuple[str, str, str, str]
    end_index: int
    answer: str | None = None


def load_questions(path: Path, *, include_labels: bool = False) -> list[PersonaQuestion]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        options = ast.literal_eval(row["all_options"])
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError("PersonaMem questions require exactly four options")
        result.append(PersonaQuestion(
            question_id=row["question_id"], context_id=row["shared_context_id"],
            question=row["user_question_or_message"], options=tuple(map(str, options)),
            end_index=int(row["end_index_in_shared_context"]),
            answer=row["correct_answer"] if include_labels else None,
        ))
    return result


def load_contexts(path: Path) -> dict[str, list[dict[str, str]]]:
    contexts = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not isinstance(record, dict) or len(record) != 1:
                raise ValueError("each context row must contain one shared_context_id")
            context_id, messages = next(iter(record.items()))
            contexts[context_id] = messages
    return contexts


def session_slices(messages: list[dict[str, str]], end_index: int) -> list[tuple[int, int]]:
    """Return session ranges within the legal prefix; system turns start sessions."""
    if not -len(messages) <= end_index <= len(messages):
        raise ValueError("end_index is outside the context")
    if end_index < 0:
        end_index = len(messages) + end_index
    starts = [index for index, message in enumerate(messages[:end_index]) if message.get("role") == "system"]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    return [(start, starts[i + 1] if i + 1 < len(starts) else end_index) for i, start in enumerate(starts)]


def render_session(messages: list[dict[str, str]], start: int, end: int) -> str:
    return "\n\n".join(
        f"[{message.get('role', 'unknown').upper()}]\n{message.get('content', '')}"
        for message in messages[start:end]
    )


def render_qa_prompt(question: PersonaQuestion, memory: str) -> str:
    options = "\n".join(
        f"({chr(ord('a') + index)}) {option}" for index, option in enumerate(question.options)
    )
    return (
        f"QUESTION:\n{question.question}\n\nRETRIEVED MEMORY:\n{memory}\n\n"
        "Output exactly one option: (a), (b), (c), or (d), and nothing else.\n\n"
        f"OPTIONS:\n{options}\n\nANSWER:"
    )


def strict_option(text: str) -> str | None:
    value = text.strip()
    return value if value in {"(a)", "(b)", "(c)", "(d)"} else None
