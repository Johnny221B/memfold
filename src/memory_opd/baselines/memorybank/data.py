from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PersonaQuestion:
    question_id: str
    shared_context_id: str
    question: str
    options: tuple[str, str, str, str]
    answer: str
    end_index: int


def _normalize_option(value: object) -> str:
    text = str(value).strip()
    if len(text) > 4 and text[0] == "(" and text[2:4] == ") ":
        return text[4:]
    return text


def load_questions(path: Path) -> dict[str, PersonaQuestion]:
    result: dict[str, PersonaQuestion] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_options = ast.literal_eval(row["all_options"])
            if not isinstance(raw_options, list) or len(raw_options) != 4:
                raise ValueError(f"question {row['question_id']} does not have four options")
            answer = row["correct_answer"].strip()
            if answer not in {"(a)", "(b)", "(c)", "(d)"}:
                raise ValueError(f"invalid answer {answer!r}")
            question = PersonaQuestion(
                question_id=row["question_id"],
                shared_context_id=row["shared_context_id"],
                question=row["user_question_or_message"].strip(),
                options=tuple(_normalize_option(x) for x in raw_options),  # type: ignore[arg-type]
                answer=answer,
                end_index=int(row["end_index_in_shared_context"]),
            )
            result[question.question_id] = question
    return result


def load_contexts(path: Path, wanted: set[str] | None = None) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict) or len(row) != 1:
                raise ValueError("each contexts JSONL row must contain one shared_context_id")
            context_id, messages = next(iter(row.items()))
            if wanted is None or context_id in wanted:
                result[context_id] = messages
    if wanted is not None and (missing := wanted - result.keys()):
        raise KeyError(f"missing contexts: {sorted(missing)[:3]}")
    return result


def load_split_ids(path: Path, partition: str) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        values = payload["question_ids"][partition]
    except KeyError as exc:
        raise KeyError(f"partition {partition!r} is absent from {path}") from exc
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate question ids in {partition}")
    return values


def prior_messages(question: PersonaQuestion, contexts: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    messages = contexts[question.shared_context_id]
    # PersonaMem uses -1 for the legal context[:-1] prefix.
    if not -len(messages) <= question.end_index <= len(messages):
        raise ValueError(f"invalid end_index for {question.question_id}")
    return messages[: question.end_index]


def select_questions(
    questions: dict[str, PersonaQuestion], ids: Iterable[str], limit: int | None = None
) -> list[PersonaQuestion]:
    selected = [questions[value] for value in ids]
    return selected if limit is None else selected[:limit]
