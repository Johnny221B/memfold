"""Contracts and prompts for PersonaMem self-memory compressed OPD."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MEMORY_KEYS = ("evidence", "temporal_relations", "derived_facts")
OPTION_LABELS = ("(a)", "(b)", "(c)", "(d)")
TEXT_READER_SYSTEM = (
    "Answer the multiple-choice question using only the supplied user memory. "
    "Return exactly one option label."
)
SOFT_READER_SYSTEM = (
    "Continuous soft-memory tokens precede this conversation. Use only that memory "
    "and the visible question. Return exactly one option label."
)


@dataclass(frozen=True)
class CompressedOPDExample:
    question_id: str
    shared_context_id: str
    split: str
    question_type: str
    question: str
    options: tuple[str, str, str, str]
    gold_label: str
    memory: dict[str, list[str]]
    memory_text: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def canonical_memory(value: Mapping[str, Any]) -> dict[str, list[str]]:
    """Validate and normalize the evidence-v1 memory schema."""

    if set(value) != set(MEMORY_KEYS):
        raise ValueError(f"memory must contain exactly {MEMORY_KEYS}")
    result: dict[str, list[str]] = {}
    for key in MEMORY_KEYS:
        items = value[key]
        if not isinstance(items, list) or not all(
            isinstance(item, str) and item.strip() for item in items
        ):
            raise ValueError(f"memory.{key} must contain non-empty strings")
        result[key] = [item.strip() for item in items]
    return result


def serialize_memory(value: Mapping[str, Any]) -> str:
    return json.dumps(canonical_memory(value), ensure_ascii=False, separators=(",", ":"))


def parse_memory_text(text: str) -> dict[str, list[str]]:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("memory is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("memory must be a JSON object")
    return canonical_memory(value)


def rendered_options(options: Sequence[str]) -> str:
    if len(options) != len(OPTION_LABELS):
        raise ValueError("PersonaMem requires exactly four options")
    return "\n".join(f"{label} {text}" for label, text in zip(OPTION_LABELS, options))


def text_reader_messages(
    memory_text: str, question: str, options: Sequence[str]
) -> list[dict[str, str]]:
    """Privileged teacher view containing the explicit self-generated memory."""

    parse_memory_text(memory_text)
    return [
        {"role": "system", "content": TEXT_READER_SYSTEM},
        {"role": "user", "content": f"QUESTION:\n{question}"},
        {"role": "assistant", "content": memory_text, "reasoning_content": ""},
        {
            "role": "user",
            "content": f"Use the memory above to answer.\n\nOPTIONS:\n{rendered_options(options)}",
        },
    ]


def soft_reader_messages(question: str, options: Sequence[str]) -> list[dict[str, str]]:
    """Student view; explicit text memory is intentionally absent."""

    return [
        {"role": "system", "content": SOFT_READER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{question}\n\nOPTIONS:\n{rendered_options(options)}\n\n"
                "Use the preceding soft memory and output exactly one option label."
            ),
        },
    ]


def deterministic_option_order(question_id: str, trial: int) -> tuple[int, int, int, int]:
    """Return a stable option permutation without consuming process RNG state."""

    ranked = sorted(
        range(4),
        key=lambda index: hashlib.sha256(
            f"{question_id}\0{trial}\0{index}".encode()
        ).digest(),
    )
    return tuple(ranked)  # type: ignore[return-value]


def permute_options(
    options: Sequence[str], gold_label: str, order: Sequence[int]
) -> tuple[tuple[str, str, str, str], str]:
    if gold_label not in OPTION_LABELS or sorted(order) != list(range(4)):
        raise ValueError("invalid gold label or option permutation")
    gold_index = OPTION_LABELS.index(gold_label)
    values = tuple(options[index] for index in order)
    return values, OPTION_LABELS[list(order).index(gold_index)]  # type: ignore[return-value]


def load_compressed_opd_examples(
    questions_path: Path,
    self_memories_path: Path,
    *,
    expected_split: str | None = None,
) -> list[CompressedOPDExample]:
    questions = read_jsonl(questions_path)
    memories = read_jsonl(self_memories_path)
    memory_by_id: dict[str, dict[str, Any]] = {}
    for row in memories:
        question_id = str(row.get("question_id", row.get("task_id", "")))
        if not question_id or question_id in memory_by_id:
            raise ValueError(f"missing or duplicate self-memory question ID: {question_id!r}")
        memory_by_id[question_id] = row

    result: list[CompressedOPDExample] = []
    seen: set[str] = set()
    for row in questions:
        question_id = str(row["question_id"])
        if question_id in seen:
            raise ValueError(f"duplicate question ID: {question_id}")
        seen.add(question_id)
        source = memory_by_id.get(question_id)
        if source is None:
            raise KeyError(f"missing self memory for {question_id}")
        split = str(row.get("split", source.get("split", "")))
        if expected_split is not None and split != expected_split:
            raise ValueError(f"expected {expected_split} row, found {split} for {question_id}")
        memory = canonical_memory(source["memory"])
        memory_text = serialize_memory(memory)
        options = tuple(str(item) for item in row["options"])
        if len(options) != 4:
            raise ValueError(f"question {question_id} does not have four options")
        gold = str(row["answer"]).lower()
        if gold not in OPTION_LABELS:
            raise ValueError(f"invalid gold label for {question_id}: {gold}")
        result.append(
            CompressedOPDExample(
                question_id=question_id,
                shared_context_id=str(row["shared_context_id"]),
                split=split,
                question_type=str(row.get("question_type", "")),
                question=str(row["question"]),
                options=options,  # type: ignore[arg-type]
                gold_label=gold,
                memory=memory,
                memory_text=memory_text,
            )
        )
    extra = set(memory_by_id) - seen
    if extra:
        raise ValueError(f"self memories contain unknown IDs: {sorted(extra)[:3]}")
    return result
