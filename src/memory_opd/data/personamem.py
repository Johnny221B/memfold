"""PersonaMem-v1 loading, normalization, leakage-safe splitting, and prompts."""

from __future__ import annotations

import ast
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LABELS = ("(a)", "(b)", "(c)", "(d)")
EXPECTED_SPLIT_COUNTS = {
    "32k": {"train": 489, "validation": 50, "test": 50},
    "128k": {"train": 2221, "validation": 273, "test": 233},
}
REQUIRED_COLUMNS = {
    "question_id",
    "user_question_or_message",
    "correct_answer",
    "all_options",
    "shared_context_id",
    "end_index_in_shared_context",
}


@dataclass(frozen=True)
class PersonaMemExample:
    question_id: str
    shared_context_id: str
    question: str
    options: tuple[str, str, str, str]
    answer: str
    end_index_in_shared_context: int
    persona_id: str = ""
    question_type: str = ""
    topic: str = ""

    def prompt(self, memory: str) -> str:
        rendered = "\n".join(f"{label} {option}" for label, option in zip(LABELS, self.options))
        return (
            f"QUESTION:\n{self.question}\n\n"
            f"RETRIEVED MEMORY:\n{memory}\n\n"
            "Output exactly one option: (a), (b), (c), or (d), and nothing else.\n\n"
            f"OPTIONS:\n{rendered}\n\nANSWER:\n"
        )

    def as_record(self, *, split: str, memory: str) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "split": split,
                "prompt": self.prompt(memory),
                "reward_model": {"style": "rule", "ground_truth": self.answer},
                "data_source": "personamem_v1",
                "extra_info": {
                    "question_id": self.question_id,
                    "shared_context_id": self.shared_context_id,
                },
            }
        )
        return record


def _parse_options(raw: str) -> tuple[str, str, str, str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"all_options is not a JSON/Python list: {raw!r}") from error
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        raise ValueError(f"all_options must contain exactly four strings: {parsed!r}")
    normalized: list[str] = []
    for index, item in enumerate(parsed):
        option = str(item).strip()
        label = LABELS[index]
        if option.lower().startswith(label):
            option = option[len(label) :].strip()
        normalized.append(option)
    options = tuple(normalized)
    if any(not item for item in options):
        raise ValueError("all_options cannot contain an empty option")
    return options  # type: ignore[return-value]


def _canonical_answer(raw: str, options: Sequence[str]) -> str:
    answer = raw.strip()
    lowered = answer.lower()
    if lowered in LABELS:
        return lowered
    if lowered in {"a", "b", "c", "d"}:
        return f"({lowered})"
    matches = [index for index, option in enumerate(options) if option.strip() == answer]
    if len(matches) != 1:
        raise ValueError(f"correct_answer must identify exactly one option: {raw!r}")
    return LABELS[matches[0]]


def load_questions(path: Path) -> tuple[PersonaMemExample, ...]:
    """Load the official PersonaMem questions CSV with strict schema checks."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"PersonaMem CSV is missing columns: {sorted(missing)}")
        examples: list[PersonaMemExample] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                options = _parse_options(row["all_options"])
                examples.append(
                    PersonaMemExample(
                        question_id=row["question_id"].strip(),
                        shared_context_id=row["shared_context_id"].strip(),
                        question=row["user_question_or_message"].strip(),
                        options=options,
                        answer=_canonical_answer(row["correct_answer"], options),
                        end_index_in_shared_context=int(row["end_index_in_shared_context"]),
                        persona_id=row.get("persona_id", "").strip(),
                        question_type=row.get("question_type", "").strip(),
                        topic=row.get("topic", "").strip(),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid PersonaMem row {line_number}: {error}") from error
    if not examples:
        raise ValueError("PersonaMem questions CSV is empty")
    question_ids = [item.question_id for item in examples]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("PersonaMem question_id values must be unique")
    if any(not item.shared_context_id or not item.question_id or not item.question for item in examples):
        raise ValueError("question_id, shared_context_id, and question must be non-empty")
    return tuple(examples)


def _subset_with_total(groups: Sequence[tuple[str, int]], target: int) -> set[str]:
    """Find a deterministic subset of context groups with exactly target questions."""

    paths: dict[int, tuple[str, ...]] = {0: ()}
    for context_id, size in groups:
        for subtotal, selected in tuple(sorted(paths.items(), reverse=True)):
            candidate = subtotal + size
            if candidate <= target and candidate not in paths:
                paths[candidate] = (*selected, context_id)
        if target in paths:
            return set(paths[target])
    raise ValueError(f"cannot form an exact split of {target} questions without context leakage")


def split_by_context(
    examples: Sequence[PersonaMemExample],
    *,
    benchmark_size: str,
    seed: int = 42,
) -> dict[str, tuple[PersonaMemExample, ...]]:
    """Create exact train/validation/test counts without splitting a context group."""

    if benchmark_size not in EXPECTED_SPLIT_COUNTS:
        raise ValueError("benchmark_size must be '32k' or '128k'")
    expected = EXPECTED_SPLIT_COUNTS[benchmark_size]
    if len(examples) != sum(expected.values()):
        raise ValueError(
            f"PersonaMem-{benchmark_size} must contain {sum(expected.values())} questions; "
            f"found {len(examples)}"
        )
    grouped: dict[str, list[PersonaMemExample]] = defaultdict(list)
    for example in examples:
        grouped[example.shared_context_id].append(example)
    rng = random.Random(seed)
    group_sizes = [(context_id, len(items)) for context_id, items in sorted(grouped.items())]
    rng.shuffle(group_sizes)
    test_ids = _subset_with_total(group_sizes, expected["test"])
    remaining = [item for item in group_sizes if item[0] not in test_ids]
    validation_ids = _subset_with_total(remaining, expected["validation"])
    split_for_id = {
        **{context_id: "test" for context_id in test_ids},
        **{context_id: "validation" for context_id in validation_ids},
    }
    result: dict[str, list[PersonaMemExample]] = defaultdict(list)
    for example in examples:
        result[split_for_id.get(example.shared_context_id, "train")].append(example)
    counts = Counter({name: len(result[name]) for name in expected})
    if dict(counts) != expected:
        raise AssertionError(f"unexpected split counts: {dict(counts)} != {expected}")
    context_sets = {
        name: {item.shared_context_id for item in result[name]} for name in expected
    }
    if any(
        context_sets[left] & context_sets[right]
        for index, left in enumerate(expected)
        for right in tuple(expected)[index + 1 :]
    ):
        raise AssertionError("shared_context_id leakage across splits")
    return {name: tuple(result[name]) for name in expected}


def load_contexts(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load official one-key-per-line PersonaMem context mappings."""

    contexts: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping) or len(value) != 1:
                raise ValueError(f"context line {line_number} must be a one-key JSON object")
            context_id, messages = next(iter(value.items()))
            if not isinstance(messages, list) or not all(isinstance(item, Mapping) for item in messages):
                raise ValueError(f"context {context_id!r} must be a list of message objects")
            if str(context_id) in contexts:
                raise ValueError(f"duplicate context id: {context_id}")
            contexts[str(context_id)] = [dict(item) for item in messages]
    return contexts


def render_context(messages: Iterable[Mapping[str, Any]]) -> str:
    """Render API-style messages without applying a model-specific chat template."""

    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role", "unknown")).upper()
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(part.get("text", part)) if isinstance(part, Mapping) else str(part) for part in content)
        rendered.append(f"{role}: {content}")
    return "\n".join(rendered)
