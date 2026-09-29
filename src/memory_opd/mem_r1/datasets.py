"""Strict dataset boundaries for the Memory-R1 LoCoMo transfer experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
LONGMEMEVAL_S_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
LOCOMO_SPLITS = {
    "train": ("conv-26",),
    "validation": ("conv-30",),
    "test": ("conv-41", "conv-42", "conv-43", "conv-44", "conv-47", "conv-48", "conv-49", "conv-50"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class LocomoSplit:
    name: str
    conversations: tuple[dict[str, Any], ...]
    questions: tuple[dict[str, Any], ...]


def load_locomo(path: Path, *, verify_hash: bool = True) -> dict[str, LocomoSplit]:
    if verify_hash and sha256(path) != LOCOMO_SHA256:
        raise ValueError("LoCoMo source hash does not match the audited release")
    conversations = json.loads(path.read_text(encoding="utf-8"))
    by_id = {item["sample_id"]: item for item in conversations}
    if set(by_id) != {item for values in LOCOMO_SPLITS.values() for item in values}:
        raise ValueError("unexpected LoCoMo conversation IDs")
    result: dict[str, LocomoSplit] = {}
    for split, ids in LOCOMO_SPLITS.items():
        selected = tuple(by_id[item] for item in ids)
        questions = tuple(
            {**question, "sample_id": conversation["sample_id"]}
            for conversation in selected
            for question in conversation["qa"]
            if str(question.get("category")) != "5"
        )
        result[split] = LocomoSplit(split, selected, questions)
    return result


def load_longmemeval_test(path: Path, *, verify_hash: bool = True) -> tuple[dict[str, Any], ...]:
    if verify_hash and sha256(path) != LONGMEMEVAL_S_SHA256:
        raise ValueError("LongMemEval-S source hash does not match the audited cleaned release")
    records = tuple(json.loads(path.read_text(encoding="utf-8")))
    if len(records) != 500 or len({item["question_id"] for item in records}) != 500:
        raise ValueError("LongMemEval-S must contain 500 unique test questions")
    return records


def assert_transfer_isolation(splits: dict[str, LocomoSplit], longmemeval: tuple[dict[str, Any], ...]) -> None:
    conversation_sets = [
        {item["sample_id"] for item in split.conversations} for split in splits.values()
    ]
    if any(conversation_sets[i] & conversation_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("LoCoMo conversation leakage across splits")
    if any("split" in item and item["split"] == "train" for item in longmemeval):
        raise AssertionError("LongMemEval records cannot enter training")
