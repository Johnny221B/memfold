"""Deterministic trajectory-level splits for ALFWorld memory experiments."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TrajectorySplit:
    memory_ids: tuple[str, ...]
    il_ids: tuple[str, ...]


def read_trajectory_headers(path: Path) -> list[dict[str, object]]:
    """Read fields required for splitting without retaining full step payloads."""

    headers: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
            trajectory_id = record.get("trajectory_id")
            if not isinstance(trajectory_id, str) or not trajectory_id:
                raise ValueError(f"line {line_number} has no trajectory_id")
            if record.get("success") is not True:
                raise ValueError(f"trajectory {trajectory_id} is not successful")
            headers.append(
                {
                    "trajectory_id": trajectory_id,
                    "task_type": record.get("task_type"),
                    "step_count": len(record.get("steps", [])),
                }
            )
    identifiers = [str(item["trajectory_id"]) for item in headers]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("trajectory IDs must be unique")
    if not identifiers:
        raise ValueError("trajectory file is empty")
    return headers


def deterministic_split(
    trajectory_ids: Iterable[str], *, memory_fraction: float, seed: int
) -> TrajectorySplit:
    """Shuffle stable IDs with a local RNG and split with floor semantics."""

    if not 0.0 < memory_fraction < 1.0:
        raise ValueError("memory_fraction must be between zero and one")
    identifiers = sorted(trajectory_ids)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("trajectory IDs must be unique")
    random.Random(seed).shuffle(identifiers)
    boundary = int(len(identifiers) * memory_fraction)
    return TrajectorySplit(
        memory_ids=tuple(identifiers[:boundary]), il_ids=tuple(identifiers[boundary:])
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_split_manifest(path: Path, *, memory_fraction: float, seed: int) -> dict:
    headers = read_trajectory_headers(path)
    split = deterministic_split(
        (str(item["trajectory_id"]) for item in headers),
        memory_fraction=memory_fraction,
        seed=seed,
    )
    task_types: dict[str, dict[str, int]] = {}
    split_by_id = {identifier: "memory" for identifier in split.memory_ids}
    split_by_id.update({identifier: "il" for identifier in split.il_ids})
    for header in headers:
        task_type = str(header["task_type"])
        counts = task_types.setdefault(task_type, {"total": 0, "memory": 0, "il": 0})
        counts["total"] += 1
        counts[split_by_id[str(header["trajectory_id"])]] += 1
    return {
        "schema_version": "1.0",
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "seed": seed,
        "algorithm": "sorted_ids_then_python_random_shuffle_floor_split",
        "memory_fraction": memory_fraction,
        "total_count": len(headers),
        "memory_count": len(split.memory_ids),
        "il_count": len(split.il_ids),
        "task_type_counts": task_types,
        "memory_trajectory_ids": list(split.memory_ids),
        "il_trajectory_ids": list(split.il_ids),
    }
