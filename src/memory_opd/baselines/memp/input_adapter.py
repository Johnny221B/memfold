"""Adapt verified ALFWorld walkthroughs to the official MemP cold-start schema."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any


OBJECTIVE_RE = re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def canonical_objective(record: dict[str, Any]) -> str:
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("trajectory has no steps")
    match = OBJECTIVE_RE.search(str(steps[0].get("observation", "")))
    if not match:
        raise ValueError("trajectory reset observation has no canonical objective")
    return match.group(1).strip()


def balanced_sample(records: Iterable[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for record in sorted(records, key=lambda item: str(item["trajectory_id"])):
        groups[str(record["task_type"])].append(record)
    selected: list[dict[str, Any]] = []
    while len(selected) < count and any(groups.values()):
        for task_type in sorted(groups):
            if groups[task_type] and len(selected) < count:
                selected.append(groups[task_type].popleft())
    if len(selected) != count:
        raise ValueError(f"requested {count} trajectories, found {len(selected)}")
    return selected


def to_memp_item(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("success") is not True:
        raise ValueError("MemP cold start accepts only successful source trajectories")
    trajectory = []
    for index, step in enumerate(record["steps"], start=1):
        trajectory.append(
            {
                "step": index,
                "action": str(step["action"]),
                "observation": str(step["next_observation"]),
                "state": "Successful",
            }
        )
    return {
        "source": str(record["trajectory_id"]),
        "query": canonical_objective(record),
        "trajectory": trajectory,
        "facts": {},
    }


def prepare_input(
    trajectories: Path, split_manifest: Path, count: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    memory_ids = set(split["memory_trajectory_ids"])
    records = []
    with trajectories.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["trajectory_id"] in memory_ids:
                records.append(record)
    if len(records) != split["memory_count"]:
        raise ValueError(f"found {len(records)} memory records; expected {split['memory_count']}")
    items = [to_memp_item(record) for record in balanced_sample(records, count)]
    encoded = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "schema_version": "1.0",
        "selection": "round_robin_sorted_task_type_and_trajectory_id",
        "count": len(items),
        "source_split_sha256": hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(encoded).hexdigest(),
        "trajectory_ids": [item["source"] for item in items],
    }
    return items, manifest
