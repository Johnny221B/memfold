"""Inventory official ALFWorld TextWorld training games."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GameRecord:
    """Stable metadata needed to replay one official ALFWorld game."""

    trajectory_id: str
    relative_gamefile: str
    relative_traj_data: str
    task_id: str
    task_type: str
    objective: str
    solvable: bool
    split: str = "train"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read valid JSON from {path}: {exc}") from exc


def _objective(traj_data: dict[str, Any]) -> str:
    annotations = traj_data.get("turk_annotations", {}).get("anns", [])
    for annotation in annotations:
        task_desc = str(annotation.get("task_desc", "")).strip()
        if task_desc:
            return task_desc
    raise ValueError("traj_data has no non-empty turk task_desc")


def _stable_id(relative_gamefile: str, task_id: str) -> str:
    digest = hashlib.sha256(relative_gamefile.encode("utf-8")).hexdigest()[:16]
    safe_task_id = task_id.replace("/", "_") or "unknown"
    return f"{safe_task_id}_{digest}"


def inventory_games(data_root: Path, split: str) -> list[GameRecord]:
    """Return official executable games for one ALFWorld split."""
    data_root = data_root.resolve()
    if split not in {"train", "valid_seen", "valid_unseen"}:
        raise ValueError(f"unsupported ALFWorld split: {split}")
    split_root = data_root / "json_2.1.1" / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"ALFWorld {split} directory not found: {split_root}")

    records: list[GameRecord] = []
    for gamefile in sorted(split_root.rglob("game.tw-pddl")):
        traj_path = gamefile.with_name("traj_data.json")
        if not traj_path.is_file():
            raise FileNotFoundError(f"Missing traj_data.json beside {gamefile}")

        game_data = _read_json(gamefile)
        traj_data = _read_json(traj_path)
        relative_gamefile = str(gamefile.relative_to(data_root))
        task_id = str(traj_data.get("task_id", ""))
        records.append(
            GameRecord(
                trajectory_id=_stable_id(relative_gamefile, task_id),
                relative_gamefile=relative_gamefile,
                relative_traj_data=str(traj_path.relative_to(data_root)),
                task_id=task_id,
                task_type=str(traj_data.get("task_type", "unknown")),
                objective=_objective(traj_data),
                solvable=bool(game_data.get("solvable", False)),
                split=split,
            )
        )
    return records


def inventory_train_games(data_root: Path) -> list[GameRecord]:
    """Backward-compatible train inventory entry point."""

    return inventory_games(data_root, "train")
