"""Atomic, independently resumable Vanilla GRPO checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch


CHECKPOINT_SCHEMA = "vanilla-grpo-checkpoint-v1"


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a checkpoint directory without overwriting one."""

    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        torch.save({"schema": CHECKPOINT_SCHEMA, **payload}, temporary / "state.pt")
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported Vanilla GRPO checkpoint schema")
    return payload
