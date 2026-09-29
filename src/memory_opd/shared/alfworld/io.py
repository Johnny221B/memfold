"""Crash-safe JSON output helpers for trajectory collection."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, value: Any, *, durable: bool = False) -> None:
    """Atomically replace a JSON file after fully writing it.

    ``durable=True`` additionally calls ``fsync``. That is useful for small
    manifests, but prohibitively slow for thousands of reproducible shards on
    network storage. Closing the temporary file before ``os.replace`` still
    prevents readers from observing partial JSON after a process interruption.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
