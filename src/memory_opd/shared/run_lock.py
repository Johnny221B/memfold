"""Process-lifetime exclusive locks for experiment output directories."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import time
from typing import TextIO


def acquire_run_lock(output_dir: Path) -> TextIO:
    """Acquire and return an exclusive lock held until the handle is closed."""

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".trainer.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(f"another trainer holds {lock_path}: {owner}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "started_unix": time.time()}) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle
