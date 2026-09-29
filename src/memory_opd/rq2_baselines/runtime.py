"""Read-only preflight checks for the locally reproduced OPSD runtime."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


EXPECTED_PACKAGES = {
    "torch": "2.8.0",
    "transformers": "4.57.1",
    "trl": "0.26.0",
    "accelerate": "1.11.0",
    "peft": "0.17.1",
    "datasets": "3.6.0",
    "deepspeed": "0.18.2",
    "vllm": "0.11.0",
}


@dataclass(frozen=True)
class RuntimeAudit:
    facts: dict[str, Any]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _command(*args: str) -> str:
    result = subprocess.run(args, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _environment_versions(python: Path) -> dict[str, str | None]:
    code = (
        "import importlib.metadata as m,json;"
        f"names={list(EXPECTED_PACKAGES)!r};"
        "print(json.dumps({n:(m.version(n) if n in {d.metadata['Name'] for d in m.distributions()} else None) for n in names}))"
    )
    return json.loads(_command(str(python), "-c", code))


def verify_opsd_source(source: Path, expected_commit: str) -> None:
    """Fail if the official source moved or has tracked local edits."""

    actual_commit = _command("git", "-C", str(source), "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise RuntimeError(f"OPSD commit mismatch: {actual_commit} != {expected_commit}")
    tracked_changes = _command("git", "-C", str(source), "status", "--short", "--untracked-files=no")
    if tracked_changes:
        raise RuntimeError("official OPSD checkout has tracked modifications")


def audit_runtime(config: Mapping[str, Any], *, model_path: Path) -> RuntimeAudit:
    errors: list[str] = []
    warnings: list[str] = []
    facts: dict[str, Any] = {}
    framework = config.get("framework", {})
    source = Path(framework.get("local_source", ""))
    python = Path(framework.get("python", ""))
    expected_commit = str(framework.get("commit", ""))
    if not source.is_dir():
        errors.append(f"official OPSD source is missing: {source}")
    else:
        try:
            actual_commit = _command("git", "-C", str(source), "rev-parse", "HEAD")
            facts["source_commit"] = actual_commit
            if actual_commit != expected_commit:
                errors.append(f"OPSD commit mismatch: {actual_commit} != {expected_commit}")
            tracked_changes = _command("git", "-C", str(source), "status", "--short", "--untracked-files=no")
            if tracked_changes:
                errors.append("official OPSD checkout has tracked modifications")
        except (OSError, subprocess.CalledProcessError) as error:
            errors.append(f"cannot audit official OPSD checkout: {error}")
    if not python.is_file():
        errors.append(f"OPSD Python is missing: {python}")
    else:
        try:
            versions = _environment_versions(python)
            facts["packages"] = versions
            for name, expected in EXPECTED_PACKAGES.items():
                if versions.get(name) != expected:
                    errors.append(f"package mismatch for {name}: {versions.get(name)} != {expected}")
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
            errors.append(f"cannot inspect OPSD Python environment: {error}")
    model_config_path = model_path / "config.json"
    if not model_config_path.is_file():
        errors.append(f"model config is missing: {model_config_path}")
    else:
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        max_positions = int(model_config.get("max_position_embeddings", 0))
        rope_scaling = model_config.get("rope_scaling")
        facts["model_max_position_embeddings"] = max_positions
        facts["model_rope_scaling"] = rope_scaling
        benchmark_size = config.get("dataset", {}).get("benchmark_size")
        required = 32_768 if benchmark_size == "32k" else 131_072
        if max_positions < required and not rope_scaling:
            errors.append(
                f"{benchmark_size} full-history input requires {required} positions, but model "
                f"declares {max_positions} without rope_scaling"
            )
        elif max_positions < required:
            warnings.append("rope_scaling is configured; validate the effective context limit in a tokenizer smoke")
    return RuntimeAudit(facts=facts, errors=tuple(errors), warnings=tuple(warnings))
