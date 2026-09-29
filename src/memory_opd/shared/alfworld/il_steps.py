"""Deterministic ALFWorld trajectory-to-per-step IL conversion."""

from __future__ import annotations

from typing import Any, Iterable
import re

from memory_opd.elasticmem.skill_cards import extract_canonical_objective


def objective_from_observation(observation: str) -> str:
    """Extract the canonical ALFWorld task objective from reset feedback."""

    match = re.search(r"Your task is to:\s*(.+?)(?:\n|$)", observation, flags=re.I)
    if not match:
        raise ValueError("reset observation has no canonical objective")
    return match.group(1).strip()


def render_step_suffix(
    steps: list[dict[str, Any]], step_index: int, history_length: int = 5
) -> str:
    """Render the paper-style last-five history, state, and action candidates."""

    if not 0 <= step_index < len(steps):
        raise IndexError("step_index is outside trajectory")
    current = steps[step_index]
    history_lines: list[str] = []
    if history_length < 0:
        raise ValueError("history_length must be non-negative")
    for previous in steps[max(0, step_index - history_length) : step_index]:
        history_lines.append(
            f"- {str(previous['action']).strip()} -> "
            f"{str(previous['next_observation']).strip()}"
        )
    history = "\n".join(history_lines) if history_lines else "(none)"
    actions = [str(action).strip() for action in current["admissible_actions"]]
    return (
        f"History (last {history_length} action-observation pairs):\n{history}\n"
        f"Current observation:\n{str(current['observation']).strip()}\n"
        "Admissible actions:\n"
        + "\n".join(f"- {action}" for action in actions)
        + "\nInstruction: Output exactly one admissible command verbatim."
    )


def trajectory_to_il_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one successful expert trajectory without crossing episode boundaries."""

    if record.get("success") is not True:
        raise ValueError("IL conversion requires a successful trajectory")
    trajectory_id = str(record.get("trajectory_id", "")).strip()
    steps = record.get("steps")
    if not trajectory_id or not isinstance(steps, list) or not steps:
        raise ValueError("trajectory_id and non-empty steps are required")
    query = extract_canonical_objective(record)
    samples: list[dict[str, Any]] = []
    for step_index, step in enumerate(steps):
        action = str(step.get("action", "")).strip()
        admissible = [str(item).strip() for item in step.get("admissible_actions", [])]
        if not action or action not in admissible:
            raise ValueError(
                f"expert action is absent from admissible actions: {trajectory_id}:{step_index}"
            )
        samples.append(
            {
                "schema_version": "1.0",
                "sample_id": f"{trajectory_id}_step_{step_index:03d}",
                "trajectory_id": trajectory_id,
                "step_index": step_index,
                "task_type": str(record.get("task_type", "unknown")),
                "query": query,
                "suffix": render_step_suffix(steps, step_index),
                "expert_action": action,
                "admissible_actions": admissible,
                "done": bool(step.get("done", False)),
            }
        )
    return samples


def select_and_expand(
    trajectories: Iterable[dict[str, Any]], selected_ids: set[str]
) -> list[dict[str, Any]]:
    """Select an exact trajectory split and return stable source-order samples."""

    seen: set[str] = set()
    samples: list[dict[str, Any]] = []
    for record in trajectories:
        trajectory_id = str(record.get("trajectory_id", ""))
        if trajectory_id in selected_ids:
            if trajectory_id in seen:
                raise ValueError(f"duplicate selected trajectory: {trajectory_id}")
            seen.add(trajectory_id)
            samples.extend(trajectory_to_il_steps(record))
    missing = selected_ids - seen
    if missing:
        raise ValueError(f"selected trajectories missing from source: {len(missing)}")
    return samples
