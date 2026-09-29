"""Shared, deterministic ALFWorld evaluation primitives for all baselines."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

OBJECTIVE_MARKER = "Your task is to:"
ACTION_PREFIX = re.compile(r"^\s*action\s*:\s*", re.IGNORECASE)


def extract_canonical_objective(reset_observation: str) -> str:
    if OBJECTIVE_MARKER not in reset_observation:
        raise ValueError(f"reset observation lacks {OBJECTIVE_MARKER!r}")
    suffix = reset_observation.rsplit(OBJECTIVE_MARKER, 1)[1].strip()
    objective = next((line.strip() for line in suffix.splitlines() if line.strip()), "")
    if not objective:
        raise ValueError("canonical objective is empty")
    return objective


def normalize_action(action: str) -> str:
    value = unicodedata.normalize("NFKC", action).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"`":
        value = value[1:-1].strip()
    return " ".join(value.lower().split())


def parse_action(model_output: str) -> str:
    value = unicodedata.normalize("NFKC", model_output).strip()
    if value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    if "</think>" in value:
        value = value.rsplit("</think>", 1)[1].strip()
    action_lines = [line.strip() for line in value.splitlines() if ACTION_PREFIX.match(line)]
    first_line = action_lines[-1] if action_lines else next(
        (line.strip() for line in reversed(value.splitlines()) if line.strip()), ""
    )
    first_line = ACTION_PREFIX.sub("", first_line)
    if len(first_line) >= 2 and first_line[0] == first_line[-1] and first_line[0] in "'\"`":
        first_line = first_line[1:-1].strip()
    return first_line


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for col, right_char in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[col] + 1, previous[col - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class SnappedAction:
    parsed: str
    executed: str
    raw_exact_valid: bool
    edit_distance: float


def snap_action(model_output: str, admissible_actions: Sequence[str]) -> SnappedAction:
    if not admissible_actions:
        raise ValueError("admissible_actions must not be empty")
    parsed = parse_action(model_output)
    normalized = normalize_action(parsed)
    normalized_actions = [normalize_action(action) for action in admissible_actions]
    distances = [levenshtein(normalized, candidate) / max(len(normalized), len(candidate), 1) for candidate in normalized_actions]
    best_index = min(range(len(distances)), key=distances.__getitem__)
    return SnappedAction(parsed, str(admissible_actions[best_index]), normalized in normalized_actions, distances[best_index])


def render_prompt(objective: str, observation: str, admissible_actions: Sequence[str], history: Sequence[tuple[str, str]], *, history_size: int = 5, memory_guidance: str = "") -> tuple[str, str]:
    if history_size < 0:
        raise ValueError("history_size must be non-negative")
    recent = history[-history_size:] if history_size else []
    history_text = "\n".join(f"Action: {action}\nObservation: {result}" for action, result in recent) or "(none)"
    actions_text = "\n".join(f"- {action}" for action in admissible_actions)
    system = "You are controlling a text-based ALFWorld environment.\nChoose the NEXT action as ONE admissible command string.\nOutput only the command, copied verbatim from the admissible list."
    user = f"Task: {objective}\n\nMemory guidance:\n{memory_guidance}\n\nInteraction history so far:\n{history_text}\n\nCurrent observation:\n{observation}\n\nAdmissible actions:\n{actions_text}\n\nAction:"
    return system, user
