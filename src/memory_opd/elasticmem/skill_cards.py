"""ElasticMem ALFWorld skill-card prompting and validation."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are an expert at analyzing household robot trajectories. "
    "Extract specific, actionable lessons from the provided trajectory."
)
PROMPT_VERSION = "elasticmem_appendix_f_table21_v1"


def extract_canonical_objective(record: dict[str, Any]) -> str:
    """Read the PDDL-style task objective rendered by ALFWorld at reset."""

    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("trajectory must contain non-empty steps")
    initial_observation = str(steps[0].get("observation", ""))
    match = re.search(
        r"Your task is to:\s*(.+?)(?:\n|$)", initial_observation, flags=re.I
    )
    if not match:
        raise ValueError("initial observation has no canonical ALFWorld objective")
    return match.group(1).strip()


def render_trajectory(record: dict[str, Any]) -> str:
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("trajectory must contain non-empty steps")
    rendered: list[str] = []
    for index, step in enumerate(steps, start=1):
        action = str(step.get("action", "")).strip()
        observation = str(step.get("next_observation", "")).strip()
        if not action or not observation:
            raise ValueError(f"trajectory step {index} is incomplete")
        rendered.append(f"{index}. {action} -> {observation}")
    return "\n".join(rendered)


def build_success_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    if record.get("success") is not True:
        raise ValueError("success prompt requires a successful trajectory")
    objective = extract_canonical_objective(record)
    user_prompt = f"""An ALFWorld household task was completed successfully.
Task: {objective}
Full trajectory (action -> observation):
{render_trajectory(record)}

Extract a reusable skill (3-5 sentences). Include:
1. The general task category (pick_and_place, heat_then_place, clean_then_place, cool_then_place, examine_in_light, pick_two)
2. The concrete step-by-step strategy that worked
3. Common locations where target objects are found (e.g. "soapbar is usually on countertop, bathtubbasin, or shelf")
Be specific. Use actual object/location types (countertop, sinkbasin, microwave).
Output format: SKILL: [your skill text]"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_skill_card(generated_text: str) -> str:
    """Extract and minimally validate the paper-specified `SKILL:` payload."""

    matches = list(re.finditer(r"(?:^|\n)\s*SKILL:\s*", generated_text, re.I))
    if not matches:
        raise ValueError("model output does not contain a SKILL: prefix")
    skill = generated_text[matches[-1].end() :].strip()
    skill = re.sub(r"\s+", " ", skill)
    if len(skill) < 40:
        raise ValueError("skill card is too short")
    if len(skill) > 3000:
        raise ValueError("skill card is unexpectedly long")
    return skill


def balanced_sample(
    records: Iterable[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Select records round-robin across sorted task types and trajectory IDs."""

    if count <= 0:
        raise ValueError("count must be positive")
    groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for record in sorted(records, key=lambda item: str(item["trajectory_id"])):
        groups[str(record.get("task_type"))].append(record)
    selected: list[dict[str, Any]] = []
    task_types = sorted(groups)
    while len(selected) < count and any(groups.values()):
        for task_type in task_types:
            if groups[task_type] and len(selected) < count:
                selected.append(groups[task_type].popleft())
    if len(selected) < count:
        raise ValueError(f"requested {count} records but only found {len(selected)}")
    return selected
