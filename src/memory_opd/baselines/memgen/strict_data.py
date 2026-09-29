"""Trajectory-level ALFWorld adapter for the base MemGen method."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = (
    "You are controlling a text-based ALFWorld environment. Complete the task by "
    "interacting with the environment. At each turn, output exactly one command from "
    "the current admissible-action list and no additional text."
)

# Pinned verbatim from MemGen revision 970cc95, memgen/utils.py.
OFFICIAL_CHAT_TEMPLATE = r"""
{# ───── main loop ───── #}
{%- for message in messages -%}
    {%- set content = message.content if message.content is string else "" -%}
    {%- if (message.role == "user") or (message.role == "system") -%}
        {{ "<|im_start|>" + message.role + "\n"  + content + "<|im_end|>\n" }}
    {%- elif message.role == "assistant" -%}
        {%- generation -%}
        {{ "<|im_start|>assistant\n" + content + "<|im_end|>\n" }}
        {%- endgeneration -%}
    {%- elif message.role == "tool" -%}
    {{ "<|im_start|>" + "user\n"  + content + "<|im_end|>\n" }}
    {%- endif -%}
{%- endfor -%}
{# ───── generation prompt ───── #}
{%- if add_generation_prompt -%}
    {{ "<|im_start|>assistant\n" }}
{%- endif -%}
""".strip()


def validate_success_trajectory(row: dict[str, Any]) -> None:
    required = ("trajectory_id", "objective", "steps", "success", "task_type")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"trajectory is missing fields: {', '.join(missing)}")
    if row["success"] is not True or not row["steps"]:
        raise ValueError("MemGen experience history requires a non-empty successful trajectory")
    for index, step in enumerate(row["steps"]):
        action = str(step.get("action", "")).strip()
        actions = [str(item).strip() for item in step.get("admissible_actions", [])]
        if not action or action not in actions:
            raise ValueError(f"step {index} expert action is not admissible")


def render_state_turn(objective: str, step: dict[str, Any], turn_index: int) -> str:
    observation = str(step["observation"]).strip()
    actions = [str(item).strip() for item in step["admissible_actions"]]
    return (
        f"Task: {objective.strip()}\n\n"
        + f"Current observation:\n{observation}\n\n"
        + "Admissible actions:\n"
        + "\n".join(f"- {action}" for action in actions)
        + "\n\nNext action:"
    )


def trajectory_to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Represent one expert trajectory as an alternating ChatML conversation."""

    validate_success_trajectory(row)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    objective = str(row["objective"])
    for index, step in enumerate(row["steps"]):
        messages.append({"role": "user", "content": render_state_turn(objective, step, index)})
        messages.append({"role": "assistant", "content": str(step["action"]).strip()})
    return messages


def build_record(row: dict[str, Any]) -> dict[str, Any]:
    messages = trajectory_to_messages(row)
    return {
        "schema_version": "1.0",
        "trajectory_id": str(row["trajectory_id"]),
        "objective": str(row["objective"]),
        "task_type": str(row["task_type"]),
        "turn_count": len(row["steps"]),
        "messages": messages,
    }
