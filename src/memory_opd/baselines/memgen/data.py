"""ALFWorld per-step adapter for the conservative MemGen Stage-1 protocol."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = (
    "You are controlling a text-based ALFWorld environment. "
    "Choose the NEXT action as ONE admissible command string. "
    "Output only the command, copied verbatim from the admissible list."
)


def validate_il_row(row: dict[str, Any]) -> None:
    required = ("sample_id", "query", "suffix", "expert_action", "admissible_actions")
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"IL row is missing fields: {', '.join(missing)}")
    action = str(row["expert_action"]).strip()
    admissible = [str(item).strip() for item in row["admissible_actions"]]
    if not action or action not in admissible:
        raise ValueError("expert action must be a non-empty admissible command")


def render_prompt(row: dict[str, Any]) -> str:
    """Render the shared non-memory prompt without a textual memory block."""

    validate_il_row(row)
    return (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
        f"[USER]\nTask: {str(row['query']).strip()}\n\n"
        f"{str(row['suffix']).strip()}\n\nAction:"
    )


def render_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Return role messages for the backbone's native chat template."""

    validate_il_row(row)
    user = (
        f"Task: {str(row['query']).strip()}\n\n"
        f"{str(row['suffix']).strip()}\n\nAction:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def render_inference_messages(query: str, suffix: str) -> list[dict[str, str]]:
    """Render a closed-loop prompt when no expert action is available."""

    query = query.strip()
    suffix = suffix.strip()
    if not query or not suffix:
        raise ValueError("query and suffix must be non-empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {query}\n\n{suffix}\n\nAction:"},
    ]
