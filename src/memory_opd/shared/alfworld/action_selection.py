"""Track-neutral helpers for mapping model text to ALFWorld actions."""

from __future__ import annotations

import difflib


def snap_action(generated: str, actions: list[str]) -> str:
    """Return the exact or closest admissible action for generated text."""

    if not actions:
        raise ValueError("cannot snap against an empty admissible-action list")
    normalized = generated.strip().lower()
    exact = {action.strip().lower(): action for action in actions}
    if normalized in exact:
        return exact[normalized]
    return max(
        actions,
        key=lambda action: difflib.SequenceMatcher(
            None, normalized, action.strip().lower()
        ).ratio(),
    )
