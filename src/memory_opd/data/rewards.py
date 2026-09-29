"""Strict multiple-choice answer parsing."""
import re

CHOICE_PATTERN = re.compile(r"^\([abcd]\)$")

def parse_choice(value: object) -> str | None:
    """Return a canonical choice only when the entire output is valid."""

    if not isinstance(value, str):
        return None
    candidate = value.strip()
    return candidate if CHOICE_PATTERN.fullmatch(candidate) else None
