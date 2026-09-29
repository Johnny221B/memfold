from __future__ import annotations

import re

STRICT_OPTION_RE = re.compile(r"^\([abcd]\)$")


def parse_strict_option(text: str) -> str | None:
    candidate = text.strip()
    return candidate if STRICT_OPTION_RE.fullmatch(candidate) else None


def canonical_gold(answer: str, options: tuple[str, str, str, str]) -> str:
    value = answer.strip()
    lowered = value.lower()
    if lowered in {"(a)", "(b)", "(c)", "(d)"}:
        return lowered
    if lowered in {"a", "b", "c", "d"}:
        return f"({lowered})"
    matches = [index for index, option in enumerate(options) if option.strip() == value]
    if len(matches) != 1:
        raise ValueError("gold answer is neither an option label nor one unique option text")
    return f"({chr(ord('a') + matches[0])})"
