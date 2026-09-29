"""CPU oracle for the pinned SEED ALFWorld action projection.

Derived from SEED ``agent_system/.../alfworld/projection.py`` at revision
2cf2fad.  The upstream file is Apache-2.0.  Training/evaluation in the
SEED-compatible protocol must execute the upstream function itself.
"""

from __future__ import annotations

import re


def alfworld_projection_reference(actions: list[str], require_think: bool = True) -> tuple[list[str], list[int]]:
    projected = list(actions)
    valid = [0] * len(projected)
    for index, original in enumerate(projected):
        lowered = original.lower()
        start = lowered.find("<action>")
        end = lowered.find("</action>")
        if start == -1 or end == -1:
            projected[index] = lowered[-30:]
            continue
        projected[index] = lowered[start + len("<action>") : end].strip().lower()
        valid[index] = 1
        if require_think and ("<think>" not in original or "</think>" not in original):
            valid[index] = 0
        if re.search(r"[\u4e00-\u9fff]", original):
            valid[index] = 0
    return projected, valid
