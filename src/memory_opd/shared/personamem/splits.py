from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Iterable

from .data import PersonaMemExample


def _select_exact(groups: list[tuple[str, int]], target: int) -> set[str]:
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group_id, size in groups:
        for total, chosen in sorted(list(reachable.items()), reverse=True):
            new_total = total + size
            if new_total <= target and new_total not in reachable:
                reachable[new_total] = chosen + (group_id,)
    if target not in reachable:
        raise ValueError(f"cannot allocate exactly {target} questions by shared_context_id")
    return set(reachable[target])


def build_group_split(
    examples: Iterable[PersonaMemExample],
    *,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Allocate whole context groups using IDs/counts only, never labels."""
    examples = list(examples)
    if len(examples) != train_count + val_count + test_count:
        raise ValueError("requested split counts do not match question count")
    sizes = Counter(item.shared_context_id for item in examples)
    groups = list(sizes.items())
    random.Random(seed).shuffle(groups)
    test_groups = _select_exact(groups, test_count)
    remaining = [item for item in groups if item[0] not in test_groups]
    val_groups = _select_exact(remaining, val_count)
    train_groups = set(sizes) - test_groups - val_groups
    split_groups = {"train": train_groups, "val": val_groups, "test": test_groups}
    result = {
        name: sorted(
            (item.question_id for item in examples if item.shared_context_id in ids),
            key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
        )
        for name, ids in split_groups.items()
    }
    assert [len(result[name]) for name in ("train", "val", "test")] == [
        train_count,
        val_count,
        test_count,
    ]
    assert not (train_groups & val_groups or train_groups & test_groups or val_groups & test_groups)
    return result
