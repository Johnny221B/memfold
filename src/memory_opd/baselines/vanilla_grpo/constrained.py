"""Finite token tries for stochastic sampling over admissible ALFWorld actions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def build_prefix_map(
    sequences: Sequence[Sequence[int]], eos_token_ids: Iterable[int]
) -> dict[tuple[int, ...], list[int]]:
    """Map every generated prefix to its valid next tokens."""

    eos = {int(value) for value in eos_token_ids}
    if not eos:
        raise ValueError("at least one EOS token id is required")
    allowed: dict[tuple[int, ...], set[int]] = {}
    for raw_sequence in sequences:
        sequence = tuple(int(value) for value in raw_sequence)
        if not sequence:
            raise ValueError("admissible action token sequence must not be empty")
        for index, token in enumerate(sequence):
            allowed.setdefault(sequence[:index], set()).add(token)
        allowed.setdefault(sequence, set()).update(eos)
    if not allowed:
        raise ValueError("at least one admissible action is required")
    return {prefix: sorted(tokens) for prefix, tokens in allowed.items()}


def allowed_next_tokens(
    prefix_map: dict[tuple[int, ...], list[int]], generated_prefix: Sequence[int]
) -> list[int]:
    prefix = tuple(int(value) for value in generated_prefix)
    if prefix not in prefix_map:
        raise ValueError(f"generated prefix is outside admissible action trie: {prefix}")
    return prefix_map[prefix]
