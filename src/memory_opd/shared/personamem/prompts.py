from __future__ import annotations

from .data import PersonaMemExample


def render_memory(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[{str(message.get('role', 'unknown')).upper()}]\n{message.get('content', '')}"
        for message in messages
    )


def render_prompt(example: PersonaMemExample, memory: str) -> str:
    choices = "\n".join(
        f"({chr(ord('a') + index)}) {choice}" for index, choice in enumerate(example.options)
    )
    return (
        f"QUESTION:\n{example.question}\n\n"
        f"RETRIEVED MEMORY:\n{memory}\n\n"
        "Output exactly one option: (a), (b), (c), or (d), and nothing else.\n\n"
        f"OPTIONS:\n{choices}\n\nANSWER:"
    )
