from __future__ import annotations

from .data import PersonaQuestion


def summary_prompt(session: str) -> str:
    return (
        "Summarize this dialogue session concisely. Preserve concrete user facts, preferences, "
        "experiences, dates, and changes. Do not invent information.\n\n" + session + "\n\nSUMMARY:"
    )


def personality_prompt(session: str) -> str:
    return (
        "Infer the user's stable personality, preferences, emotions, and an appropriate response "
        "strategy from this dialogue. Be concise and do not invent unsupported facts.\n\n"
        + session
        + "\n\nUSER PROFILE AND RESPONSE STRATEGY:"
    )


def answer_prompt(question: PersonaQuestion, memories: list[str]) -> str:
    memory_text = "\n\n".join(f"MEMORY {i + 1}:\n{text}" for i, text in enumerate(memories))
    options = "\n".join(f"({letter}) {text}" for letter, text in zip("abcd", question.options))
    return f"""Answer the multiple-choice question using the retrieved user memories.

QUESTION:
{question.question}

RETRIEVED MEMORY:
{memory_text}

Output exactly one option: (a), (b), (c), or (d), and nothing else.

OPTIONS:
{options}

ANSWER:"""
