"""Shared contracts for the RQ2 PersonaMem GRPO and OPSD baselines."""

from .personamem import PersonaMemExample, load_questions, split_by_context
from .rewards import parse_choice, strict_choice_reward

__all__ = [
    "PersonaMemExample",
    "load_questions",
    "parse_choice",
    "split_by_context",
    "strict_choice_reward",
]
