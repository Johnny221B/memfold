"""Rule rewards shared by the full-history OPSD/GRPO QA baselines."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

CHOICE_PATTERN = re.compile(r"^\([abcd]\)$")

def parse_choice(value: object) -> str | None:
    """Return a canonical choice only when the entire output is valid."""

    if not isinstance(value, str):
        return None
    candidate = value.strip()
    return candidate if CHOICE_PATTERN.fullmatch(candidate) else None

def strict_choice_reward(prediction: object, gold: object) -> float:
    """Binary exact-match reward used by the Vanilla GRPO baseline."""

    parsed_prediction = parse_choice(prediction)
    parsed_gold = parse_choice(gold)
    if parsed_gold is None:
        raise ValueError(f"gold answer must be one of (a), (b), (c), (d): {gold!r}")
    return float(parsed_prediction == parsed_gold)

def _answer_text(value: Any) -> str:
    """Extract a LaMP-style answer while leaving ordinary text untouched."""

    text = str(value or "").strip()
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text
    if isinstance(parsed, dict) and "personalized_answer" in parsed:
        return str(parsed["personalized_answer"]).strip()
    return text

def normalized_token_f1(completion: str, gold: str) -> float:
    """Deterministic dense reward for free-form LoCoMo and LaMP-QA answers."""

    def tokens(value: Any) -> list[str]:
        return re.findall(r"[\w]+", _answer_text(value).casefold(), flags=re.UNICODE)

    predicted = tokens(completion)
    expected = tokens(gold)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)

def qa_reward(completion: str, gold: str, reward_type: str) -> float:
    if reward_type == "strict_choice":
        return strict_choice_reward(completion, gold)
    if reward_type == "normalized_token_f1":
        return normalized_token_f1(completion, gold)
    raise ValueError(f"unsupported reward_type: {reward_type!r}")
