"""QA adapters for the tested TRL GRPO and official OPSD trainers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .rewards import qa_reward

def load_prepared_records(path: Path, *, allowed_splits: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            split = record.get("split")
            if split not in allowed_splits:
                raise ValueError(f"row {line_number} has forbidden split {split!r}")
            if not all(key in record for key in ("prompt", "answer", "question_id")):
                raise ValueError(f"row {line_number} lacks prompt/answer/question_id")
            records.append(record)
    if not records:
        raise ValueError(f"no prepared records in {path}")
    return records

def trl_grpo_reward(
    completions: Sequence[Any],
    answer: Sequence[str],
    reward_type: Sequence[str] | str | None = None,
    **_: Any,
) -> list[float]:
    """Dataset-aware TRL callback for strict-choice and free-form QA."""

    if len(completions) != len(answer):
        raise ValueError("completion and answer batch lengths differ")
    texts: list[str] = []
    for completion in completions:
        if isinstance(completion, str):
            texts.append(completion)
        elif isinstance(completion, list) and completion and isinstance(completion[-1], Mapping):
            texts.append(str(completion[-1].get("content", "")))
        else:
            texts.append("")
    kinds = (
        [reward_type or "strict_choice"] * len(texts)
        if isinstance(reward_type, (str, type(None)))
        else list(reward_type)
    )
    if len(kinds) != len(texts):
        raise ValueError("reward_type and completion batch lengths differ")
    return [qa_reward(text, gold, kind) for text, gold, kind in zip(texts, answer, kinds)]

class QAOPSDDataCollator:
    """Produce the exact tensor keys consumed by official ``OPSDTrainer``."""

    choice_transition = (
        "The reference answer above is correct. Use it only as privileged guidance, then "
        "answer the original multiple-choice question independently. Output exactly one "
        "option: (a), (b), (c), or (d), and nothing else."
    )
    freeform_transition = (
        "The reference answer above is privileged guidance. Answer the original task "
        "independently and obey its requested output format."
    )

    def __init__(self, tokenizer: Any, *, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.padding_side = "right"

    def _chat(self, content: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _encode_untruncated(self, prompts: list[str]) -> tuple[list[list[int]], list[int]]:
        encoded = self.tokenizer(prompts, padding=False, truncation=False)
        input_ids = encoded["input_ids"]
        lengths = [len(ids) for ids in input_ids]
        if max(lengths) > self.max_length:
            raise ValueError(
                f"prompt length {max(lengths)} exceeds configured max_length={self.max_length}; "
                "silent truncation is forbidden"
            )
        return input_ids, lengths

    def _pad(self, prompts: list[str], max_length: int) -> Mapping[str, torch.Tensor]:
        return self.tokenizer(
            prompts,
            padding="max_length",
            truncation=False,
            max_length=max_length,
            return_tensors="pt",
        )

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("OPSD collator received an empty batch")
        student_texts = [self._chat(str(row["prompt"])) for row in features]
        teacher_texts = [
            self._chat(
                f"{row['prompt']}\n\nPRIVILEGED REFERENCE ANSWER:\n{row['answer']}\n\n"
                f"{self.choice_transition if row.get('reward_type', 'strict_choice') == 'strict_choice' else self.freeform_transition}"
            )
            for row in features
        ]
        _, student_lengths = self._encode_untruncated(student_texts)
        _, teacher_lengths = self._encode_untruncated(teacher_texts)
        max_student = max(student_lengths)
        max_teacher = max(teacher_lengths)
        student = self._pad(student_texts, max_student)
        teacher = self._pad(teacher_texts, max_teacher)
        return {
            "student_prompts": student["input_ids"],
            "student_prompt_attention_mask": student["attention_mask"],
            "student_prompt_length": max_student,
            "student_prompt_lengths_per_example": torch.tensor(student_lengths),
            "teacher_prompts": teacher["input_ids"],
            "teacher_prompt_attention_mask": teacher["attention_mask"],
            "teacher_prompt_length": max_teacher,
            "teacher_prompt_lengths_per_example": torch.tensor(teacher_lengths),
        }

# Backward-compatible public name used by the original RQ2 entrypoint.
PersonaMemOPSDDataCollator = QAOPSDDataCollator
