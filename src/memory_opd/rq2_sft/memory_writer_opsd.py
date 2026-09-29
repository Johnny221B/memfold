"""Leakage-safe views for memory-writer on-policy distillation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from memory_opd.rq2_baselines.personamem import LABELS, PersonaMemExample, load_questions


PRIVILEGED_MARKER = "PRIVILEGED TRAINING QUESTIONS (TEACHER ONLY)"
TEACHER_INSTRUCTION = (
    "Use the privileged training questions only to judge which reusable facts in the preceding "
    "history matter. Score the student's generated memory as a compact, time-aware JSON memory. "
    "Do not answer the questions and do not copy question or option wording into the memory."
)
TEACHER_SYSTEM = (
    "Judge the generated compact, time-aware user memory using the history and privileged training "
    "questions. Prefer valid JSON with stable, current, and changes fields. Do not answer or copy questions."
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _render_privileged(question_ids: Sequence[str], by_id: Mapping[str, PersonaMemExample]) -> str:
    blocks = [PRIVILEGED_MARKER]
    for index, question_id in enumerate(question_ids, start=1):
        item = by_id[question_id]
        answer_text = item.options[LABELS.index(item.answer)]
        blocks.append(
            f"[Q{index}] Type: {item.question_type}\nQuestion: {item.question}\nCorrect answer: {answer_text}"
        )
    return "\n\n".join(blocks)


def build_memory_writer_records(
    requests_path: Path, compact_sft_path: Path, questions_path: Path
) -> list[dict[str, Any]]:
    requests = {row["task_id"]: row for row in _jsonl(requests_path)}
    questions = {
        item.question_id: item for item in load_questions(questions_path)}
    result: list[dict[str, Any]] = []
    for row in _jsonl(compact_sft_path):
        task_id = row["id"]
        request = requests.get(task_id)
        if request is None:
            raise ValueError(f"missing request for compact SFT row {task_id}")
        if row.get("split") != "train" or request.get("split") != "train":
            raise ValueError(
                f"non-training row in memory-writer data: {task_id}")
        metadata = row["metadata"]
        for key in ("shared_context_id", "history_end_index", "history_sha256"):
            if metadata[key] != request[key]:
                raise ValueError(f"{key} mismatch for {task_id}")
        student_messages = row["messages"][:-1]
        if student_messages != request["messages"]:
            raise ValueError(
                f"student history differs from annotation request for {task_id}")
        question_ids = request["question_ids"]
        if not question_ids or any(qid not in questions for qid in question_ids):
            raise ValueError(f"unknown/empty question IDs for {task_id}")
        if any(questions[qid].shared_context_id != request["shared_context_id"] for qid in question_ids):
            raise ValueError(f"question context mismatch for {task_id}")
        student_serialized = json.dumps(student_messages, ensure_ascii=False)
        if PRIVILEGED_MARKER in student_serialized:
            raise ValueError(
                f"privileged marker leaked into student prompt for {task_id}")
        result.append(
            {
                "id": task_id,
                "student_messages": student_messages,
                "teacher_messages": [
                    {"role": "system", "content": TEACHER_SYSTEM},
                    *student_messages[1:],
                    {"role": "user", "content": _render_privileged(
                        question_ids, questions)},
                ],
                "target": row["messages"][-1]["content"],
                # Required by the SFTTrainer parent even though preprocessing is disabled.
                "completion": row["messages"][-1]["content"],
                "question_ids": question_ids,
            }
        )
    if len(result) != len(requests):
        raise ValueError(
            f"compact/request row count mismatch: {len(result)} != {len(requests)}")
    return result


class MemoryWriterOPSDCollator:
    """Build official OPSD tensors plus a compact-target SFT anchor view."""

    def __init__(self, tokenizer: Any, *, max_prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        tokenizer.padding_side = "right"

    def _chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def _ids(self, texts: Sequence[str]) -> tuple[list[list[int]], list[int]]:
        ids = self.tokenizer(list(texts), padding=False,
                             truncation=False)["input_ids"]
        lengths = [len(value) for value in ids]
        if max(lengths) > self.max_prompt_length:
            raise ValueError(
                f"prompt length {max(lengths)} exceeds budget {self.max_prompt_length}; truncation is forbidden"
            )
        return ids, lengths

    def _pad(self, ids: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(map(len, ids))
        pad = self.tokenizer.pad_token_id
        values = [list(item) + [pad] * (width - len(item)) for item in ids]
        masks = [[1] * len(item) + [0] * (width - len(item)) for item in ids]
        return torch.tensor(values), torch.tensor(masks)

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        student_texts = [self._chat(row["student_messages"])
                         for row in features]
        teacher_texts = [self._chat(row["teacher_messages"])
                         for row in features]
        student_ids, student_lengths = self._ids(student_texts)
        teacher_ids, teacher_lengths = self._ids(teacher_texts)
        student, student_mask = self._pad(student_ids)
        teacher, teacher_mask = self._pad(teacher_ids)
        return {
            "student_prompts": student,
            "student_prompt_attention_mask": student_mask,
            "student_prompt_length": student.shape[1],
            "student_prompt_lengths_per_example": torch.tensor(student_lengths),
            "teacher_prompts": teacher,
            "teacher_prompt_attention_mask": teacher_mask,
            "teacher_prompt_length": teacher.shape[1],
            "teacher_prompt_lengths_per_example": torch.tensor(teacher_lengths),
        }
