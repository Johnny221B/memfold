import json

import pytest

from memory_opd.compressed_opd import (
    deterministic_option_order,
    load_compressed_opd_examples,
    parse_memory_text,
    permute_options,
    soft_reader_messages,
    text_reader_messages,
)


MEMORY = {
    "evidence": ["The user likes jazz."],
    "temporal_relations": [],
    "derived_facts": ["Jazz is a stable preference."],
}


def test_teacher_and_student_use_identical_content_but_only_teacher_sees_text():
    text = json.dumps(MEMORY)
    teacher = json.dumps(text_reader_messages(text, "Question?", ["A", "B", "C", "D"]))
    student = json.dumps(soft_reader_messages("Question?", ["A", "B", "C", "D"]))

    assert "The user likes jazz" in teacher
    assert "The user likes jazz" not in student
    assert "Question?" in teacher and "Question?" in student


def test_memory_schema_and_option_permutation_are_strict():
    assert parse_memory_text(json.dumps(MEMORY)) == MEMORY
    with pytest.raises(ValueError, match="exactly"):
        parse_memory_text(json.dumps({**MEMORY, "answer": "(a)"}))

    first = deterministic_option_order("q1", 2)
    assert first == deterministic_option_order("q1", 2)
    options, expected = permute_options(("A", "B", "C", "D"), "(c)", first)
    assert options[first.index(2)] == "C"
    assert expected == ("(a)", "(b)", "(c)", "(d)")[first.index(2)]


def test_loader_joins_one_self_memory_per_official_question(tmp_path):
    questions = tmp_path / "questions.jsonl"
    memories = tmp_path / "memories.jsonl"
    questions.write_text(json.dumps({
        "question_id": "q1", "shared_context_id": "c1", "split": "train",
        "question_type": "recall", "question": "Question?",
        "options": ["A", "B", "C", "D"], "answer": "(b)",
    }) + "\n")
    memories.write_text(json.dumps({
        "question_id": "q1", "split": "train", "memory": MEMORY,
    }) + "\n")

    rows = load_compressed_opd_examples(
        questions, memories, expected_split="train"
    )

    assert len(rows) == 1
    assert rows[0].question_id == "q1"
    assert rows[0].question_type == "recall"
    assert rows[0].gold_label == "(b)"
    assert "Question?" not in rows[0].memory_text
