#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from memory_opd.baselines.memorybank.data import (
    load_contexts,
    load_questions,
    load_split_ids,
    prior_messages,
    select_questions,
)
from memory_opd.baselines.memorybank.generation import TransformersGenerator
from memory_opd.baselines.memorybank.prompts import answer_prompt, personality_prompt, summary_prompt
from memory_opd.baselines.memorybank.retrieval import (
    MemoryBank,
    MemoryEntry,
    extract_persona,
    render_messages,
    session_blocks,
)

STRICT_ANSWER = re.compile(r"^\([abcd]\)$")
UPSTREAM_COMMIT = "cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MemoryBank baseline for PersonaMem-v1")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "val", "test"), default="test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turns-per-session", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--summary-max-new-tokens", type=int, default=160)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="validate data/split boundaries without loading a model")
    parser.add_argument(
        "--memory-mode",
        choices=("summary", "raw-session"),
        default="summary",
        help="raw-session is a cheap diagnostic, not the primary MemoryBank reproduction",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["question_id"] for line in path.read_text(encoding="utf-8").splitlines() if line}


def build_memories(question, messages, generator, args) -> tuple[list[MemoryEntry], int]:
    cache_dir = args.output / "memory_cache"
    cache_path = cache_dir / f"{question.question_id}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return [MemoryEntry(**entry) for entry in payload["entries"]], payload["summary_input_tokens"]
    entries: list[MemoryEntry] = []
    summary_input_tokens = 0
    persona = extract_persona(messages)
    if persona:
        entries.append(MemoryEntry("persona", "overall_personality", persona, -1))
    for index, block in enumerate(session_blocks(messages, args.turns_per_session)):
        rendered = render_messages(block)
        if args.memory_mode == "raw-session":
            entries.append(MemoryEntry(f"session-{index:04d}", "raw_session", rendered, index))
            continue
        summary, used = generator.generate(summary_prompt(rendered), args.summary_max_new_tokens)
        profile, profile_used = generator.generate(personality_prompt(rendered), args.summary_max_new_tokens)
        summary_input_tokens += used + profile_used
        entries.append(MemoryEntry(f"summary-{index:04d}", "session_summary", summary, index))
        entries.append(MemoryEntry(f"profile-{index:04d}", "session_personality", profile, index))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"entries": [entry.to_dict() for entry in entries], "summary_input_tokens": summary_input_tokens},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return entries, summary_input_tokens


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / "predictions.jsonl"
    done = load_done(predictions_path) if args.resume else set()
    ids = load_split_ids(args.split, args.partition)
    questions = select_questions(load_questions(args.questions), ids, args.limit)
    wanted_contexts = {question.shared_context_id for question in questions}
    contexts = load_contexts(args.contexts, wanted_contexts)
    manifest = {
        "method": "MemoryBank-PersonaMem adaptation",
        "upstream_commit": UPSTREAM_COMMIT,
        "questions": str(args.questions),
        "questions_sha256": sha256(args.questions),
        "contexts": str(args.contexts),
        "contexts_sha256": sha256(args.contexts),
        "split": str(args.split),
        "split_sha256": sha256(args.split),
        "partition": args.partition,
        "model": args.model,
        "memory_mode": args.memory_mode,
        "turns_per_session": args.turns_per_session,
        "top_k": args.top_k,
        "forgetting_curve": "disabled (no meaningful PersonaMem wall-clock dates; deterministic primary run)",
        "future_context_policy": "messages[:end_index_in_shared_context]",
        "strict_answer_regex": STRICT_ANSWER.pattern,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.validate_only:
        validation = {
            "status": "validated",
            "questions": len(questions),
            "shared_context_ids": len(wanted_contexts),
            "minimum_prior_messages": min(len(prior_messages(q, contexts)) for q in questions),
            "maximum_prior_messages": max(len(prior_messages(q, contexts)) for q in questions),
        }
        (args.output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        print(json.dumps(validation))
        return
    generator = TransformersGenerator(args.model, args.max_input_tokens)
    started = time.time()
    mode = "a" if args.resume else "w"
    with predictions_path.open(mode, encoding="utf-8") as handle:
        for index, question in enumerate(questions, 1):
            if question.question_id in done:
                continue
            messages = prior_messages(question, contexts)
            entries, summary_tokens = build_memories(question, messages, generator, args)
            bank = MemoryBank(entries)
            query = question.question + "\n" + "\n".join(question.options)
            retrieved = bank.search(query, args.top_k)
            prompt = answer_prompt(question, [entry.text for entry, _ in retrieved])
            prediction, answer_input_tokens = generator.generate(prompt, 5)
            parsed = prediction if STRICT_ANSWER.fullmatch(prediction) else None
            row = {
                "question_id": question.question_id,
                "shared_context_id": question.shared_context_id,
                "prediction": prediction,
                "parsed_answer": parsed,
                "gold": question.answer,
                "correct": parsed == question.answer,
                "memory_entries": len(entries),
                "retrieved": [
                    {"memory_id": entry.memory_id, "kind": entry.kind, "score": score}
                    for entry, score in retrieved
                ],
                "summary_input_tokens": summary_tokens,
                "answer_input_tokens": answer_input_tokens,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"progress": f"{index}/{len(questions)}", **row}), flush=True)
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line]
    result = {
        "status": "completed",
        "questions": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / max(1, len(rows)),
        "invalid_predictions": sum(row["parsed_answer"] is None for row in rows),
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
