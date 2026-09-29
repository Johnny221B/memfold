#!/usr/bin/env python3
"""Build resumable session-level persona memories without loading QA labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_opd.baselines.memp.persona import load_contexts, load_questions, render_session, session_slices


SYSTEM = (
    "Convert the supplied prior user-assistant session into one concise persona memory. "
    "Record only explicit user facts, preferences, constraints, changes, and their temporal order. "
    "Do not infer unsupported traits, answer any future question, or add outside knowledge."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    test_ids = set(json.loads(args.split.read_text())["question_ids"]["test"])
    questions = [q for q in load_questions(args.questions, include_labels=False) if q.question_id in test_ids]
    contexts = load_contexts(args.contexts)
    required = set()
    for question in questions:
        for start, end in session_slices(contexts[question.context_id], question.end_index):
            required.add((question.context_id, start, end))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [item for item in sorted(required) if not (args.output_dir / f"{item[0]}_{item[1]}_{item[2]}.json").exists()]
    print(json.dumps({"required": len(required), "pending": len(pending), "label_fields_read": False}), flush=True)
    if not pending:
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    for ordinal, (context_id, start, end) in enumerate(pending, 1):
        session = render_session(contexts[context_id], start, end)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": session}]
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            enable_thinking=False,
        ).to(args.device)
        with torch.inference_mode():
            output = model.generate(
                **encoded, do_sample=False, num_beams=1, max_new_tokens=256,
                temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.eos_token_id,
            )
        memory = tokenizer.decode(output[0, encoded.input_ids.shape[1]:], skip_special_tokens=True).strip()
        if not memory:
            raise RuntimeError("empty persona memory")
        record = {
            "context_id": context_id, "start_index": start, "end_index": end,
            "memory": memory, "source_sha256": hashlib.sha256(session.encode()).hexdigest(),
            "builder": args.model, "do_sample": False, "label_fields_read": False,
        }
        path = args.output_dir / f"{context_id}_{start}_{end}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"completed": ordinal, "total": len(pending)}), flush=True)


if __name__ == "__main__":
    main()
