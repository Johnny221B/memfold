#!/usr/bin/env python3
"""Generate sealed PersonaMem predictions from a trained MemGen Weaver checkpoint."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import GenerationConfig

from memory_opd.baselines.memgen.persona_data import SYSTEM, load_contexts, render_history
from train_memgen_strict_weaver_sft import build_model


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_option(text: str) -> str | None:
    candidate = text.strip().lower()
    return candidate if candidate in {"(a)", "(b)", "(c)", "(d)"} else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("output and manifest paths must be new")

    split = json.loads(args.split.read_text(encoding="utf-8"))
    test_ids = list(split["question_ids"]["test"])
    if args.limit is not None:
        test_ids = test_ids[: args.limit]
    allowed = set(test_ids)
    # Deliberate label firewall: correct_answer is never retained or referenced.
    rows = {}
    with args.questions.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["question_id"] in allowed:
                rows[raw["question_id"]] = {
                    key: raw[key] for key in (
                        "question_id", "shared_context_id", "end_index_in_shared_context",
                        "user_question_or_message", "all_options",
                    )
                }
    if set(rows) != allowed:
        raise ValueError("test IDs do not match question rows")
    contexts = load_contexts(args.contexts)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = build_model(args.model, True, False)
    state_path = args.checkpoint / "pytorch_model.bin"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected checkpoint keys: {incompatible.unexpected_keys}")
    model.to(device).eval()
    tokenizer = model.tokenizer
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    generation.trigger_do_sample = False
    generation.weaver_do_sample = False
    generation.temperature = 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total_tokens = 0
    with args.output.open("x", encoding="utf-8") as output:
        for ordinal, question_id in enumerate(test_ids):
            row = rows[question_id]
            history = contexts[row["shared_context_id"]][: int(row["end_index_in_shared_context"])]
            options = list(ast.literal_eval(row["all_options"]))
            choices = "\n".join(
                f"({chr(ord('a') + index)}) {option}" for index, option in enumerate(options)
            )
            user = (
                f"PRIOR CONVERSATION:\n{render_history(history)}\n\n"
                f"QUESTION:\n{row['user_question_or_message']}\n\n"
                f"OPTIONS:\n{choices}\n\nANSWER:"
            )
            encoded = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            )
            input_tokens = int(encoded["attention_mask"].sum())
            if input_tokens > args.max_input_tokens:
                record = {
                    "question_id": question_id, "status": "overflow", "input_tokens": input_tokens,
                    "generated_tokens": 0, "prediction": None, "parsed_prediction": None,
                }
            else:
                # Weaver SFT runs these FP32 projection modules under BF16 autocast;
                # upstream MemGen.generate does not establish that context itself.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    generated = model.generate(
                        encoded["input_ids"].to(device), encoded["attention_mask"].to(device),
                        generation_config=generation,
                    )
                generated_ids = generated[0, encoded["input_ids"].shape[1]:]
                prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
                record = {
                    "question_id": question_id, "status": "ok", "input_tokens": input_tokens,
                    "generated_tokens": int(generated_ids.numel()), "prediction": prediction,
                    "parsed_prediction": strict_option(prediction),
                }
            total_tokens += record["input_tokens"] + record["generated_tokens"]
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"{ordinal + 1}/{len(test_ids)} {question_id} {record['status']}", flush=True)

    manifest = {
        "schema_version": "memgen_personamem_eval_v1",
        "stage": "weaver_sft_trigger_inactive_always_augment",
        "records": len(test_ids), "limit": args.limit,
        "questions_sha256": digest(args.questions), "contexts_sha256": digest(args.contexts),
        "split_sha256": digest(args.split), "checkpoint_state_sha256": digest(state_path),
        "output_sha256": digest(args.output), "mean_total_tokens": total_tokens / len(test_ids),
        "seconds": time.time() - started,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
