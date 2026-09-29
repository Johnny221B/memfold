#!/usr/bin/env python3
"""Jointly retain writer/text-memory skills while teaching one LoRA to read soft memory."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_opd.soft_reconstruction import load_text_memory_examples, serialize_memory
from train_auxiliary_reasoning_adaptation import cached_state, load_bridge, reasoning_inputs


MEMORY_BUDGET = (
    "\n\nOUTPUT BUDGET: Return no more than 8 evidence items, 4 temporal_relations "
    "items, and 4 derived_facts items. Keep every item at most 160 characters. "
    "Finish the complete JSON object within 1200 tokens."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def options_text(question: dict[str, Any]) -> str:
    return "\n".join(
        f"({chr(ord('a') + index)}) {text}"
        for index, text in enumerate(question["options"])
    )


def labeled_text_batch(
    model: Any,
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    target: str,
    device: torch.device,
    maximum_target_tokens: int,
    maximum_sequence_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    target_ids = tokenizer(
        target + (tokenizer.eos_token or ""), add_special_tokens=False
    ).input_ids[:maximum_target_tokens]
    if not target_ids:
        raise ValueError("empty target")
    room = maximum_sequence_tokens - len(target_ids)
    if room <= 0:
        raise ValueError("target exceeds maximum sequence length")
    # Long writer histories are left-truncated only if they exceed the model budget.
    prompt_ids = prompt_ids[-room:]
    ids = torch.tensor([prompt_ids + target_ids], device=device, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    labels[:, len(prompt_ids):] = torch.tensor(target_ids, device=device)
    return ids, labels, len(target_ids)


def text_reasoning_messages(question: dict[str, Any], memory: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a careful memory-grounded reasoning assistant."},
        {
            "role": "user",
            "content": (
                "Use the supplied memory to give concise evidence-grounded reasoning. "
                "End with `Final answer: (a)`, `(b)`, `(c)`, or `(d)`.\n\n"
                f"MEMORY:\n{serialize_memory(memory)}\n\n"
                f"QUESTION:\n{question['question']}\n\nOPTIONS:\n{options_text(question)}"
            ),
        },
    ]


def exact_schedule(steps: int, text_ratio: float, writer_ratio: float, seed: int) -> list[str]:
    writer = round(steps * writer_ratio)
    text = round(steps * text_ratio)
    if writer + text > steps:
        raise ValueError("text and writer ratios sum to more than one")
    schedule = ["writer"] * writer + ["text"] * text + ["soft"] * (steps - writer - text)
    random.Random(seed).shuffle(schedule)
    return schedule


def supervised_reasoning_target(
    tokenizer: Any, reasoning: str, answer: str, maximum_tokens: int
) -> str:
    """Keep the final answer in-budget even when the trace is long."""
    suffix = f"\nFinal answer: {answer}"
    suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
    budget = maximum_tokens - len(suffix_ids) - 1
    if budget <= 0:
        raise ValueError("target budget cannot fit answer suffix")
    ids = tokenizer(reasoning, add_special_tokens=False).input_ids[:budget]
    return tokenizer.decode(ids, skip_special_tokens=True).rstrip() + suffix


def answer_reasoning_inputs(
    model: Any, tokenizer: Any, soft: torch.Tensor, question: dict[str, Any],
    target: str, maximum_target_tokens: int,
):
    options = options_text(question)
    messages = [
        {"role": "system", "content": "Continuous soft-memory tokens precede this conversation. Use them as the only memory evidence."},
        {"role": "user", "content": (
            "Give concise evidence-grounded reasoning and end with `Final answer: (a)`, `(b)`, `(c)`, or `(d)`.\n\n"
            f"QUESTION:\n{question['question']}\n\nOPTIONS:\n{options}"
        )},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    prompt = torch.tensor([prompt_ids], device=soft.device, dtype=torch.long)
    target_ids = tokenizer(
        target + (tokenizer.eos_token or ""), add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(soft.device)[:, :maximum_target_tokens]
    embedding = model.get_input_embeddings()
    inputs = torch.cat((soft, embedding(prompt), embedding(target_ids)), dim=1)
    labels = torch.full(inputs.shape[:2], -100, device=soft.device, dtype=torch.long)
    labels[:, soft.shape[1] + prompt.shape[1]:] = target_ids
    return inputs, torch.ones(inputs.shape[:2], device=soft.device, dtype=torch.long), labels, int(target_ids.numel())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--writer-inputs", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--compressor-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--steps", type=int, default=489)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--text-ratio", type=float, default=0.0)
    parser.add_argument("--writer-ratio", type=float, default=0.0)
    parser.add_argument("--ranking-weight", type=float, default=0.2)
    parser.add_argument("--null-ranking-weight", type=float, default=0.2)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument("--maximum-reasoning-tokens", type=int, default=256)
    parser.add_argument("--maximum-writer-tokens", type=int, default=1200)
    parser.add_argument("--maximum-sequence-tokens", type=int, default=40960)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output}")
    if args.steps <= 0 or min(args.text_ratio, args.writer_ratio) < 0:
        parser.error("steps must be positive and ratios nonnegative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    questions = read_jsonl(args.questions)
    memories = read_jsonl(args.memories)
    traces = read_jsonl(args.traces)
    writers = read_jsonl(args.writer_inputs)
    question_by_id = {str(row["question_id"]): row for row in questions}
    memory_by_id = {str(row["question_id"]): row for row in memories}
    trace_by_id = {str(row["question_id"]): str(row["reasoning"]) for row in traces}
    writer_by_id = {str(row.get("task_id", row.get("id"))): row for row in writers}
    expected = set(question_by_id)
    for name, mapping in (("memory", memory_by_id), ("trace", trace_by_id), ("writer", writer_by_id)):
        if set(mapping) != expected:
            raise ValueError(f"{name} IDs differ from question IDs")
    if any(len(mapping) != len(expected) for mapping in (question_by_id, memory_by_id, trace_by_id, writer_by_id)):
        raise ValueError("duplicate IDs")
    examples, _ = load_text_memory_examples(args.questions, args.memories)
    example_by_id = {example.question_id: example for example in examples}
    ordered_ids = sorted(expected)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, args.initial_adapter, is_trainable=True).to(device).train()
    bridge, bridge_config = load_bridge(args.compressor_checkpoint, device)
    bridge.eval().requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)

    schedule = exact_schedule(args.steps, args.text_ratio, args.writer_ratio, args.seed)
    task_counts = {task: schedule.count(task) for task in ("soft", "text", "writer")}
    args.output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        "task_counts": task_counts,
        "token_count": int(bridge_config["token_count"]),
        "compressor_trainable": False,
        "answer_tokens_supervised": True,
        "one_shared_lora": True,
    })
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    per_task_order: dict[str, list[str]] = {}
    per_task_position = {task: 0 for task in task_counts}
    for task in task_counts:
        order = list(ordered_ids)
        random.Random(args.seed + {"soft": 1, "text": 2, "writer": 3}[task]).shuffle(order)
        per_task_order[task] = order
    metrics_path = args.output / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    for step, task in enumerate(schedule, 1):
        position = per_task_position[task]
        question_id = per_task_order[task][position % len(ordered_ids)]
        per_task_position[task] += 1
        question = question_by_id[question_id]
        reasoning = supervised_reasoning_target(
            tokenizer, trace_by_id[question_id], str(question["answer"]),
            args.maximum_reasoning_tokens,
        )
        target_tokens = 0
        ranking_loss = torch.zeros((), device=device)
        null_ranking_loss = torch.zeros((), device=device)
        shuffled_loss_value = None
        null_loss_value = None

        if task == "soft":
            example = example_by_id[question_id]
            negative_id = ordered_ids[(ordered_ids.index(question_id) + 1 + step * 97) % len(ordered_ids)]
            if negative_id == question_id:
                negative_id = ordered_ids[(ordered_ids.index(question_id) + 1) % len(ordered_ids)]
            states, mask = cached_state(args.cache_dir, example, device)
            negative_states, negative_mask = cached_state(args.cache_dir, example_by_id[negative_id], device)
            with torch.no_grad():
                soft = bridge(states, mask)
                negative_soft = bridge(negative_states, negative_mask)
            inputs, attention, labels, target_tokens = answer_reasoning_inputs(
                model, tokenizer, soft, question, reasoning, args.maximum_reasoning_tokens
            )
            own_loss = model(inputs_embeds=inputs, attention_mask=attention, labels=labels, use_cache=False).loss
            negative_inputs, negative_attention, negative_labels, _ = answer_reasoning_inputs(
                model, tokenizer, negative_soft, question, reasoning, args.maximum_reasoning_tokens
            )
            shuffled_loss = model(
                inputs_embeds=negative_inputs, attention_mask=negative_attention,
                labels=negative_labels, use_cache=False,
            ).loss
            ranking_loss = torch.relu(args.ranking_margin + own_loss - shuffled_loss)
            null_inputs, null_attention, null_labels, _ = answer_reasoning_inputs(
                model, tokenizer, torch.zeros_like(soft), question, reasoning,
                args.maximum_reasoning_tokens,
            )
            null_loss = model(
                inputs_embeds=null_inputs, attention_mask=null_attention,
                labels=null_labels, use_cache=False,
            ).loss
            null_ranking_loss = torch.relu(args.ranking_margin + own_loss - null_loss)
            loss = own_loss + args.ranking_weight * ranking_loss + args.null_ranking_weight * null_ranking_loss
            shuffled_loss_value = float(shuffled_loss.detach())
            null_loss_value = float(null_loss.detach())
        elif task == "text":
            ids, labels, target_tokens = labeled_text_batch(
                model, tokenizer, text_reasoning_messages(question, memory_by_id[question_id]["memory"]),
                reasoning, device, args.maximum_reasoning_tokens, args.maximum_sequence_tokens,
            )
            own_loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss
            loss = own_loss
        else:
            writer = writer_by_id[question_id]
            if "writer_messages" in writer:
                messages = [dict(item) for item in writer["writer_messages"]]
                messages[0]["content"] = str(messages[0]["content"]) + MEMORY_BUDGET
            else:
                source_messages = writer["messages"]
                if [item["role"] for item in source_messages[:3]] != ["system", "user", "assistant"]:
                    raise ValueError(f"unexpected writer messages: {question_id}")
                messages = [dict(item) for item in source_messages[:2]]
            ids, labels, target_tokens = labeled_text_batch(
                model, tokenizer, messages, serialize_memory(memory_by_id[question_id]["memory"]),
                device, args.maximum_writer_tokens, args.maximum_sequence_tokens,
            )
            own_loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss
            loss = own_loss

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metric = {
            "step": step, "task": task, "question_id": question_id,
            "loss": float(loss.detach()), "own_loss": float(own_loss.detach()),
            "shuffled_loss": shuffled_loss_value, "null_loss": null_loss_value,
            "ranking_loss": float(ranking_loss.detach()),
            "null_ranking_loss": float(null_ranking_loss.detach()),
            "gradient_norm": float(grad_norm), "target_tokens": target_tokens,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric) + "\n")
        print(json.dumps(metric), flush=True)

    adapter_dir = args.output / "checkpoints" / "epoch-1" / "adapter"
    adapter_dir.mkdir(parents=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    result = {"status": "completed", "steps": args.steps, "adapter": str(adapter_dir), "task_counts": task_counts}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
