#!/usr/bin/env python3
"""Train a K-token context bridge and Qwen LoRA on reasoning-only targets."""

from __future__ import annotations

PROCEDURE = "auxiliary_reasoning_adaptation"

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_opd.soft_reconstruction import (
    ContextResampler,
    ContextToSoftTokens,
    ReconstructionExample,
    SoftTokenProjector,
    cross_context_separation_loss,
    different_context_example,
    load_reconstruction_examples,
    load_text_memory_examples,
)


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_bridge(path: Path, device: torch.device) -> tuple[ContextToSoftTokens, dict[str, Any]]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved["config"]
    bridge = ContextToSoftTokens(
        ContextResampler(
            int(config["context_dim"]),
            latent_dim=int(config["latent_dim"]),
            token_count=int(config["token_count"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
            context_residual=bool(config.get("context_residual", False)),
        ),
        SoftTokenProjector(int(config["latent_dim"]), int(config["lm_dim"])),
    )
    bridge.load_state_dict(saved["bridge"])
    return bridge.to(device=device, dtype=torch.bfloat16), config


def cached_state(cache_dir: Path, example: ReconstructionExample, device: torch.device):
    payload = torch.load(
        cache_dir / "states" / f"{example.state_id}.pt",
        map_location="cpu",
        weights_only=True,
    )
    states = payload["states"].to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    mask = torch.ones(states.shape[:2], device=device, dtype=torch.long)
    return states, mask


def prompt_ids(tokenizer: Any, question: dict[str, Any], device: torch.device) -> torch.Tensor:
    options = "\n".join(
        f"({chr(ord('a') + index)}) {text}"
        for index, text in enumerate(question["options"])
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Continuous soft-memory tokens precede this conversation. "
                "Use their information to reason about the question."
            ),
        },
        {
            "role": "user",
            "content": (
                "Give concise evidence-grounded reasoning. Do not state a final answer.\n\n"
                f"QUESTION:\n{question['question']}\n\nOPTIONS:\n{options}"
            ),
        },
    ]
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return torch.tensor([ids], device=device, dtype=torch.long)


def reasoning_inputs(
    model: Any,
    tokenizer: Any,
    soft: torch.Tensor,
    question: dict[str, Any],
    reasoning: str,
    maximum_target_tokens: int,
):
    prompt = prompt_ids(tokenizer, question, soft.device)
    target = tokenizer(
        reasoning + (tokenizer.eos_token or ""),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(soft.device)
    target = target[:, :maximum_target_tokens]
    if target.numel() == 0:
        raise ValueError(f"empty reasoning target for {question['question_id']}")
    embedding = model.get_input_embeddings()
    inputs = torch.cat((soft, embedding(prompt), embedding(target)), dim=1)
    labels = torch.full(inputs.shape[:2], -100, device=soft.device, dtype=torch.long)
    labels[:, soft.shape[1] + prompt.shape[1] :] = target
    attention = torch.ones(inputs.shape[:2], device=soft.device, dtype=torch.long)
    return inputs, attention, labels, int(target.numel())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path)
    parser.add_argument("--input-mode", choices=("context", "text-memory"), default="context")
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--compressor-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--compressor-learning-rate", type=float, default=3e-5)
    parser.add_argument("--freeze-resampler", action="store_true")
    parser.add_argument("--ranking-weight", type=float, default=0.2)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument("--separation-weight", type=float, default=0.1)
    parser.add_argument("--maximum-cross-context-cosine", type=float, default=0.8)
    parser.add_argument("--maximum-target-tokens", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.procedure = "auxiliary_reasoning_adaptation"
    if args.steps <= 0:
        parser.error("--steps must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    questions = read_jsonl(args.questions)
    question_by_id = {str(row["question_id"]): row for row in questions}
    traces = read_jsonl(args.traces)
    trace_by_id = {str(row["question_id"]): str(row["reasoning"]) for row in traces}
    if len(question_by_id) != len(questions) or len(trace_by_id) != len(traces):
        raise ValueError("duplicate question/trace ID")
    if set(question_by_id) != set(trace_by_id):
        raise ValueError("question and trace ID sets differ")
    if args.input_mode == "context":
        if args.contexts is None:
            parser.error("--contexts is required when --input-mode=context")
        examples, _ = load_reconstruction_examples(args.questions, args.contexts, args.memories)
    else:
        examples, _ = load_text_memory_examples(args.questions, args.memories)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.lora_checkpoint is not None:
        model = PeftModel.from_pretrained(
            base, args.lora_checkpoint, is_trainable=True
        )
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_rank,
                lora_alpha=2 * args.lora_rank,
                lora_dropout=0.0,
                bias="none",
                target_modules=list(LORA_TARGETS),
            ),
        )
    model = model.to(device).train()
    bridge, bridge_config = load_bridge(args.compressor_checkpoint, device)
    if args.freeze_resampler:
        bridge.resampler.requires_grad_(False)
    bridge.train()
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in bridge.parameters() if p.requires_grad], "lr": args.compressor_learning_rate},
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.learning_rate},
        ],
        weight_decay=0.01,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "model": str(args.model),
        "questions": str(args.questions),
        "contexts": str(args.contexts) if args.contexts is not None else None,
        "input_mode": args.input_mode,
        "memories": str(args.memories),
        "traces": str(args.traces),
        "cache_dir": str(args.cache_dir),
        "compressor_checkpoint": str(args.compressor_checkpoint),
        "output": str(args.output),
        "token_count": int(bridge_config["token_count"]),
        "answer_tokens_supervised": False,
        "train_examples": len(examples),
    }
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    order: list[ReconstructionExample] = []
    metrics_path = args.output / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        index = (step - 1) % len(examples)
        if index == 0:
            local_epoch = (step - 1) // len(examples)
            order = list(examples)
            random.Random(args.seed + local_epoch).shuffle(order)
            index = 0
        example = order[index]
        negative = different_context_example(order, index)
        states, mask = cached_state(args.cache_dir, example, device)
        negative_states, negative_mask = cached_state(args.cache_dir, negative, device)
        soft = bridge(states, mask)
        negative_soft = bridge(negative_states, negative_mask)
        question = question_by_id[example.question_id]
        reasoning = trace_by_id[example.question_id]
        own_inputs, own_attention, own_labels, target_tokens = reasoning_inputs(
            model, tokenizer, soft, question, reasoning, args.maximum_target_tokens
        )
        auxiliary_reasoning_loss = model(
            inputs_embeds=own_inputs,
            attention_mask=own_attention,
            labels=own_labels,
            use_cache=False,
        ).loss
        shuffled_inputs, shuffled_attention, shuffled_labels, _ = reasoning_inputs(
            model, tokenizer, negative_soft, question, reasoning, args.maximum_target_tokens
        )
        shuffled_loss = model(
            inputs_embeds=shuffled_inputs,
            attention_mask=shuffled_attention,
            labels=shuffled_labels,
            use_cache=False,
        ).loss
        ranking_loss = torch.relu(args.ranking_margin + auxiliary_reasoning_loss - shuffled_loss)
        separation_loss, cosine = cross_context_separation_loss(
            soft,
            negative_soft,
            maximum_cosine=args.maximum_cross_context_cosine,
        )
        objective = (
            auxiliary_reasoning_loss
            + args.ranking_weight * ranking_loss
            + args.separation_weight * separation_loss
        )
        objective.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], 1.0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metric = {"procedure": PROCEDURE,
            "step": step,
            "question_id": example.question_id,
            "negative_question_id": negative.question_id,
            "auxiliary_reasoning_loss": float(auxiliary_reasoning_loss.detach()),
            "shuffled_loss": float(shuffled_loss.detach()),
            "own_minus_shuffled": float((auxiliary_reasoning_loss - shuffled_loss).detach()),
            "ranking_loss": float(ranking_loss.detach()),
            "separation_loss": float(separation_loss.detach()),
            "cross_context_cosine": float(cosine.detach()),
            "objective": float(objective.detach()),
            "gradient_norm": float(grad_norm),
            "target_tokens": target_tokens,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric) + "\n")
        print(json.dumps(metric), flush=True)

    adapter_dir = args.output / f"lora-step-{args.steps:06d}"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    checkpoint = args.output / f"bridge-step-{args.steps:06d}.pt"
    torch.save(
        {"procedure": PROCEDURE,
            "bridge": bridge.state_dict(),
            "config": bridge_config,
            "auxiliary_reasoning_adaptation_steps": args.steps,
            "lora_adapter": str(adapter_dir),
        },
        checkpoint,
    )
    result = {"procedure": PROCEDURE, "status": "completed", "steps": args.steps, "bridge": str(checkpoint), "lora": str(adapter_dir), "last_metric": metric}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
