#!/usr/bin/env python3
"""Train one LoRA to read frozen soft self memory with a frozen text teacher."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_opd.compressed_opd import (
    CompressedOPDExample,
    load_compressed_opd_examples,
    soft_reader_messages,
    text_reader_messages,
)
from memory_opd.opd.seed_loss import seed_sampled_token_opd_loss
from memory_opd.data.rewards import parse_choice
from memory_opd.soft_reconstruction import (
    ContextResampler,
    ContextToSoftTokens,
    SoftTokenProjector,
    text_memory_state_id,
)


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
    bridge = bridge.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    return bridge, config


def cached_soft_memory(
    cache_dir: Path,
    example: CompressedOPDExample,
    bridge: ContextToSoftTokens,
    device: torch.device,
) -> torch.Tensor:
    state_id = text_memory_state_id(example.question_id, example.memory_text)
    path = cache_dir / "states" / f"{state_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing self-memory state cache: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    states = payload["states"].to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    mask = torch.ones(states.shape[:2], device=device, dtype=torch.long)
    with torch.no_grad():
        return bridge(states, mask)


def chat_ids(tokenizer: Any, messages: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return torch.tensor([ids], device=device, dtype=torch.long)


@torch.no_grad()
def rollout_student(
    model: Any,
    prefix: torch.Tensor,
    tokenizer: Any,
    maximum_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    was_training = model.training
    model.eval()
    embeddings = prefix
    sampled: list[torch.Tensor] = []
    for _ in range(maximum_tokens):
        attention = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long)
        token = model(inputs_embeds=embeddings, attention_mask=attention, use_cache=False).logits[:, -1].argmax(-1)
        sampled.append(token)
        embeddings = torch.cat((embeddings, model.get_input_embeddings()(token[:, None])), dim=1)
        if tokenizer.eos_token_id is not None and bool(torch.all(token == tokenizer.eos_token_id)):
            break
    if was_training:
        model.train()
    response = torch.stack(sampled, dim=1)
    mask = torch.ones_like(response)
    if tokenizer.eos_token_id is not None:
        positions = (response[0] == tokenizer.eos_token_id).nonzero()
        if positions.numel():
            mask[:, int(positions[0]) + 1 :] = 0
    return response, mask


def student_token_log_probs(model: Any, prefix: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    full = torch.cat((prefix, model.get_input_embeddings()(response)), dim=1)
    attention = torch.ones(full.shape[:2], device=full.device, dtype=torch.long)
    logits = model(inputs_embeds=full, attention_mask=attention, use_cache=False).logits.float()
    start = prefix.shape[1] - 1
    selected = logits[:, start : start + response.shape[1]]
    return F.log_softmax(selected, dim=-1).gather(-1, response[..., None]).squeeze(-1)


@torch.no_grad()
def teacher_token_log_probs(
    model: Any,
    prompt: torch.Tensor,
    response: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    teacher_device = prompt.device
    response = response.to(teacher_device)
    response_mask = response_mask.to(teacher_device)
    input_ids = torch.cat((prompt, response), dim=1)
    attention = torch.cat((torch.ones_like(prompt), response_mask), dim=1)
    logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits.float()
    start = prompt.shape[1] - 1
    selected = logits[:, start : start + response.shape[1]]
    return F.log_softmax(selected, dim=-1).gather(-1, response[..., None]).squeeze(-1)


def load_policy(
    model_path: Path,
    adapter_path: Path,
    device: torch.device,
    *,
    trainable: bool,
) -> PeftModel:
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    base.requires_grad_(False)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=trainable).to(device)
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
    else:
        model.requires_grad_(False).eval()
    return model


def reasoning_reader_messages(
    question: str,
    options: tuple[str, str, str, str],
    *,
    memory_text: str | None,
) -> list[dict[str, str]]:
    rendered = "\n".join(
        f"({chr(97 + index)}) {option}" for index, option in enumerate(options)
    )
    instruction = (
        "Give exactly one evidence-grounded reasoning sentence of at most 30 words. "
        "Then, on a new line, end with `Final answer: (a)`, `(b)`, `(c)`, or `(d)`."
    )
    if memory_text is None:
        return [
            {
                "role": "system",
                "content": (
                    "Continuous soft-memory tokens precede this conversation. "
                    "Use them as the only memory evidence."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{instruction}\n\nQUESTION:\n{question}\n\nOPTIONS:\n{rendered}"
                ),
            },
        ]
    return [
        {
            "role": "system",
            "content": "You are a careful memory-grounded reasoning assistant.",
        },
        {
            "role": "user",
            "content": (
                f"{instruction}\n\nMEMORY:\n{memory_text}\n\n"
                f"QUESTION:\n{question}\n\nOPTIONS:\n{rendered}"
            ),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--self-memories", type=Path, required=True)
    parser.add_argument(
        "--teacher-memories", type=Path,
        help="Privileged text memories for the frozen teacher; defaults to self memories.",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--compressor-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", default="train")
    parser.add_argument("--expected-examples", type=int, default=489)
    parser.add_argument("--student-device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--opd-weight", type=float, default=0.01)
    parser.add_argument("--gate-beta", type=float, default=5.0)
    parser.add_argument(
        "--chain-text-weight", type=float, default=0.0,
        help=(
            "When positive, train the shared text branch from the privileged teacher "
            "and the soft branch from the detached self-text branch."
        ),
    )
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--maximum-response-tokens", type=int, default=5)
    parser.add_argument("--expected-token-count", type=int, default=256)
    parser.add_argument(
        "--response-format",
        choices=("label", "reasoning-answer"),
        default="label",
    )
    parser.add_argument("--maximum-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output}")
    if min(args.epochs, args.learning_rate, args.gradient_accumulation,
           args.maximum_response_tokens) <= 0:
        parser.error("epochs, learning rate, accumulation, and response tokens must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    student_device = torch.device(args.student_device)
    teacher_device = torch.device(args.teacher_device)
    examples = load_compressed_opd_examples(
        args.questions, args.self_memories, expected_split=args.expected_split
    )
    teacher_examples = load_compressed_opd_examples(
        args.questions,
        args.teacher_memories or args.self_memories,
        expected_split=args.expected_split,
    )
    teacher_memory_by_id = {
        example.question_id: example.memory_text for example in teacher_examples
    }
    if args.expected_examples and len(examples) != args.expected_examples:
        raise ValueError(f"expected {args.expected_examples} examples, found {len(examples)}")
    if set(teacher_memory_by_id) != {example.question_id for example in examples}:
        raise ValueError("teacher and student memory IDs differ")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    student = load_policy(
        args.model, args.initial_adapter, student_device, trainable=True
    )
    teacher = load_policy(
        args.model, args.initial_adapter, teacher_device, trainable=False
    )
    bridge, bridge_config = load_bridge(args.compressor_checkpoint, student_device)
    if int(bridge_config["token_count"]) != args.expected_token_count:
        raise ValueError(
            f"expected K={args.expected_token_count}, bridge has K={bridge_config['token_count']}"
        )
    if int(bridge_config["lm_dim"]) != int(student.config.hidden_size):
        raise ValueError("bridge LM dimension does not match the student backbone")

    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    trainable_names = [name for name, parameter in student.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable_names):
        raise RuntimeError("only student LoRA parameters may be trainable")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "model": str(args.model),
        "initial_adapter": str(args.initial_adapter),
        "questions": str(args.questions),
        "self_memories": str(args.self_memories),
        "teacher_memories": str(args.teacher_memories or args.self_memories),
        "cache_dir": str(args.cache_dir),
        "compressor_checkpoint": str(args.compressor_checkpoint),
        "output": str(args.output),
        "teacher_frozen": True,
        "teacher_and_student_initial_adapter_identical": True,
        "teacher_and_student_memory_content_identical": all(
            teacher_memory_by_id[example.question_id] == example.memory_text
            for example in examples
        ),
        "training_mode": "chain" if args.chain_text_weight > 0 else "direct",
        "compressor_frozen": True,
        "memory_encoder_frozen": True,
        "gold_answer_supervised": False,
        "token_count": int(bridge_config["token_count"]),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }
    (args.output / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )

    maximum_steps = args.maximum_steps or args.epochs * len(examples)
    metrics_path = args.output / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    order: list[CompressedOPDExample] = []
    for step in range(maximum_steps):
        index = step % len(examples)
        if index == 0:
            epoch = step // len(examples)
            order = list(examples)
            random.Random(args.seed + epoch).shuffle(order)
        example = order[index]
        soft = cached_soft_memory(args.cache_dir, example, bridge, student_device)
        student_messages = (
            reasoning_reader_messages(
                example.question, example.options, memory_text=None
            )
            if args.response_format == "reasoning-answer"
            else soft_reader_messages(example.question, example.options)
        )
        student_prompt = chat_ids(tokenizer, student_messages, student_device)
        student_prefix = torch.cat(
            (soft, student.get_input_embeddings()(student_prompt)), dim=1
        )
        response, response_mask = rollout_student(
            student, student_prefix, tokenizer, args.maximum_response_tokens
        )
        student_logp = student_token_log_probs(student, student_prefix, response)
        teacher_messages = (
            reasoning_reader_messages(
                example.question,
                example.options,
                memory_text=teacher_memory_by_id[example.question_id],
            )
            if args.response_format == "reasoning-answer"
            else text_reader_messages(
                teacher_memory_by_id[example.question_id],
                example.question,
                example.options,
            )
        )
        teacher_prompt = chat_ids(tokenizer, teacher_messages, teacher_device)
        teacher_logp = teacher_token_log_probs(
            teacher,
            teacher_prompt,
            response,
            response_mask,
        ).to(student_device)
        if args.chain_text_weight > 0:
            text_messages = (
                reasoning_reader_messages(
                    example.question,
                    example.options,
                    memory_text=example.memory_text,
                )
                if args.response_format == "reasoning-answer"
                else text_reader_messages(
                    example.memory_text, example.question, example.options
                )
            )
            text_prompt = chat_ids(tokenizer, text_messages, student_device)
            text_prefix = student.get_input_embeddings()(text_prompt)
            text_student_logp = student_token_log_probs(
                student, text_prefix, response
            )
            text_opd, _ = seed_sampled_token_opd_loss(
                text_student_logp,
                teacher_logp,
                response_mask,
                gate_beta=args.gate_beta,
            )
            soft_opd, opd_metrics = seed_sampled_token_opd_loss(
                student_logp,
                text_student_logp.detach(),
                response_mask,
                gate_beta=args.gate_beta,
            )
            opd = text_opd + args.chain_text_weight * soft_opd
        else:
            text_opd = torch.zeros((), device=student_device)
            soft_opd, opd_metrics = seed_sampled_token_opd_loss(
                student_logp,
                teacher_logp,
                response_mask,
                gate_beta=args.gate_beta,
            )
            opd = soft_opd
        loss = args.opd_weight * opd
        epoch_index = step % len(examples)
        epoch_base = step - epoch_index
        window_start = (
            epoch_index // args.gradient_accumulation
        ) * args.gradient_accumulation
        window_size = min(
            args.gradient_accumulation,
            len(examples) - window_start,
            maximum_steps - (epoch_base + window_start),
        )
        (loss / window_size).backward()
        update = (
            (epoch_index + 1) % args.gradient_accumulation == 0
            or epoch_index + 1 == len(examples)
            or step + 1 == maximum_steps
        )
        grad_norm = None
        if update:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite student LoRA gradient norm")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        response_text = tokenizer.decode(
            response[0][response_mask[0].bool()], skip_special_tokens=True
        ).strip()
        metric = {
            "step": step + 1,
            "epoch": step // len(examples) + 1,
            "question_id": example.question_id,
            "loss": float(loss.detach()),
            "opd": float(opd.detach()),
            "text_opd": float(text_opd.detach()),
            "soft_opd": float(soft_opd.detach()),
            "gate": float(opd_metrics.gate_mean),
            "gate_active_ratio": float(opd_metrics.gate_active_ratio),
            "teacher_gap": float(opd_metrics.teacher_gap_mean),
            "response_tokens": int(response_mask.sum()),
            "response": response_text,
            "parsed": parse_choice(response_text),
            "correct_for_audit_only": parse_choice(response_text) == example.gold_label,
            "update": update,
            "gradient_norm": float(grad_norm) if grad_norm is not None else None,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric, ensure_ascii=False) + "\n")
        print(json.dumps(metric, ensure_ascii=False), flush=True)

        epoch_boundary = (step + 1) % len(examples) == 0 or step + 1 == maximum_steps
        if epoch_boundary:
            epoch_number = (step // len(examples)) + 1
            checkpoint = args.output / "checkpoints" / f"epoch-{epoch_number}"
            checkpoint.mkdir(parents=True, exist_ok=False)
            student.save_pretrained(checkpoint / "adapter", safe_serialization=True)
            torch.save(
                {
                    "step": step + 1,
                    "epoch": epoch_number,
                    "optimizer": optimizer.state_dict(),
                    "rng": torch.get_rng_state(),
                },
                checkpoint / "training_state.pt",
            )

    result = {
        "status": "completed",
        "steps": maximum_steps,
        "epochs_completed": maximum_steps / len(examples),
        "final_adapter": str(checkpoint / "adapter"),
        "last_metric": metric,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
