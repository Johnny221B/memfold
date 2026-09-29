#!/usr/bin/env python3
"""DDP soft-memory SEED training with static or synchronized text teachers."""

from __future__ import annotations

PROCEDURE = "on_policy_optimization"

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from memory_opd.compressed_opd import (
    load_compressed_opd_examples,
    soft_reader_messages,
    text_reader_messages,
)
from memory_opd.opd.seed_loss import seed_sampled_token_opd_loss
from memory_opd.opd.soft_seed import group_normalized_advantages, soft_seed_mixed_loss
from memory_opd.data.rewards import parse_choice
from policy_utils import (
    cached_soft_memory,
    chat_ids,
    load_bridge,
    load_policy,
    teacher_token_log_probs,
)


def token_log_probs(model, policy, prefix: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    full = torch.cat((prefix, policy.get_input_embeddings()(response)), dim=1)
    attention = torch.ones(full.shape[:2], device=full.device, dtype=torch.long)
    logits = model(inputs_embeds=full, attention_mask=attention, use_cache=False).logits.float()
    start = prefix.shape[1] - 1
    return torch.log_softmax(
        logits[:, start : start + response.shape[1]], dim=-1
    ).gather(-1, response[..., None]).squeeze(-1)


@torch.no_grad()
def reference_token_log_probs(
    reference_policy, prefix: torch.Tensor, response: torch.Tensor
) -> torch.Tensor:
    """Score the frozen KL reference without retaining a graph through prefix."""
    return token_log_probs(
        reference_policy, reference_policy, prefix.detach(), response.detach()
    )


def select_teacher_policy(policy, reference_policy, update_mode: str):
    """Select the text teacher without conflating it with the KL reference."""
    if update_mode == "frozen_initial":
        return reference_policy
    if update_mode == "seed_sync":
        return policy
    raise ValueError(f"unsupported teacher update mode: {update_mode}")


def teacher_log_probs_for_update(
    teacher_policy,
    prompt: torch.Tensor,
    response: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Score with an update-local frozen teacher and restore its train mode."""
    was_training = teacher_policy.training
    teacher_policy.eval()
    try:
        return teacher_token_log_probs(teacher_policy, prompt, response, response_mask)
    finally:
        teacher_policy.train(was_training)


@torch.no_grad()
def sample_group(
    policy,
    prefix: torch.Tensor,
    tokenizer,
    *,
    maximum_tokens: int,
    group_size: int,
    temperature: float,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    was_training = policy.training
    policy.eval()
    embeddings = prefix.repeat(group_size, 1, 1)
    sampled: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    alive = torch.ones(group_size, device=embeddings.device, dtype=torch.bool)
    for _ in range(maximum_tokens):
        attention = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long)
        logits = policy(
            inputs_embeds=embeddings, attention_mask=attention, use_cache=False
        ).logits[:, -1].float()
        if temperature <= 0:
            token = logits.argmax(-1)
        else:
            sorted_logits, sorted_indices = torch.sort(
                logits / temperature, descending=True, dim=-1
            )
            probabilities = F.softmax(sorted_logits, dim=-1)
            remove = probabilities.cumsum(dim=-1) - probabilities > top_p
            probabilities = F.softmax(
                sorted_logits.masked_fill(remove, float("-inf")), dim=-1
            )
            sampled_index = torch.multinomial(probabilities, 1).squeeze(-1)
            token = sorted_indices.gather(-1, sampled_index[:, None]).squeeze(-1)
        masks.append(alive.long())
        if tokenizer.eos_token_id is not None:
            token = torch.where(alive, token, torch.full_like(token, tokenizer.eos_token_id))
        sampled.append(token)
        embeddings = torch.cat(
            (embeddings, policy.get_input_embeddings()(token[:, None])), dim=1
        )
        if tokenizer.eos_token_id is not None:
            alive = alive & (token != tokenizer.eos_token_id)
            if not bool(torch.any(alive)):
                break
    if was_training:
        policy.train()
    return torch.stack(sampled, dim=1), torch.stack(masks, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--self-memories", type=Path, required=True)
    parser.add_argument("--teacher-memories", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--compressor-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("opd_only", "opd_grpo"), required=True)
    parser.add_argument("--expected-split", default="train")
    parser.add_argument("--expected-examples", type=int, default=489)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--opd-weight", type=float, default=0.01)
    parser.add_argument("--grpo-weight", type=float, default=1.0)
    parser.add_argument("--reference-kl-weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher-update-mode",
        choices=("frozen_initial", "seed_sync"),
        default="frozen_initial",
        help=(
            "frozen_initial keeps the reader initialization policy as the text teacher; "
            "seed_sync uses the current pre-update student as the no-grad text "
            "teacher while retaining a frozen reader initialization KL reference"
        ),
    )
    parser.add_argument("--gate-beta", type=float, default=5.0)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--sampling-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--maximum-response-tokens", type=int, default=5)
    parser.add_argument("--expected-token-count", type=int, default=256)
    parser.add_argument("--maximum-updates-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.procedure = "on_policy_optimization"
    if args.mode == "opd_grpo" and args.group_size < 2:
        parser.error("opd_grpo requires group-size >= 2")
    if not 0 < args.top_p <= 1:
        parser.error("top-p must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 1:
        raise ValueError(f"invalid DDP world size: {world_size}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    examples = load_compressed_opd_examples(
        args.questions, args.self_memories, expected_split=args.expected_split
    )
    teacher_examples = load_compressed_opd_examples(
        args.questions, args.teacher_memories, expected_split=args.expected_split
    )
    if len(examples) != args.expected_examples:
        raise ValueError(f"expected {args.expected_examples}, found {len(examples)}")
    example_by_id = {item.question_id: item for item in examples}
    teacher_memory_by_id = {item.question_id: item.memory_text for item in teacher_examples}
    expected_ids = set(example_by_id)
    if set(teacher_memory_by_id) != expected_ids:
        raise ValueError("teacher and student memory IDs differ")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = load_policy(args.model, args.initial_adapter, device, trainable=True)
    reference_policy = load_policy(args.model, args.initial_adapter, device, trainable=False)
    ddp = DistributedDataParallel(
        policy, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False, find_unused_parameters=False,
    )
    bridge, bridge_config = load_bridge(args.compressor_checkpoint, device)
    bridge.eval().requires_grad_(False)
    if int(bridge_config["token_count"]) != args.expected_token_count:
        raise ValueError(
            f"expected K={args.expected_token_count}, bridge has K={bridge_config['token_count']}"
        )
    trainable = [parameter for parameter in ddp.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    updates_per_epoch = math.ceil(len(examples) / world_size)
    if args.maximum_updates_per_epoch:
        updates_per_epoch = min(updates_per_epoch, args.maximum_updates_per_epoch)
    group_size = args.group_size if args.mode == "opd_grpo" else 1
    if rank == 0:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite: {args.output}")
        args.output.mkdir(parents=True)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update({
            "world_size": world_size,
            "questions_per_global_update": world_size,
            "rollouts_per_question": group_size,
            "rollouts_per_global_update": world_size * group_size,
            "updates_per_epoch": updates_per_epoch,
            "total_updates": updates_per_epoch * args.epochs,
            "teacher_update_mode": args.teacher_update_mode,
            "teacher_frozen_for_entire_run": args.teacher_update_mode == "frozen_initial",
            "teacher_frozen_within_update": True,
            "teacher_refresh_interval_updates": (
                None if args.teacher_update_mode == "frozen_initial" else 1
            ),
            "teacher_memory_refresh_mode": "static_input_file",
            "reference_policy_frozen": True,
            "compressor_frozen": True,
            "gold_used_only_as_grpo_reward": args.mode == "opd_grpo",
            "teacher_and_student_memory_content_identical": all(
                teacher_memory_by_id[key] == example_by_id[key].memory_text for key in expected_ids
            ),
        })
        (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    dist.barrier()

    metrics_path = args.output / "metrics.jsonl"
    global_update = 0
    for epoch in range(args.epochs):
        order = sorted(expected_ids)
        random.Random(args.seed + epoch).shuffle(order)
        slots = updates_per_epoch * world_size
        padded = order + order[: slots - len(order)]
        for update in range(updates_per_epoch):
            global_update += 1
            question_id = padded[update * world_size + rank]
            example = example_by_id[question_id]
            soft = cached_soft_memory(args.cache_dir, example, bridge, device)
            prompt = chat_ids(tokenizer, soft_reader_messages(example.question, example.options), device)
            prefix = torch.cat((soft, policy.get_input_embeddings()(prompt)), dim=1)
            response, response_mask = sample_group(
                policy, prefix, tokenizer, maximum_tokens=args.maximum_response_tokens,
                group_size=group_size,
                temperature=args.sampling_temperature if args.mode == "opd_grpo" else 0.0,
                top_p=args.top_p,
            )
            prefix = prefix.repeat(group_size, 1, 1)
            teacher_prompt = chat_ids(
                tokenizer,
                text_reader_messages(
                    teacher_memory_by_id[question_id], example.question, example.options
                ),
                device,
            ).repeat(group_size, 1)
            teacher_policy = select_teacher_policy(
                policy, reference_policy, args.teacher_update_mode
            )
            teacher_logp = teacher_log_probs_for_update(
                teacher_policy, teacher_prompt, response, response_mask
            )
            reference_logp = None
            if args.mode == "opd_grpo" and args.reference_kl_weight > 0:
                # Score frozen targets before constructing the trainable policy graph.
                # The detach also prevents gradients through policy prompt embeddings.
                reference_logp = reference_token_log_probs(
                    reference_policy, prefix, response
                )
            current_logp = token_log_probs(ddp, policy, prefix, response)
            texts = [
                tokenizer.decode(tokens[mask.bool()], skip_special_tokens=True).strip()
                for tokens, mask in zip(response, response_mask)
            ]
            rewards = torch.tensor(
                [float(parse_choice(text) == example.gold_label) for text in texts], device=device
            )
            advantages, advantage_metrics = group_normalized_advantages(rewards)
            if args.mode == "opd_grpo":
                loss, mixed = soft_seed_mixed_loss(
                    current_log_prob=current_logp,
                    old_log_prob=current_logp.detach(),
                    teacher_log_prob=teacher_logp,
                    reference_log_prob=reference_logp,
                    advantages=advantages,
                    response_mask=response_mask,
                    opd_weight=args.opd_weight,
                    grpo_weight=args.grpo_weight,
                    reference_kl_weight=args.reference_kl_weight,
                    gate_beta=args.gate_beta,
                    clip_range=args.clip_range,
                )
                opd, grpo, reference_kl = mixed.opd_loss, mixed.grpo_loss, mixed.reference_kl
                gate, gap = mixed.gate_mean, mixed.teacher_gap_mean
            else:
                opd, opd_metrics = seed_sampled_token_opd_loss(
                    current_logp, teacher_logp, response_mask, gate_beta=args.gate_beta
                )
                loss = args.opd_weight * opd
                grpo = reference_kl = loss.detach() * 0
                gate, gap = opd_metrics.gate_mean, opd_metrics.teacher_gap_mean
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            values = torch.tensor([
                float(loss.detach()), float(opd), float(grpo), float(reference_kl),
                float(gate), float(gap), float(rewards.mean()),
                float(advantage_metrics.zero_variance), advantage_metrics.reward_variance,
            ], device=device)
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= world_size
            if rank == 0:
                metric = {
                    "epoch": epoch + 1, "update_in_epoch": update + 1,
                    "global_update": global_update,
                    "teacher_update_mode": args.teacher_update_mode,
                    "mean_loss": float(values[0]), "mean_opd": float(values[1]),
                    "mean_grpo": float(values[2]), "mean_reference_kl": float(values[3]),
                    "mean_gate": float(values[4]), "mean_teacher_gap": float(values[5]),
                    "sampled_answer_accuracy_for_audit_only": float(values[6]),
                    "zero_variance_group_ratio": float(values[7]),
                    "mean_reward_variance": float(values[8]),
                    "rank0_response_group": texts, "rank0_question_id": question_id,
                    "rank0_gradient_norm": float(grad_norm),
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metric) + "\n")
                print(json.dumps(metric), flush=True)
        dist.barrier()
        if rank == 0:
            adapter = args.output / "checkpoints" / f"epoch-{epoch + 1}" / "adapter"
            adapter.mkdir(parents=True)
            policy.save_pretrained(adapter, safe_serialization=True)
        dist.barrier()

    if rank == 0:
        result = {"procedure": PROCEDURE,
            "status": "completed", "mode": args.mode, "epochs": args.epochs,
            "teacher_update_mode": args.teacher_update_mode,
            "total_updates": global_update,
            "final_adapter": str(args.output / "checkpoints" / f"epoch-{args.epochs}" / "adapter"),
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
