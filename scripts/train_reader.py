#!/usr/bin/env python3
"""DDP soft-reader SFT with one example per rank."""

from __future__ import annotations

PROCEDURE = "reader_initialization"

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_opd.soft_reconstruction import load_text_memory_examples
from memory_opd.writer.training import encode_sft_record
from reader_utils import (
    answer_reasoning_inputs,
    cached_state,
    load_bridge,
    read_jsonl,
    supervised_reasoning_target,
)


def reconstruction_inputs(
    policy: Any,
    tokenizer: Any,
    soft: torch.Tensor,
    question: dict[str, Any],
    target: str,
    maximum_target_tokens: int,
    *,
    include_question: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    instruction = (
        "Reconstruct the compact evidence memory encoded by the preceding soft "
        "tokens. Do not answer a question. Return only the memory JSON."
    )
    if include_question:
        instruction += f"\n\nQUESTION:\n{question['question']}"
    messages = [
        {
            "role": "system",
            "content": "Continuous soft-memory tokens precede this conversation.",
        },
        {"role": "user", "content": instruction},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt = torch.tensor([prompt_ids], device=soft.device, dtype=torch.long)
    target_ids = tokenizer(
        target + (tokenizer.eos_token or ""),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(soft.device)[:, :maximum_target_tokens]
    embedding = policy.get_input_embeddings()
    inputs = torch.cat((soft, embedding(prompt), embedding(target_ids)), dim=1)
    labels = torch.full(inputs.shape[:2], -100, device=soft.device, dtype=torch.long)
    labels[:, soft.shape[1] + prompt.shape[1] :] = target_ids
    attention = torch.ones(inputs.shape[:2], device=soft.device, dtype=torch.long)
    return inputs, attention, labels, int(target_ids.numel())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--compressor-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--epoch-offset",
        type=int,
        default=0,
        help="Number of already completed epochs; advances shuffling and checkpoint names.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--compressor-learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--compressor-trainable-scope",
        choices=("none", "projector", "last-resampler-projector"),
        default="last-resampler-projector",
    )
    parser.add_argument(
        "--soft-output-anchor-weight",
        type=float,
        default=0.1,
        help="Weight on relative MSE to the initial bridge soft tokens.",
    )
    parser.add_argument(
        "--qa-target",
        choices=("trace", "answer-only"),
        default="trace",
        help="answer-only never trains on an external reasoning trajectory.",
    )
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument(
        "--objective",
        choices=("qa", "reconstruct-question", "reconstruct-soft-only"),
        default="qa",
    )
    parser.add_argument("--maximum-target-tokens", type=int, default=256)
    parser.add_argument("--maximum-updates-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--writer-replay-data", type=Path)
    parser.add_argument("--writer-replay-ratio", type=float, default=0.0)
    parser.add_argument("--writer-maximum-sequence-tokens", type=int, default=40960)
    args = parser.parse_args()
    args.procedure = "reader_initialization"
    if args.epoch_offset < 0:
        parser.error("epoch offset must be non-negative")
    if not 0.0 <= args.writer_replay_ratio < 1.0:
        parser.error("writer replay ratio must be in [0, 1)")
    if bool(args.writer_replay_data) != bool(args.writer_replay_ratio):
        parser.error("writer replay data and a positive ratio must be supplied together")
    if args.compressor_learning_rate <= 0 or args.soft_output_anchor_weight < 0:
        parser.error("bridge learning rate must be positive and anchor weight nonnegative")
    if args.compressor_trainable_scope == "none" and args.soft_output_anchor_weight:
        parser.error("bridge anchor has no effect when the bridge is frozen")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 1:
        raise ValueError(f"invalid DDP world size: {world_size}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    questions = read_jsonl(args.questions)
    memories = read_jsonl(args.memories)
    traces = read_jsonl(args.traces)
    question_by_id = {str(row["question_id"]): row for row in questions}
    memory_ids = {str(row["question_id"]) for row in memories}
    trace_by_id = {str(row["question_id"]): row for row in traces}
    expected = set(question_by_id)
    if memory_ids != expected or set(trace_by_id) != expected:
        raise ValueError("question, memory, and trace IDs differ")
    examples, _ = load_text_memory_examples(args.questions, args.memories)
    example_by_id = {example.question_id: example for example in examples}
    negative_ids = {
        example.question_id: sorted(
            candidate.question_id
            for candidate in examples
            if candidate.context_id != example.context_id
        )
        for example in examples
    }
    if any(not candidates for candidates in negative_ids.values()):
        raise ValueError("ranking requires at least two distinct shared contexts")

    writer_by_id: dict[str, dict[str, Any]] = {}
    if args.writer_replay_data:
        writer_records = read_jsonl(args.writer_replay_data)
        writer_by_id = {str(row["id"]): row for row in writer_records}
        if set(writer_by_id) != expected or len(writer_by_id) != len(expected):
            raise ValueError("Mode A writer replay IDs differ from question IDs")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    policy = PeftModel.from_pretrained(
        base, args.initial_adapter, is_trainable=True
    ).to(device).train()
    model = DistributedDataParallel(
        policy,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    bridge, bridge_config = load_bridge(args.compressor_checkpoint, device)
    bridge_reference = None
    bridge_ddp = None
    bridge.requires_grad_(False)
    if args.compressor_trainable_scope != "none":
        bridge_reference = copy.deepcopy(bridge).eval().requires_grad_(False)
        for parameter in bridge.projector.parameters():
            parameter.requires_grad_(True)
        if args.compressor_trainable_scope == "last-resampler-projector":
            for parameter in bridge.resampler.decoder.layers[-1].parameters():
                parameter.requires_grad_(True)
            for parameter in bridge.resampler.output_norm.parameters():
                parameter.requires_grad_(True)
        bridge.train()
        bridge_ddp = DistributedDataParallel(
            bridge,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        bridge.eval()
    policy_trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    compressor_trainable = [
        parameter for parameter in bridge.parameters() if parameter.requires_grad
    ]
    trainable = policy_trainable + compressor_trainable
    parameter_groups: list[dict[str, Any]] = [
        {"params": policy_trainable, "lr": args.learning_rate}
    ]
    if compressor_trainable:
        parameter_groups.append(
            {"params": compressor_trainable, "lr": args.compressor_learning_rate}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01)

    updates_per_epoch = math.ceil(len(examples) / world_size)
    if args.maximum_updates_per_epoch > 0:
        updates_per_epoch = min(updates_per_epoch, args.maximum_updates_per_epoch)
    if rank == 0:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite: {args.output}")
        args.output.mkdir(parents=True)
        config: dict[str, Any] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        config.update(
            {
                "world_size": world_size,
                "micro_batch_per_gpu": 1,
                "effective_global_batch": world_size,
                "examples": len(examples),
                "updates_per_epoch": updates_per_epoch,
                "updates_this_run": updates_per_epoch * args.epochs,
                "initial_global_update": updates_per_epoch * args.epoch_offset,
                "final_global_update": updates_per_epoch * (args.epoch_offset + args.epochs),
                "token_count": int(bridge_config["token_count"]),
                "compressor_trainable": bool(compressor_trainable),
                "compressor_trainable_parameters": sum(
                    parameter.numel() for parameter in compressor_trainable
                ),
                "policy_trainable_parameters": sum(
                    parameter.numel() for parameter in policy_trainable
                ),
                "answer_tokens_supervised": args.objective == "qa",
                "external_reasoning_tokens_supervised": (
                    args.objective == "qa" and args.qa_target == "trace"
                ),
                "writer_replay_updates_per_epoch": round(
                    updates_per_epoch * args.writer_replay_ratio
                ),
            }
        )
        (args.output / "run_config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
    dist.barrier()

    metrics_path = args.output / "metrics.jsonl"
    global_update = updates_per_epoch * args.epoch_offset
    for local_epoch in range(args.epochs):
        epoch = args.epoch_offset + local_epoch
        epoch_number = epoch + 1
        order = sorted(expected)
        random.Random(args.seed + epoch).shuffle(order)
        padded = list(order)
        slots = updates_per_epoch * world_size
        padded = padded[:slots]
        if len(padded) < slots:
            padded.extend(order[: slots - len(padded)])
        writer_updates = round(updates_per_epoch * args.writer_replay_ratio)
        update_tasks = ["writer"] * writer_updates + ["soft"] * (
            updates_per_epoch - writer_updates
        )
        random.Random(args.seed + 10000 + epoch).shuffle(update_tasks)
        for update in range(updates_per_epoch):
            global_update += 1
            question_id = padded[update * world_size + rank]
            question = question_by_id[question_id]
            trace = trace_by_id[question_id]
            audit_correct = bool(trace.get("answer_correct_for_audit_only"))
            task = update_tasks[update]
            if args.objective == "qa":
                reasoning = (
                    str(trace["reasoning"])
                    if args.qa_target == "trace" and audit_correct
                    else ""
                )
                target = supervised_reasoning_target(
                    tokenizer,
                    reasoning,
                    str(question["answer"]),
                    args.maximum_target_tokens,
                )
            else:
                # The target is the exact compressor input, never an answer.
                target = example_by_id[question_id].memory_text
            negative_id: str | None = None
            ranking_enabled = 0.0
            ranking_loss = torch.zeros((), device=device)
            anchor_loss = torch.zeros((), device=device)
            relative_soft_drift = torch.zeros((), device=device)
            if task == "writer":
                input_ids, writer_labels, writer_audit = encode_sft_record(
                    tokenizer, writer_by_id[question_id],
                    args.writer_maximum_sequence_tokens,
                )
                input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)
                label_tensor = torch.tensor([writer_labels], device=device, dtype=torch.long)
                logits = model(
                    input_ids=input_tensor,
                    attention_mask=torch.ones_like(input_tensor),
                    use_cache=False,
                ).logits
                shifted_labels = label_tensor[:, 1:]
                supervised = shifted_labels.ne(-100)
                own_loss = F.cross_entropy(
                    logits[:, :-1, :][supervised].float(), shifted_labels[supervised]
                )
                shuffled_loss = own_loss.detach()
                target_tokens = int(writer_audit["target_tokens"])
                loss = own_loss
            else:
                states, mask = cached_state(
                    args.cache_dir, example_by_id[question_id], device
                )
                candidates = negative_ids[question_id]
                negative_id = candidates[
                    (global_update * 97 + rank * 193) % len(candidates)
                ]
                if bridge_ddp is None:
                    with torch.no_grad():
                        soft = bridge(states, mask)
                else:
                    soft = bridge_ddp(states, mask)
                    with torch.no_grad():
                        initial_soft = bridge_reference(states, mask)
                    reference_energy = initial_soft.float().square().mean().clamp_min(1e-8)
                    anchor_loss = F.mse_loss(
                        soft.float(), initial_soft.float()
                    ) / reference_energy
                    relative_soft_drift = anchor_loss.detach().sqrt()
                if args.objective == "qa":
                    inputs, attention, labels, target_tokens = answer_reasoning_inputs(
                        policy, tokenizer, soft, question, target,
                        args.maximum_target_tokens,
                    )
                else:
                    inputs, attention, labels, target_tokens = reconstruction_inputs(
                        policy, tokenizer, soft, question, target,
                        args.maximum_target_tokens,
                        include_question=args.objective == "reconstruct-question",
                    )
                own_loss = model(
                    inputs_embeds=inputs, attention_mask=attention, labels=labels,
                    use_cache=False,
                ).loss
                shuffled_loss = own_loss.detach()
                if args.ranking_weight:
                    negative_states, negative_mask = cached_state(
                        args.cache_dir, example_by_id[negative_id], device
                    )
                    if bridge_ddp is None:
                        with torch.no_grad():
                            negative_soft = bridge(negative_states, negative_mask)
                    else:
                        negative_soft = bridge_ddp(negative_states, negative_mask)
                    if args.objective == "qa":
                        negative_inputs, negative_attention, negative_labels, _ = (
                            answer_reasoning_inputs(
                                policy, tokenizer, negative_soft, question, target,
                                args.maximum_target_tokens,
                            )
                        )
                    else:
                        negative_inputs, negative_attention, negative_labels, _ = (
                            reconstruction_inputs(
                                policy, tokenizer, negative_soft, question, target,
                                args.maximum_target_tokens,
                                include_question=args.objective == "reconstruct-question",
                            )
                        )
                    shuffled_loss = model(
                        inputs_embeds=negative_inputs,
                        attention_mask=negative_attention,
                        labels=negative_labels,
                        use_cache=False,
                    ).loss
                    ranking_loss = torch.relu(
                        args.ranking_margin + own_loss - shuffled_loss
                    )
                    ranking_enabled = float(args.objective != "qa" or audit_correct)
                loss = (
                    own_loss
                    + args.ranking_weight * ranking_enabled * ranking_loss
                    + args.soft_output_anchor_weight * anchor_loss
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite gradient at epoch={epoch_number} update={update + 1}"
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            averages = torch.stack(
                (
                    loss.detach(),
                    own_loss.detach(),
                    shuffled_loss.detach(),
                    ranking_loss.detach(),
                    anchor_loss.detach(),
                    relative_soft_drift.detach(),
                )
            )
            dist.all_reduce(averages, op=dist.ReduceOp.SUM)
            averages /= world_size
            if rank == 0:
                metric = {
                    "epoch": epoch_number,
                    "task": task,
                    "update_in_epoch": update + 1,
                    "global_update": global_update,
                    "global_examples_seen": global_update * world_size,
                    "rank0_question_id": question_id,
                    "mean_loss": float(averages[0]),
                    "mean_own_loss": float(averages[1]),
                    "mean_shuffled_loss": float(averages[2]),
                    "mean_ranking_loss": float(averages[3]),
                    "mean_bridge_anchor_loss": float(averages[4]),
                    "mean_relative_soft_drift": float(averages[5]),
                    "rank0_ranking_enabled": bool(ranking_enabled),
                    "rank0_negative_question_id": negative_id,
                    "gradient_norm": float(grad_norm),
                    "rank0_target_tokens": target_tokens,
                    "peak_gpu_memory_bytes_rank0": torch.cuda.max_memory_allocated(device),
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metric) + "\n")
                print(json.dumps(metric), flush=True)

        dist.barrier()
        if rank == 0:
            adapter = args.output / "checkpoints" / f"epoch-{epoch_number}" / "adapter"
            adapter.mkdir(parents=True)
            policy.save_pretrained(adapter, safe_serialization=True)
            bridge_path = (
                args.output / "checkpoints" / f"epoch-{epoch_number}" / "bridge.pt"
            )
            saved_config = dict(bridge_config)
            saved_config.update(
                {
                    "joint_reader_source": str(args.compressor_checkpoint),
                    "joint_reader_epoch": epoch_number,
                    "trainable_scope": args.compressor_trainable_scope,
                    "anchor_weight": args.soft_output_anchor_weight,
                }
            )
            torch.save(
                {"procedure": PROCEDURE,
                    "bridge": {
                        name: value.detach().cpu()
                        for name, value in bridge.state_dict().items()
                    },
                    "config": saved_config,
                    "global_step": global_update,
                },
                bridge_path,
            )
        dist.barrier()

    if rank == 0:
        result = {"procedure": PROCEDURE,
            "status": "completed",
            "epochs": args.epochs,
            "updates_this_run": updates_per_epoch * args.epochs,
            "final_global_update": global_update,
            "adapter": str(
                args.output
                / "checkpoints"
                / f"epoch-{args.epoch_offset + args.epochs}"
                / "adapter"
            ),
            "bridge": str(
                args.output
                / "checkpoints"
                / f"epoch-{args.epoch_offset + args.epochs}"
                / "bridge.pt"
            ),
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
