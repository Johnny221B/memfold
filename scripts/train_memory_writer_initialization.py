#!/usr/bin/env python3
"""Memory-writer initialization with LoRA and PersonaMem memory supervision."""

from __future__ import annotations

PROCEDURE = "memory_writer_initialization"

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from memory_opd.rq2_sft.training import (
    encode_sft_record,
    encode_weighted_multiturn_record,
    ordered_record_indices,
)


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(model, optimizer, scheduler, output: Path, epoch: int, micro_step: int) -> None:
    checkpoint = output / f"epoch-{epoch}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(checkpoint, safe_serialization=True)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "micro_step": micro_step},
        checkpoint / "trainer_state.pt",
    )


def save_eval_adapter(model, output: Path, optimizer_step: int) -> None:
    checkpoint = output / f"step-{optimizer_step:04d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(checkpoint, safe_serialization=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-micro-steps", type=int, default=0)
    parser.add_argument("--expected-records", type=int, default=179)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--reset-training-state", action="store_true")
    parser.add_argument("--initial-epoch", type=int, default=0)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--save-at-updates",
        type=int,
        nargs="*",
        default=[],
        metavar="UPDATE",
        help="Save lightweight evaluation adapters only at these optimizer updates.",
    )
    checkpoint_group.add_argument(
        "--save-every-updates",
        type=int,
        default=0,
        help="Legacy dense-saving option; prefer --save-at-updates.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eos-repetitions", type=int, default=1)
    parser.add_argument("--curriculum-short-first", action="store_true")
    parser.add_argument(
        "--require-no-truncation", action="store_true",
        help="fail immediately if any encoded training example is truncated",
    )
    args = parser.parse_args()
    args.procedure = "memory_writer_initialization"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    if any(update <= 0 for update in args.save_at_updates):
        parser.error("--save-at-updates values must be positive")
    if args.eos_repetitions <= 0:
        parser.error("--eos-repetitions must be positive")
    if args.reset_training_state and not args.resume_from:
        parser.error("--reset-training-state requires --resume-from")
    if args.initial_epoch < 0 or args.initial_epoch >= args.epochs:
        parser.error("require 0 <= initial_epoch < epochs")
    if args.initial_epoch and not args.reset_training_state:
        parser.error("--initial-epoch requires --reset-training-state")
    selected_updates = set(args.save_at_updates)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != args.expected_records or any(row["split"] != "train" for row in records):
        raise ValueError(
            f"PersonaMem SFT requires exactly {args.expected_records} train records"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.eos_repetitions > 1 and tokenizer.eos_token_id is None:
        raise ValueError("EOS repetition requires tokenizer.eos_token_id")
    # Selected evidence records already carry tokenizer-measured target lengths.
    # Reuse them to avoid tokenizing every 32K history once before training.
    target_lengths = [
        int(record.get("metadata", {}).get("memory_tokens", 0))
        for record in records
    ]
    if any(length <= 0 for length in target_lengths):
        target_lengths = [
            encode_sft_record(tokenizer, record, args.max_length)[2]["target_tokens"]
            for record in records
        ]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.resume_from:
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=TARGET_MODULES,
            ),
        )
    model.to(device).train()
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    samples_per_rank = math.ceil(len(records) / world_size)
    updates_per_epoch = math.ceil(samples_per_rank / args.grad_accum)
    start_epoch = args.initial_epoch if args.reset_training_state else 0
    total_updates = updates_per_epoch * (args.epochs - start_epoch)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, math.ceil(total_updates * args.warmup_ratio)),
        num_training_steps=total_updates,
    )
    micro_step = 0
    optimizer_step = 0
    if args.resume_from and not args.reset_training_state:
        state = torch.load(args.resume_from / "trainer_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        micro_step = int(state["micro_step"])
        optimizer_step = int(scheduler.last_epoch)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "run_config.json").write_text(
            json.dumps({**vars(args), "model": str(args.model), "data": str(args.data), "output": str(args.output), "trainable_parameters": trainable, "total_updates": total_updates, "world_size": world_size}, indent=2, default=str) + "\n"
        )
    if distributed:
        dist.barrier()
    optimizer.zero_grad(set_to_none=True)
    for epoch_index in range(start_epoch, args.epochs):
        order = ordered_record_indices(
            target_lengths,
            seed=args.seed,
            epoch_index=epoch_index,
            short_first=args.curriculum_short_first,
        )
        padded = order + order[: samples_per_rank * world_size - len(order)]
        local_order = padded[rank::world_size]
        for position, index in enumerate(local_order, start=1):
            weighted = bool(records[index].get("metadata", {}).get("assistant_loss_weights"))
            if weighted:
                input_ids, labels, loss_weights, audit = encode_weighted_multiturn_record(
                    tokenizer, records[index], args.max_length
                )
            else:
                input_ids, labels, audit = encode_sft_record(tokenizer, records[index], args.max_length)
                loss_weights = [1.0 if label != -100 else 0.0 for label in labels]
            if args.require_no_truncation and int(audit.get("left_truncated", 0)) != 0:
                raise RuntimeError(
                    f"full-context training forbids truncation: id={records[index]['id']} "
                    f"left_truncated={audit['left_truncated']} max_length={args.max_length}"
                )
            if args.eos_repetitions > 1:
                extra_eos = [tokenizer.eos_token_id] * (args.eos_repetitions - 1)
                input_ids = (input_ids + extra_eos)[-args.max_length :]
                labels = (labels + extra_eos)[-args.max_length :]
                loss_weights = (loss_weights + [1.0] * len(extra_eos))[-args.max_length :]
                audit["tokens"] = len(input_ids)
                audit["target_tokens"] += len(extra_eos)
                audit["extra_eos_tokens"] = len(extra_eos)
            input_tensor = torch.tensor([input_ids], device=device)
            label_tensor = torch.tensor([labels], device=device)
            weight_tensor = torch.tensor([loss_weights], device=device, dtype=torch.float32)
            group_start = ((position - 1) // args.grad_accum) * args.grad_accum + 1
            group_size = min(args.grad_accum, len(local_order) - group_start + 1)
            update = position % args.grad_accum == 0 or position == len(local_order)
            sync_context = nullcontext() if update or not distributed else model.no_sync()
            with sync_context:
                policy = unwrap_model(model)
                causal_lm = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
                hidden = causal_lm.model(
                    input_ids=input_tensor,
                    attention_mask=torch.ones_like(input_tensor),
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
                shifted_labels = label_tensor[:, 1:]
                supervised = shifted_labels.ne(-100)
                # Only supervised memory-target positions need vocabulary logits.
                selected_hidden = hidden[:, :-1, :][supervised]
                selected_logits = causal_lm.get_output_embeddings()(selected_hidden)
                selected_labels = shifted_labels[supervised]
                selected_weights = weight_tensor[:, 1:][supervised]
                token_loss = F.cross_entropy(
                    selected_logits.float(), selected_labels, reduction="none",
                )
                loss = (token_loss * selected_weights).sum() / selected_weights.sum().clamp_min(1.0)
                (loss / group_size).backward()
            micro_step += 1
            if update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if optimizer_step in selected_updates or (
                    args.save_every_updates > 0
                    and optimizer_step % args.save_every_updates == 0
                ):
                    if rank == 0:
                        save_eval_adapter(model, args.output, optimizer_step)
            if rank == 0:
                print(json.dumps({"epoch": epoch_index + 1, "micro_step": micro_step, "optimizer_step": optimizer_step, "id": records[index]["id"], "loss": float(loss.detach()), "learning_rate": scheduler.get_last_lr()[0], **audit}), flush=True)
            if args.max_micro_steps and micro_step >= args.max_micro_steps:
                if rank == 0:
                    save_checkpoint(model, optimizer, scheduler, args.output, epoch_index + 1, micro_step)
                if distributed:
                    dist.barrier()
                    dist.destroy_process_group()
                return
        if distributed:
            dist.barrier()
        if rank == 0:
            save_checkpoint(model, optimizer, scheduler, args.output, epoch_index + 1, micro_step)
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
