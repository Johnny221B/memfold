#!/usr/bin/env python3
"""Recoverable full-coverage MemGen Weaver SFT for chunked PersonaMem records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

from train_memgen_strict_weaver_sft import build_model


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--stop-after-steps", type=int,
        help="Stop this invocation after N additional steps without changing the schedule.",
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_records(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scheduler_factor(step: int, total: int, warmup: int) -> float:
    if warmup and step < warmup:
        return float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, state: dict) -> None:
    if path.exists():
        raise FileExistsError(f"checkpoint path already exists: {path}")
    path.mkdir(parents=True)
    torch.save(trainable_state(model), path / "trainable_state.pt")
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save(scheduler.state_dict(), path / "scheduler.pt")
    torch.save({
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }, path / "rng_state.pt")
    (path / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n")


def load_checkpoint(path: Path, model, optimizer, scheduler) -> dict:
    state_dict = torch.load(path / "trainable_state.pt", map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected checkpoint keys: {incompatible.unexpected_keys}")
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True))
    scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True))
    rng = torch.load(path / "rng_state.pt", map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state_all(rng["cuda"])
    return json.loads((path / "trainer_state.json").read_text())


def forward_loss(model, row: dict, device: torch.device, backward: bool) -> tuple[torch.Tensor, int]:
    tokenizer = model.tokenizer
    latent_blocks = []
    chunk_inputs = []
    chunk_rng_states = []
    for text in row["memory_chunks"]:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        ids = encoded.input_ids.to(device)
        mask = encoded.attention_mask.to(device)
        if backward:
            chunk_inputs.append((ids, mask))
            chunk_rng_states.append(torch.cuda.get_rng_state(device))
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            base_embeds = model.reasoner.get_input_embeddings()(ids)
            positions = model._generate_position_ids(mask)
            weaver_inputs = model.reasoner_to_weaver(base_embeds)
            hidden, _, _ = model.weaver.augment_prompt(weaver_inputs, mask, positions)
            latent = model.weaver_to_reasoner(hidden).detach()
            if backward:
                latent.requires_grad_(True)
            latent_blocks.append(latent)
    memory_latents = torch.cat(latent_blocks, dim=1)
    task = tokenizer.apply_chat_template(
        row["task_messages"], tokenize=True, return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids = torch.tensor([task["input_ids"]], device=device)
    labels = torch.tensor([[
        token if assistant else -100
        for token, assistant in zip(task["input_ids"], task["assistant_masks"])
    ]], device=device)
    supervised_positions = (labels[0] != -100).nonzero()
    if not len(supervised_positions):
        raise ValueError(f"no assistant labels for {row['trajectory_id']}")
    first = int(supervised_positions[0])
    task_embeds = model.reasoner.get_input_embeddings()(ids)
    combined = torch.cat([task_embeds[:, :first], memory_latents, task_embeds[:, first:]], dim=1)
    augmented_labels = torch.cat([
        labels[:, :first],
        torch.full((1, memory_latents.size(1)), -100, device=device, dtype=labels.dtype),
        labels[:, first:],
    ], dim=1)
    attention = torch.ones(combined.shape[:2], device=device, dtype=torch.long)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model.reasoner.model(
            inputs_embeds=combined, attention_mask=attention,
            position_ids=model._generate_position_ids(attention),
        )
        shifted_hidden = outputs.last_hidden_state[:, :-1].reshape(
            -1, outputs.last_hidden_state.size(-1)
        )
        shifted_labels = augmented_labels[:, 1:].reshape(-1)
        supervised = shifted_labels != -100
        loss = LigerFusedLinearCrossEntropyLoss()(model.reasoner.lm_head.weight,
                                                   shifted_hidden[supervised],
                                                   shifted_labels[supervised])
    if backward:
        loss.backward()
        latent_gradients = [block.grad.detach().clone() for block in latent_blocks]
        del outputs, combined, memory_latents
        for (chunk_ids, chunk_mask), rng_state, latent_gradient in zip(
            chunk_inputs, chunk_rng_states, latent_gradients
        ):
            torch.cuda.set_rng_state(rng_state, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                base_embeds = model.reasoner.get_input_embeddings()(chunk_ids)
                positions = model._generate_position_ids(chunk_mask)
                weaver_inputs = model.reasoner_to_weaver(base_embeds)
                hidden, _, _ = model.weaver.augment_prompt(weaver_inputs, chunk_mask, positions)
                recomputed_latent = model.weaver_to_reasoner(hidden)
            recomputed_latent.backward(latent_gradient)
    return loss.detach(), int(len(row["memory_chunks"]))


@torch.no_grad()
def validate(model, rows: list[dict], device: torch.device) -> float:
    model.eval()
    losses = [float(forward_loss(model, row, device, False)[0]) for row in rows]
    model.train()
    return sum(losses) / len(losses)


def main() -> int:
    args = arguments()
    if args.output_dir.exists() and not args.resume_from_checkpoint:
        raise FileExistsError("output-dir must be new unless resuming")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train_rows = read_records(args.train_data, args.train_limit)
    valid_rows = read_records(args.valid_data)
    random.Random(args.seed + 10_000).shuffle(valid_rows)
    valid_rows = valid_rows[: args.valid_limit]
    if not train_rows or not valid_rows:
        raise ValueError("train and validation records must be non-empty")
    total_steps = args.epochs * len(train_rows)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = build_model(args.model, True, False)
    model.fix_component("trigger")
    for parameter in model.reasoner.parameters():
        parameter.requires_grad_(False)
    model.reasoner.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.weaver.model.base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.reasoner.config.use_cache = False
    model.weaver.model.base_model.config.use_cache = False
    model.to(device).train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: scheduler_factor(step, total_steps, warmup_steps)
    )
    state = {
        "schema_version": "persona_memgen_chunked_train_v1",
        "global_step": 0, "next_epoch": 0, "next_position": 0,
        "total_steps": total_steps, "train_records": len(train_rows),
        "valid_records": len(valid_rows), "seed": args.seed,
        "train_data_sha256": digest(args.train_data),
        "valid_data_sha256": digest(args.valid_data),
    }
    if args.resume_from_checkpoint:
        state = load_checkpoint(args.resume_from_checkpoint, model, optimizer, scheduler)
        if state["total_steps"] != total_steps or state["train_records"] != len(train_rows):
            raise ValueError("resume checkpoint does not match current training schedule")

    log_path = args.output_dir / "train_log.jsonl"
    log_mode = "a" if args.resume_from_checkpoint else "x"
    started = time.time()
    invocation_start_step = state["global_step"]
    stop = False
    with log_path.open(log_mode, encoding="utf-8") as log:
        for epoch in range(state["next_epoch"], args.epochs):
            order = list(range(len(train_rows)))
            random.Random(args.seed + epoch).shuffle(order)
            start_position = state["next_position"] if epoch == state["next_epoch"] else 0
            for position in range(start_position, len(order)):
                if state["global_step"] >= total_steps or (
                    args.stop_after_steps is not None
                    and state["global_step"] - invocation_start_step >= args.stop_after_steps
                ):
                    stop = True
                    break
                step_started = time.time()
                optimizer.zero_grad(set_to_none=True)
                loss, chunks = forward_loss(model, train_rows[order[position]], device, True)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                    raise RuntimeError(f"non-finite loss or gradient at step {state['global_step'] + 1}")
                optimizer.step()
                scheduler.step()
                state["global_step"] += 1
                state["next_epoch"] = epoch
                state["next_position"] = position + 1
                if state["next_position"] == len(order):
                    state["next_epoch"] = epoch + 1
                    state["next_position"] = 0
                event = {
                    "step": state["global_step"], "epoch": epoch,
                    "position": position, "trajectory_id": train_rows[order[position]]["trajectory_id"],
                    "chunks": chunks, "loss": float(loss), "grad_norm": float(grad_norm),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "seconds": time.time() - step_started,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                }
                log.write(json.dumps(event) + "\n")
                log.flush()
                print(json.dumps(event), flush=True)
                if state["global_step"] % args.save_steps == 0:
                    save_checkpoint(args.output_dir / f"checkpoint-{state['global_step']}",
                                    model, optimizer, scheduler, state)
            if stop:
                break
            if state["next_epoch"] == epoch + 1 and state["global_step"] < total_steps:
                validation_loss = validate(model, valid_rows, device)
                validation = {"step": state["global_step"], "epoch": epoch,
                              "validation_loss": validation_loss}
                log.write(json.dumps(validation) + "\n")
                log.flush()
                print(json.dumps(validation), flush=True)

    final_checkpoint = args.output_dir / f"checkpoint-{state['global_step']}"
    if not final_checkpoint.exists():
        save_checkpoint(final_checkpoint, model, optimizer, scheduler, state)
    if state["global_step"] == args.epochs * len(train_rows):
        model.save_pretrained(str(args.output_dir / "final"))
    summary = dict(state)
    summary.update({
        "completed": state["global_step"] == args.epochs * len(train_rows),
        "seconds_this_invocation": time.time() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
    })
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
