#!/usr/bin/env python3
"""Cache context features and train a minimal soft-memory reconstruction SFT."""

from __future__ import annotations

PROCEDURE = "compressor_reconstruction"

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
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
    same_context_example,
    reconstruction_prompt,
    soft_memory_vector,
    split_examples_by_context,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cache", "train", "evaluate", "dry-run"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument(
        "--input-mode", choices=("context", "text-memory"), default="context"
    )
    parser.add_argument("--contexts", type=Path)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--token-count", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--context-residual", action="store_true")
    parser.add_argument("--encoder-chunk-tokens", type=int, default=2048)
    parser.add_argument("--pool-tokens", type=int, default=32)
    parser.add_argument("--encoder-batch-size", type=int, default=2)
    parser.add_argument("--cache-shards", type=int, default=1)
    parser.add_argument("--cache-shard-index", type=int, default=0)
    parser.add_argument("--validation-contexts", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument("--separation-weight", type=float, default=0.0)
    parser.add_argument("--alignment-weight", type=float, default=0.0)
    parser.add_argument("--maximum-cross-context-cosine", type=float, default=0.8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-target-tokens", type=int, default=2048)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--limit-states", type=int)
    parser.add_argument("--eval-max-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.procedure = "compressor_reconstruction"
    return args


def load_data(args: argparse.Namespace):
    if args.input_mode == "context":
        if args.contexts is None:
            raise ValueError("--contexts is required when --input-mode=context")
        examples, state_text = load_reconstruction_examples(
            args.questions, args.contexts, args.memories
        )
    else:
        examples, state_text = load_text_memory_examples(
            args.questions, args.memories
        )
    train, validation = split_examples_by_context(
        examples, validation_contexts=args.validation_contexts, seed=args.seed
    )
    if args.limit_samples is not None:
        train = train[: args.limit_samples]
        validation = validation[: args.limit_samples]
    return examples, train, validation, state_text


def cache_path(cache_dir: Path, state_id: str) -> Path:
    return cache_dir / "states" / f"{state_id}.pt"


def batches(values: list[list[int]], size: int) -> Iterable[list[list[int]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def pool_hidden(hidden: torch.Tensor, valid: int, pool_tokens: int) -> list[torch.Tensor]:
    return [hidden[start : min(valid, start + pool_tokens)].mean(0).cpu()
            for start in range(0, valid, pool_tokens)]


def prepare_cache(args: argparse.Namespace, state_text: dict[str, str]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for context feature caching")
    if min(args.encoder_chunk_tokens, args.pool_tokens, args.encoder_batch_size) <= 0:
        raise ValueError("cache chunk/pool/batch sizes must be positive")
    if args.cache_shards <= 0 or not 0 <= args.cache_shard_index < args.cache_shards:
        raise ValueError("cache shard index must be in [0, cache_shards)")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.requires_grad_(False).eval()
    base = getattr(model, "model", None)
    if base is None:
        raise TypeError(f"{type(model).__name__} does not expose its decoder as .model")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "states").mkdir(parents=True, exist_ok=True)
    selected = sorted(state_text.items())
    if args.limit_states is not None:
        selected = selected[: args.limit_states]
    selected = selected[args.cache_shard_index :: args.cache_shards]
    records: list[dict[str, Any]] = []
    for number, (state_id, text) in enumerate(selected, 1):
        output = cache_path(args.cache_dir, state_id)
        if output.exists():
            payload = torch.load(output, map_location="cpu", weights_only=True)
            records.append({"state_id": state_id, "vectors": int(payload["states"].shape[0]),
                            "tokens": int(payload["token_count"]), "resumed": True})
            continue
        token_ids = tokenizer(text, add_special_tokens=True).input_ids
        chunks = [token_ids[start : start + args.encoder_chunk_tokens]
                  for start in range(0, len(token_ids), args.encoder_chunk_tokens)]
        pooled: list[torch.Tensor] = []
        with torch.inference_mode():
            for group in batches(chunks, args.encoder_batch_size):
                width = max(len(item) for item in group)
                ids = torch.full((len(group), width), tokenizer.pad_token_id,
                                 device=device, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for row, item in enumerate(group):
                    ids[row, : len(item)] = torch.tensor(item, device=device)
                    mask[row, : len(item)] = 1
                hidden = base(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
                for row, item in enumerate(group):
                    pooled.extend(pool_hidden(hidden[row], len(item), args.pool_tokens))
        states = torch.stack(pooled).to(torch.float16)
        temporary = output.with_suffix(".tmp")
        torch.save({"states": states, "token_count": len(token_ids)}, temporary)
        temporary.replace(output)
        records.append({"state_id": state_id, "vectors": int(states.shape[0]),
                        "tokens": len(token_ids), "resumed": False})
        print(json.dumps({"cached": number, "selected": len(selected), **records[-1]}), flush=True)
    manifest = {
        "schema_version": "soft_reconstruction_input_cache_v2",
        "input_mode": args.input_mode,
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256(args.model / "config.json"),
        "questions_sha256": sha256(args.questions),
        "contexts_sha256": sha256(args.contexts) if args.contexts else None,
        "memories_sha256": sha256(args.memories),
        "encoder_chunk_tokens": args.encoder_chunk_tokens,
        "pool_tokens": args.pool_tokens,
        "states_expected": len(state_text),
        "states_selected": len(selected),
        "states_cached": len(records),
        "cache_shards": args.cache_shards,
        "cache_shard_index": args.cache_shard_index,
        "records": records,
    }
    manifest_name = (
        "manifest.json"
        if args.cache_shards == 1
        else f"manifest-shard-{args.cache_shard_index:05d}-of-{args.cache_shards:05d}.json"
    )
    (args.cache_dir / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")


def load_cached(state_id: str, cache_dir: Path, device: torch.device, dtype: torch.dtype):
    path = cache_path(cache_dir, state_id)
    if not path.exists():
        raise FileNotFoundError(f"missing cached state: {path}")
    states = torch.load(path, map_location="cpu", weights_only=True)["states"]
    return states.to(device=device, dtype=dtype).unsqueeze(0), torch.ones(
        (1, states.shape[0]), device=device, dtype=torch.long
    )


def build_bridge(args: argparse.Namespace, context_dim: int, lm_dim: int, dtype: torch.dtype, device):
    bridge = ContextToSoftTokens(
        ContextResampler(
            context_dim,
            latent_dim=args.latent_dim,
            token_count=args.token_count,
            layers=args.layers,
            heads=args.heads,
            context_residual=args.context_residual,
        ),
        SoftTokenProjector(args.latent_dim, lm_dim),
    ).to(device=device, dtype=dtype)
    return bridge


def reconstruction_inputs(model, tokenizer, soft: torch.Tensor, example: ReconstructionExample,
                          max_target_tokens: int):
    prompt_ids = tokenizer(
        reconstruction_prompt(example.question), add_special_tokens=True,
        return_tensors="pt",
    ).input_ids.to(soft.device)
    target_ids = tokenizer(
        example.memory_text + (tokenizer.eos_token or ""),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(soft.device)
    if target_ids.shape[1] > max_target_tokens:
        raise ValueError(
            f"target for {example.question_id} has {target_ids.shape[1]} tokens; "
            f"limit is {max_target_tokens}"
        )
    embedding = model.get_input_embeddings()
    prompt = embedding(prompt_ids)
    target = embedding(target_ids)
    inputs = torch.cat((soft, prompt, target), dim=1)
    labels = torch.full(inputs.shape[:2], -100, device=soft.device, dtype=torch.long)
    labels[:, soft.shape[1] + prompt.shape[1] :] = target_ids
    attention = torch.ones(inputs.shape[:2], device=soft.device, dtype=torch.long)
    return inputs, attention, labels, int(target_ids.numel())


def load_model(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False
    model.requires_grad_(False).eval()
    return model, tokenizer, device, dtype


def state_dimension(cache_dir: Path, examples: list[ReconstructionExample]) -> int:
    for example in examples:
        path = cache_path(cache_dir, example.state_id)
        if path.exists():
            return int(torch.load(path, map_location="cpu", weights_only=True)["states"].shape[-1])
    raise FileNotFoundError("none of the requested examples has a cached context state")


def checkpoint_payload(bridge, args, step, losses):
    return {"procedure": PROCEDURE,
        "bridge": bridge.state_dict(),
        "config": {
            "context_dim": bridge.resampler.context_projection.in_features,
            "lm_dim": bridge.projector.projection.out_features,
            "latent_dim": args.latent_dim,
            "token_count": args.token_count,
            "layers": args.layers,
            "heads": args.heads,
            "context_residual": args.context_residual,
            "ranking_weight": args.ranking_weight,
            "ranking_margin": args.ranking_margin,
            "separation_weight": args.separation_weight,
            "maximum_cross_context_cosine": args.maximum_cross_context_cosine,
            "input_mode": args.input_mode,
        },
        "global_step": step,
        "losses": losses,
    }


def train(args: argparse.Namespace, train_examples: list[ReconstructionExample]) -> None:
    if min(args.ranking_weight, args.ranking_margin, args.separation_weight) < 0:
        raise ValueError("ranking/separation weights and margin must be non-negative")
    if not -1.0 <= args.maximum_cross_context_cosine <= 1.0:
        raise ValueError("maximum cross-context cosine must be in [-1, 1]")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer, device, dtype = load_model(args)
    context_dim = state_dimension(args.cache_dir, train_examples)
    bridge = build_bridge(args, context_dim, int(model.config.hidden_size), dtype, device).train()
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    total = len(train_examples) * args.epochs
    if args.max_steps is not None:
        total = min(total, args.max_steps)
    global_step = 0
    optimizer_step = 0
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        order = list(train_examples)
        random.Random(args.seed + epoch).shuffle(order)
        for example_index, example in enumerate(order):
            if global_step >= total:
                break
            states, mask = load_cached(example.state_id, args.cache_dir, device, dtype)
            soft = bridge(states, mask)
            inputs, attention, labels, target_tokens = reconstruction_inputs(
                model, tokenizer, soft, example, args.max_target_tokens
            )
            output = model(inputs_embeds=inputs, attention_mask=attention, labels=labels)
            reconstruction_loss = output.loss
            shuffled_loss = None
            ranking_loss = reconstruction_loss.new_zeros(())
            separation_loss = reconstruction_loss.new_zeros(())
            alignment_loss = reconstruction_loss.new_zeros(())
            cross_context_cosine = None
            negative = None
            if args.ranking_weight > 0 or args.separation_weight > 0 or args.alignment_weight > 0:
                negative = different_context_example(order, example_index)
                negative_states, negative_mask = load_cached(
                    negative.state_id, args.cache_dir, device, dtype
                )
                negative_soft = bridge(negative_states, negative_mask)
                separation_loss, cosine = cross_context_separation_loss(
                    soft,
                    negative_soft,
                    maximum_cosine=args.maximum_cross_context_cosine,
                )
                cross_context_cosine = float(cosine.detach())
            if args.alignment_weight > 0:
                positive = same_context_example(order, example_index)
                positive_states, positive_mask = load_cached(positive.state_id, args.cache_dir, device, dtype)
                positive_soft = bridge(positive_states, positive_mask)
                alignment_loss = 1.0 - F.cosine_similarity(
                    soft_memory_vector(soft).unsqueeze(0), soft_memory_vector(positive_soft).unsqueeze(0)
                ).mean()
            if args.ranking_weight > 0:
                negative_inputs, negative_attention, negative_labels, _ = reconstruction_inputs(
                    model, tokenizer, negative_soft, example, args.max_target_tokens
                )
                shuffled_output = model(
                    inputs_embeds=negative_inputs,
                    attention_mask=negative_attention,
                    labels=negative_labels,
                )
                shuffled_loss = shuffled_output.loss
                ranking_loss = torch.relu(
                    args.ranking_margin + reconstruction_loss - shuffled_loss
                )
            objective = (
                reconstruction_loss
                + args.ranking_weight * ranking_loss
                + args.separation_weight * separation_loss
                + args.alignment_weight * alignment_loss
            )
            loss = objective / args.gradient_accumulation
            loss.backward()
            global_step += 1
            losses.append(float(reconstruction_loss.detach()))
            should_step = global_step % args.gradient_accumulation == 0 or global_step == total
            grad_norm = None
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            metric = {"procedure": PROCEDURE,
                "global_step": global_step,
                "optimizer_step": optimizer_step,
                "epoch": epoch,
                "question_id": example.question_id,
                "state_id": example.state_id,
                "reconstruction_loss": float(reconstruction_loss.detach()),
                "objective_loss": float(objective.detach()),
                "shuffled_loss": (
                    None if shuffled_loss is None else float(shuffled_loss.detach())
                ),
                "ranking_loss": float(ranking_loss.detach()),
                "separation_loss": float(separation_loss.detach()),
                "alignment_loss": float(alignment_loss.detach()),
                "cross_context_cosine": cross_context_cosine,
                "negative_state_id": None if negative is None else negative.state_id,
                "target_tokens": target_tokens,
                "gradient_norm": None if grad_norm is None else float(grad_norm),
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
            }
            with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric) + "\n")
            print(json.dumps(metric), flush=True)
        if global_step >= total:
            break
    checkpoint = args.output / f"checkpoint-{global_step:06d}.pt"
    torch.save(checkpoint_payload(bridge, args, global_step, losses), checkpoint)
    result = {"procedure": PROCEDURE,
        "status": "completed",
        "objective": "target_memory_ce_with_optional_cross_context_losses",
        "answers_used": False,
        "base_model_trainable": False,
        "global_step": global_step,
        "optimizer_steps": optimizer_step,
        "mean_loss": sum(losses) / len(losses),
        "checkpoint": str(checkpoint),
        "trainable_parameters": sum(p.numel() for p in bridge.parameters()),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "ranking_weight": args.ranking_weight,
        "ranking_margin": args.ranking_margin,
        "separation_weight": args.separation_weight,
        "maximum_cross_context_cosine": args.maximum_cross_context_cosine,
        "input_mode": args.input_mode,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


@torch.no_grad()
def nll(model, tokenizer, bridge, states, mask, example, max_target_tokens, condition):
    soft = bridge(states, mask)
    if condition == "null":
        soft = torch.zeros_like(soft)
    inputs, attention, labels, _ = reconstruction_inputs(
        model, tokenizer, soft, example, max_target_tokens
    )
    return float(model(inputs_embeds=inputs, attention_mask=attention, labels=labels).loss)


def evaluate(args: argparse.Namespace, validation: list[ReconstructionExample]) -> None:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for evaluate")
    model, tokenizer, device, dtype = load_model(args)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    for name in ("latent_dim", "token_count", "layers", "heads"):
        setattr(args, name, int(config[name]))
    args.context_residual = bool(config.get("context_residual", False))
    bridge = build_bridge(args, int(config["context_dim"]), int(config["lm_dim"]), dtype, device)
    bridge.load_state_dict(saved["bridge"])
    bridge.eval()
    selected = list(validation[: args.eval_max_samples])
    if len(selected) < 1:
        raise ValueError("evaluation requires at least one validation sample")
    # Validation examples can be context-grouped, so the initial window may
    # contain only one context even when the full split has many contexts.
    # Add one deterministic example from another context for shuffled controls.
    if len({example.context_id for example in selected}) < 2:
        anchor_context = selected[0].context_id
        candidate = next(
            (example for example in validation if example.context_id != anchor_context),
            None,
        )
        if candidate is None:
            raise ValueError("evaluation requires at least two distinct validation contexts")
        selected.append(candidate)
    rows = []
    for index, example in enumerate(selected):
        own_states, own_mask = load_cached(example.state_id, args.cache_dir, device, dtype)
        shuffled = different_context_example(selected, index)
        shuffled_states, shuffled_mask = load_cached(shuffled.state_id, args.cache_dir, device, dtype)
        row = {
            "question_id": example.question_id,
            "state_id": example.state_id,
            "shuffled_state_id": shuffled.state_id,
            "own_nll": nll(model, tokenizer, bridge, own_states, own_mask, example,
                           args.max_target_tokens, "own"),
            "shuffled_nll": nll(model, tokenizer, bridge, shuffled_states, shuffled_mask, example,
                                args.max_target_tokens, "shuffled"),
            "null_nll": nll(model, tokenizer, bridge, own_states, own_mask, example,
                            args.max_target_tokens, "null"),
            "cross_context_cosine": float(
                (
                    soft_memory_vector(bridge(own_states, own_mask))
                    * soft_memory_vector(bridge(shuffled_states, shuffled_mask))
                ).sum().detach()
            ),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {
        "samples": len(rows),
        "own_nll": sum(row["own_nll"] for row in rows) / len(rows),
        "shuffled_nll": sum(row["shuffled_nll"] for row in rows) / len(rows),
        "null_nll": sum(row["null_nll"] for row in rows) / len(rows),
        "own_beats_shuffled_rate": sum(row["own_nll"] < row["shuffled_nll"] for row in rows) / len(rows),
        "own_beats_null_rate": sum(row["own_nll"] < row["null_nll"] for row in rows) / len(rows),
        "mean_cross_context_cosine": sum(
            row["cross_context_cosine"] for row in rows
        ) / len(rows),
        "answers_used": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reconstruction_controls.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (args.output / "reconstruction_controls_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = arguments()
    examples, train_examples, validation, state_text = load_data(args)
    summary = {
        "examples": len(examples),
        "unique_contexts": len({item.context_id for item in examples}),
        "unique_context_states": len(state_text),
        "train_examples": len(train_examples),
        "validation_examples": len(validation),
        "answers_loaded_into_examples": False,
        "options_loaded_into_examples": False,
        "input_mode": args.input_mode,
    }
    print(json.dumps(summary, indent=2))
    if args.mode == "dry-run":
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "dry_run.json").write_text(json.dumps(summary, indent=2) + "\n")
    elif args.mode == "cache":
        prepare_cache(args, state_text)
    elif args.mode == "train":
        train(args, train_examples)
    else:
        evaluate(args, validation)


if __name__ == "__main__":
    main()
