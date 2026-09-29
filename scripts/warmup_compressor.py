#!/usr/bin/env python3
"""Fine-tune a soft-memory compressor with decoder-free all-context losses."""

from __future__ import annotations

PROCEDURE = "representation_warmup"

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from memory_opd.soft_reconstruction import (
    ContextResampler,
    ContextToSoftTokens,
    ReconstructionExample,
    SoftTokenProjector,
    load_reconstruction_examples,
    soft_memory_vector,
    split_examples_by_context,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--maximum-cross-context-cosine", type=float, default=0.8)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--gram-regularization-weight", type=float, default=0.1)
    parser.add_argument("--views-per-context", type=int, default=2)
    parser.add_argument("--validation-contexts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.procedure = "representation_warmup"
    return args


def cache_path(cache_dir: Path, state_id: str) -> Path:
    return cache_dir / "states" / f"{state_id}.pt"


def build_bridge(config: dict, device: torch.device, dtype: torch.dtype):
    return ContextToSoftTokens(
        ContextResampler(
            int(config["context_dim"]),
            latent_dim=int(config["latent_dim"]),
            token_count=int(config["token_count"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
            context_residual=bool(config.get("context_residual", False)),
        ),
        SoftTokenProjector(int(config["latent_dim"]), int(config["lm_dim"])),
    ).to(device=device, dtype=dtype)


def unique_states_by_context(
    examples: list[ReconstructionExample],
) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        grouped[example.context_id].add(example.state_id)
    return {context: sorted(states) for context, states in sorted(grouped.items())}


def all_context_losses(
    vectors: torch.Tensor,
    labels: torch.Tensor,
    *,
    maximum_cosine: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return negative hinge, within-context alignment, and prototype Gram losses."""
    if vectors.ndim != 2 or labels.shape != vectors.shape[:1]:
        raise ValueError("vectors/labels must have shapes [batch, dim]/[batch]")
    if not -1.0 <= maximum_cosine <= 1.0:
        raise ValueError("maximum cosine must be in [-1, 1]")
    cosine = vectors @ vectors.T
    diagonal = torch.eye(len(labels), device=vectors.device, dtype=torch.bool)
    same = labels[:, None] == labels[None, :]
    positive_mask = same & ~diagonal
    negative_mask = ~same
    minimum_distance = (2.0 - 2.0 * maximum_cosine) ** 0.5
    difference = vectors[:, None, :] - vectors[None, :, :]
    distance = torch.linalg.vector_norm(difference, dim=-1)
    separation_loss = torch.relu(minimum_distance - distance[negative_mask]).mean()
    alignment_loss = (1.0 - cosine[positive_mask]).mean()
    unique_labels = labels.unique(sorted=True)
    prototypes = torch.stack([
        vectors[labels == label].mean(dim=0) for label in unique_labels
    ])
    prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
    prototype_cosine = prototypes @ prototypes.T
    prototype_off_diagonal = ~torch.eye(
        len(prototypes), device=vectors.device, dtype=torch.bool
    )
    gram_regularization = prototype_cosine[prototype_off_diagonal].square().mean()
    negative_cosine = cosine[negative_mask]
    metrics = {
        "mean_cross_context_cosine": float(negative_cosine.mean().detach()),
        "maximum_cross_context_cosine": float(negative_cosine.max().detach()),
        "cross_context_collision_rate_cosine_ge_0_90": float(
            (negative_cosine >= 0.90).float().mean().detach()
        ),
        "mean_within_context_cosine": float(cosine[positive_mask].mean().detach()),
    }
    return separation_loss, alignment_loss, gram_regularization, metrics


def load_batch(
    selected_state_ids: list[str], cache_dir: Path, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [
        torch.load(cache_path(cache_dir, state_id), map_location="cpu", weights_only=True)[
            "states"
        ]
        for state_id in selected_state_ids
    ]
    width = max(row.shape[0] for row in rows)
    states = torch.zeros(
        len(rows), width, rows[0].shape[-1], device=device, dtype=dtype
    )
    mask = torch.zeros(len(rows), width, device=device, dtype=torch.long)
    for index, row in enumerate(rows):
        states[index, : row.shape[0]] = row.to(device=device, dtype=dtype)
        mask[index, : row.shape[0]] = 1
    return states, mask


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.steps, args.learning_rate, args.views_per_context) <= 0:
        raise ValueError("steps, learning rate, and views per context must be positive")
    if min(args.alignment_weight, args.gram_regularization_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    examples, _ = load_reconstruction_examples(
        args.questions, args.contexts, args.memories
    )
    train, _ = split_examples_by_context(
        examples, validation_contexts=args.validation_contexts, seed=args.seed
    )
    grouped = unique_states_by_context(train)
    if len(grouped) < 2:
        raise ValueError("at least two training contexts are required")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    bridge = build_bridge(saved["config"], device, dtype)
    bridge.load_state_dict(saved["bridge"])
    bridge.train()
    optimizer = torch.optim.AdamW(
        bridge.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output.mkdir(parents=True, exist_ok=True)
    context_ids = list(grouped)
    metrics_path = args.output / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to append to existing run: {metrics_path}")
    for step in range(1, args.steps + 1):
        state_ids = []
        labels = []
        for context_index, context_id in enumerate(context_ids):
            candidates = grouped[context_id]
            start = (args.seed + step + context_index) % len(candidates)
            for view in range(args.views_per_context):
                state_ids.append(candidates[(start + view) % len(candidates)])
                labels.append(context_index)
        states, mask = load_batch(state_ids, args.cache_dir, device, dtype)
        soft = bridge(states, mask)
        vectors = soft_memory_vector(soft)
        label_tensor = torch.tensor(labels, device=device, dtype=torch.long)
        separation_loss, alignment_loss, gram_regularization, diagnostics = all_context_losses(
            vectors,
            label_tensor,
            maximum_cosine=args.maximum_cross_context_cosine,
        )
        objective = (
            separation_loss
            + args.alignment_weight * alignment_loss
            + args.gram_regularization_weight * gram_regularization
        )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()
        row = {"procedure": PROCEDURE,
            "step": step,
            "objective_loss": float(objective.detach()),
            "separation_loss": float(separation_loss.detach()),
            "alignment_loss": float(alignment_loss.detach()),
            "gram_regularization": float(gram_regularization.detach()),
            "gradient_norm": float(gradient_norm),
            **diagnostics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(json.dumps(row), flush=True)
    checkpoint = args.output / f"checkpoint-representation_warmup-{args.steps:06d}.pt"
    config = dict(saved["config"])
    config["representation_training"] = {
        "base_checkpoint": str(args.checkpoint.resolve()),
        "steps": args.steps,
        "maximum_cross_context_cosine": args.maximum_cross_context_cosine,
        "alignment_weight": args.alignment_weight,
        "gram_regularization_weight": args.gram_regularization_weight,
        "views_per_context": args.views_per_context,
    }
    torch.save(
        {"procedure": PROCEDURE,
            "bridge": bridge.state_dict(),
            "config": config,
            "global_step": int(saved["global_step"]) + args.steps,
            "representation_steps": args.steps,
        },
        checkpoint,
    )
    result = {"procedure": PROCEDURE,
        "status": "completed",
        "decoder_used": False,
        "answers_used": False,
        "contexts_per_step": len(context_ids),
        "states_per_step": len(context_ids) * args.views_per_context,
        "checkpoint": str(checkpoint),
        "final_metrics": row,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
