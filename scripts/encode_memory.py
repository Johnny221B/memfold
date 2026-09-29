#!/usr/bin/env python3
"""Cache frozen Qwen hidden states for one extracted text memory per question."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_opd.soft_reconstruction import load_text_memory_examples


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batches(values: list[list[int]], size: int) -> Iterable[list[list[int]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encoder-chunk-tokens", type=int, default=2048)
    parser.add_argument("--pool-tokens", type=int, default=32)
    parser.add_argument("--encoder-batch-size", type=int, default=4)
    parser.add_argument("--cache-shards", type=int, default=1)
    parser.add_argument("--cache-shard-index", type=int, default=0)
    args = parser.parse_args()
    if min(args.encoder_chunk_tokens, args.pool_tokens, args.encoder_batch_size) <= 0:
        parser.error("chunk, pool, and batch sizes must be positive")
    if args.cache_shards <= 0 or not 0 <= args.cache_shard_index < args.cache_shards:
        parser.error("cache shard index must be in [0, cache_shards)")

    _, state_text = load_text_memory_examples(args.questions, args.memories)
    selected = sorted(state_text.items())[args.cache_shard_index :: args.cache_shards]
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).requires_grad_(False).eval()
    decoder = getattr(model, "model", None)
    if decoder is None:
        raise TypeError(f"{type(model).__name__} does not expose .model")

    states_dir = args.cache_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for number, (state_id, text) in enumerate(selected, 1):
        output = states_dir / f"{state_id}.pt"
        if output.exists():
            payload = torch.load(output, map_location="cpu", weights_only=True)
            record = {"state_id": state_id, "vectors": int(payload["states"].shape[0]),
                      "tokens": int(payload["token_count"]), "resumed": True}
            records.append(record)
            print(json.dumps({"cached": number, "selected": len(selected), **record}), flush=True)
            continue
        token_ids = tokenizer(text, add_special_tokens=True).input_ids
        chunks = [token_ids[start:start + args.encoder_chunk_tokens]
                  for start in range(0, len(token_ids), args.encoder_chunk_tokens)]
        pooled: list[torch.Tensor] = []
        with torch.inference_mode():
            for group in batches(chunks, args.encoder_batch_size):
                width = max(map(len, group))
                ids = torch.full((len(group), width), tokenizer.pad_token_id,
                                 device=device, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for row, item in enumerate(group):
                    ids[row, :len(item)] = torch.tensor(item, device=device)
                    mask[row, :len(item)] = 1
                hidden = decoder(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
                for row, item in enumerate(group):
                    pooled.extend(
                        hidden[row, start:min(len(item), start + args.pool_tokens)].mean(0).cpu()
                        for start in range(0, len(item), args.pool_tokens)
                    )
        states = torch.stack(pooled).to(torch.float16)
        temporary = output.with_suffix(".tmp")
        torch.save({"states": states, "token_count": len(token_ids)}, temporary)
        temporary.replace(output)
        record = {"state_id": state_id, "vectors": int(states.shape[0]),
                  "tokens": len(token_ids), "resumed": False}
        records.append(record)
        print(json.dumps({"cached": number, "selected": len(selected), **record}), flush=True)

    manifest = {
        "schema_version": "text_memory_state_cache_v1",
        "input_mode": "text-memory",
        "split": args.split_name,
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256(args.model / "config.json"),
        "questions_sha256": sha256(args.questions),
        "memories_sha256": sha256(args.memories),
        "pool_tokens": args.pool_tokens,
        "states_expected": len(state_text),
        "states_selected": len(selected),
        "states_cached": len(records),
        "cache_shards": args.cache_shards,
        "cache_shard_index": args.cache_shard_index,
        "records": records,
    }
    path = args.cache_dir / (
        f"manifest-{args.split_name}-shard-{args.cache_shard_index:05d}"
        f"-of-{args.cache_shards:05d}.json"
    )
    path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
