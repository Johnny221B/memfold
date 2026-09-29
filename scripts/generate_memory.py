#!/usr/bin/env python3
"""Generate deterministic evidence-v1 self memories from a frozen OPD adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from memory_opd.compressed_opd import canonical_memory, parse_memory_text, read_jsonl


MEMORY_BUDGET = (
    "\n\nOUTPUT BUDGET: Return no more than 8 evidence items, 4 temporal_relations "
    "items, and 4 derived_facts items. Keep every item at most 160 characters. "
    "Finish the complete JSON object within 1200 tokens."
)


def compact_memory_schema() -> dict[str, Any]:
    fact = {"type": "string", "minLength": 1, "maxLength": 160}
    return {
        "type": "object",
        "properties": {
            "evidence": {"type": "array", "items": fact, "maxItems": 8},
            "temporal_relations": {"type": "array", "items": fact, "maxItems": 4},
            "derived_facts": {"type": "array", "items": fact, "maxItems": 4},
        },
        "required": ["evidence", "temporal_relations", "derived_facts"],
        "additionalProperties": False,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def tolerant_memory(text: str) -> tuple[dict[str, list[str]], bool]:
    """Keep occasional schema failures usable instead of aborting a whole split."""

    try:
        return canonical_memory(parse_memory_text(text)), True
    except ValueError:
        pass

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None

    memory: dict[str, list[str]] = {
        "evidence": [],
        "temporal_relations": [],
        "derived_facts": [],
    }
    if isinstance(value, dict):
        for key in memory:
            items = value.get(key, [])
            if isinstance(items, str) and items.strip():
                memory[key] = [items.strip()]
            elif isinstance(items, list):
                memory[key] = [str(item).strip() for item in items if str(item).strip()]

    if not any(memory.values()) and text.strip():
        memory["evidence"] = [text.strip()]
    return canonical_memory(memory), False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--writer-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split", choices=("train", "validation", "test"))
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--maximum-memory-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument(
        "--structured-output",
        action="store_true",
        help=(
            "Constrain decoding with a JSON schema. Disabled by default because "
            "vLLM 0.11 V1 can stall before the first long-context request."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output}")

    rows = read_jsonl(args.writer_inputs)
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} writer rows, found {len(rows)}")
    if args.expected_split is not None and any(
        str(row.get("split")) != args.expected_split for row in rows
    ):
        raise ValueError(f"writer inputs are not exclusively {args.expected_split}")
    if len({str(row["task_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate writer task IDs")
    if args.limit:
        rows = rows[: args.limit]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = []
    for row in rows:
        messages = [dict(message) for message in row["writer_messages"]]
        if not messages or messages[0].get("role") != "system":
            raise ValueError(f"missing writer system prompt: {row['task_id']}")
        messages[0]["content"] = str(messages[0]["content"]) + MEMORY_BUDGET
        serialized = json.dumps(messages, ensure_ascii=False)
        if str(row.get("gold_label", "")) in serialized:
            # A label character may naturally occur; reject only the explicit answer field form.
            if "gold_label" in serialized or "Correct answer:" in serialized:
                raise ValueError(f"privileged answer leaked into writer messages: {row['task_id']}")
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            truncation=True,
            max_length=131000,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(tokenizer.decode(prompt_ids, skip_special_tokens=False))

    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=64,
    )
    sampling_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": args.maximum_memory_tokens,
        "seed": args.seed,
    }
    if args.structured_output:
        sampling_kwargs["structured_outputs"] = StructuredOutputsParams(
            json=compact_memory_schema()
        )
    outputs = llm.generate(
        prompts,
        SamplingParams(**sampling_kwargs),
        lora_request=LoRARequest("epoch3-batch7-writer", 1, str(args.adapter)),
        use_tqdm=True,
    )

    generated: list[dict[str, Any]] = []
    valid_count = 0
    for row, prompt, output in zip(rows, prompts, outputs):
        text = output.outputs[0].text.strip()
        memory, schema_valid = tolerant_memory(text)
        valid_count += int(schema_valid)
        generated.append(
            {
                "schema_version": "evidence-v1",
                "question_id": str(row["task_id"]),
                "task_id": str(row["task_id"]),
                "split": str(row["split"]),
                "memory": memory,
                "memory_text": json.dumps(memory, ensure_ascii=False, separators=(",", ":")),
                "memory_tokens": len(output.outputs[0].token_ids),
                "schema_valid": schema_valid,
                "writer_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "source_adapter": str(args.adapter.resolve()),
            }
        )

    args.output.mkdir(parents=True, exist_ok=False)
    memories_path = args.output / "memories.jsonl"
    write_jsonl_atomic(memories_path, generated)
    manifest = {
        "schema_version": "personamem-compressed-opd-self-memory-v1",
        "model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "writer_inputs": str(args.writer_inputs.resolve()),
        "writer_inputs_sha256": sha256(args.writer_inputs),
        "rows": len(generated),
        "valid": valid_count,
        "valid_rate": valid_count / len(generated),
        "decoding": (
            "greedy-json-schema" if args.structured_output else "greedy-native-json"
        ),
        "maximum_memory_tokens": args.maximum_memory_tokens,
        "seed": args.seed,
        "answers_sent_to_writer": False,
        "memories": str(memories_path),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
