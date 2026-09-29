#!/usr/bin/env python3
"""Build token-complete, message-boundary PersonaMem records for chunked MemGen."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from memory_opd.baselines.memgen.persona_data import SYSTEM, gold_option, load_contexts, render_history
from memory_opd.baselines.memgen.strict_data import OFFICIAL_CHAT_TEMPLATE


def token_length(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def split_oversize_text(tokenizer, text, maximum):
    """Split one rendered message without dropping or rewriting any characters."""
    pieces = []
    remaining = text
    while remaining:
        if token_length(tokenizer, remaining) <= maximum:
            pieces.append(remaining)
            break
        low, high = 1, len(remaining)
        while low < high:
            middle = (low + high + 1) // 2
            if token_length(tokenizer, remaining[:middle]) <= maximum:
                low = middle
            else:
                high = middle - 1
        if low <= 0:
            raise ValueError("chunk token limit cannot encode even one source character")
        piece = remaining[:low]
        if token_length(tokenizer, piece) > maximum:
            raise AssertionError("oversize splitter emitted an invalid chunk")
        pieces.append(piece)
        remaining = remaining[low:]
    if "".join(pieces) != text:
        raise AssertionError("oversize splitter changed source text")
    return pieces


def chunk_messages(tokenizer, messages, maximum):
    """Return exact rendered chunks using O(chunks * log(messages)) tokenizations.

    The former append-and-retokenize loop was quadratic in conversation length.
    Binary search preserves the same greedy, message-boundary result.  A single
    rendered message above the GPU-safe limit is split into contiguous source-text
    slices; this is audited separately in the manifest.
    """
    rendered = [render_history([message]) for message in messages]
    chunks = []
    split_messages = 0
    start = 0
    while start < len(rendered):
        if token_length(tokenizer, rendered[start]) > maximum:
            chunks.extend(split_oversize_text(tokenizer, rendered[start], maximum))
            split_messages += 1
            start += 1
            continue
        low, high = start + 1, len(rendered)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = "\n\n".join(rendered[start:middle])
            if token_length(tokenizer, candidate) <= maximum:
                low = middle
            else:
                high = middle - 1
        chunks.append("\n\n".join(rendered[start:low]))
        start = low
    lengths = [token_length(tokenizer, chunk) for chunk in chunks]
    if not chunks or any(length > maximum for length in lengths):
        raise AssertionError("invalid chunking result")
    return chunks, lengths, split_messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "val"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=28672)
    parser.add_argument("--smoke-longest", action="store_true")
    parser.add_argument("--smoke-largest-message", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke_longest and args.smoke_largest_message:
        raise ValueError("choose only one smoke selector")
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("output and manifest must be new")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.chat_template = OFFICIAL_CHAT_TEMPLATE
    split = json.loads(args.split.read_text(encoding="utf-8"))
    allowed = set(split["question_ids"][args.partition])
    contexts = load_contexts(args.contexts)
    rows = [row for row in csv.DictReader(args.questions.open(newline="", encoding="utf-8"))
            if row["question_id"] in allowed]
    if len(rows) != len(allowed):
        raise ValueError("frozen partition IDs do not match question rows")
    if args.smoke_longest:
        rows = [max(rows, key=lambda row: len(render_history(
            contexts[row["shared_context_id"]][:int(row["end_index_in_shared_context"])]
        )))]
    elif args.smoke_largest_message:
        rows = [max(rows, key=lambda row: max(
            len(render_history([message])) for message in
            contexts[row["shared_context_id"]][:int(row["end_index_in_shared_context"])]
        ))]
    records = []
    split_oversize_messages = 0
    for row in rows:
        messages = contexts[row["shared_context_id"]][:int(row["end_index_in_shared_context"])]
        chunk_texts, chunk_lengths, split_messages = chunk_messages(
            tokenizer, messages, args.chunk_tokens
        )
        split_oversize_messages += split_messages
        options = list(ast.literal_eval(row["all_options"]))
        choices = "\n".join(f"({chr(ord('a') + i)}) {x}" for i, x in enumerate(options))
        task_messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": (
                f"QUESTION:\n{row['user_question_or_message']}\n\nOPTIONS:\n{choices}\n\nANSWER:"
            )},
            {"role": "assistant", "content": gold_option(row["correct_answer"], options)},
        ]
        records.append({
            "schema_version": "persona_memgen_chunked_v1",
            "trajectory_id": row["question_id"],
            "shared_context_id": row["shared_context_id"],
            "partition": args.partition,
            "legal_end_index": int(row["end_index_in_shared_context"]),
            "memory_chunks": chunk_texts,
            "chunk_token_lengths": chunk_lengths,
            "source_memory_tokens": sum(chunk_lengths),
            "task_messages": task_messages,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
    manifest = {
        "schema_version": "persona_memgen_chunked_v1",
        "records": len(records), "partition_records": len(allowed),
        "smoke_longest": args.smoke_longest,
        "smoke_largest_message": args.smoke_largest_message,
        "chunk_tokens": args.chunk_tokens,
        "max_chunks": max(len(x["memory_chunks"]) for x in records),
        "max_source_memory_tokens": max(x["source_memory_tokens"] for x in records),
        "max_single_chunk_tokens": max(max(x["chunk_token_lengths"]) for x in records),
        "truncated_messages": 0, "dropped_messages": 0,
        "split_oversize_messages": split_oversize_messages,
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
