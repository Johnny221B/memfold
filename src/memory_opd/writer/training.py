"""Leakage-safe PersonaMem extraction-only SFT records and tokenization."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Mapping, Sequence

from .glm_client import MEMORY_V1_KEYS, validate_memory


def _shorten_text(value: str, maximum_characters: int) -> str:
    text = " ".join(value.split())
    if len(text) <= maximum_characters:
        return text
    shortened = text[:maximum_characters].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return shortened or text[:maximum_characters]


def _deduplicate_strings(values: Sequence[str], maximum_items: int, maximum_characters: int) -> list[str]:
    kept: list[str] = []
    seen: set[str] = set()
    for value in values:
        shortened = _shorten_text(value, maximum_characters)
        key = " ".join(shortened.casefold().split())
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(shortened)
        if len(kept) == maximum_items:
            break
    return kept


def compact_memory_target(
    memory: Mapping[str, Any],
    *,
    maximum_stable: int = 14,
    maximum_current: int = 18,
    maximum_changes: int = 8,
    maximum_fact_characters: int = 176,
    maximum_change_field_characters: int = 144,
) -> dict[str, Any]:
    """Deterministically shorten a validated v1 target without changing its schema."""

    validated = validate_memory(memory)
    stable = _deduplicate_strings(
        validated["stable"], maximum_stable, maximum_fact_characters
    )
    current = _deduplicate_strings(
        validated["current"], maximum_current, maximum_fact_characters
    )
    changes: list[dict[str, str]] = []
    seen_changes: set[tuple[str, str, str, str]] = set()
    for change in validated["changes"]:
        shortened = {
            key: _shorten_text(change[key], maximum_change_field_characters)
            for key in ("topic", "previous", "current", "reason")
        }
        identity = tuple(shortened[key].casefold() for key in (
            "topic", "previous", "current", "reason"))
        if identity in seen_changes:
            continue
        seen_changes.add(identity)
        changes.append(shortened)
        if len(changes) == maximum_changes:
            break
    return validate_memory({"stable": stable, "current": current, "changes": changes})


def parse_generated_memory(text: str) -> dict[str, Any]:
    """Extract and validate one v1 JSON object from generated text."""

    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start < 0:
            raise ValueError("generated output contains no JSON object")
        try:
            value, _ = json.JSONDecoder().raw_decode(stripped[start:])
        except json.JSONDecodeError as error:
            # Greedy decoding occasionally emits a complete v1 object except for
            # its final root brace. Repair only that unambiguous one-character
            # truncation; do not guess missing fields or array contents.
            try:
                value = json.loads(stripped[start:] + "}")
            except json.JSONDecodeError:
                raise ValueError(
                    "generated output does not contain valid JSON") from error
    memory = validate_memory(value)
    if set(memory) != MEMORY_V1_KEYS:
        raise ValueError("generated memory does not use the v1 schema")
    return memory


def build_sft_records(
    requests: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    *,
    compact_target: bool = False,
    maximum_stable: int = 14,
    maximum_current: int = 18,
    maximum_changes: int = 8,
    maximum_fact_characters: int = 176,
    maximum_change_field_characters: int = 144,
) -> list[dict[str, Any]]:
    response_by_id = {str(row["task_id"]): row for row in responses}
    if len(response_by_id) != len(responses):
        raise ValueError("duplicate response task IDs")
    records: list[dict[str, Any]] = []
    for request in requests:
        task_id = str(request["task_id"])
        if task_id not in response_by_id:
            raise ValueError(f"missing response for {task_id}")
        response = response_by_id[task_id]
        if request["split"] != "train" or request["prompt_version"] != "personamem-memory-v1":
            raise ValueError(
                "SFT accepts only train personamem-memory-v1 records")
        if response["prompt_version"] != request["prompt_version"]:
            raise ValueError(f"prompt version mismatch for {task_id}")
        expected_hash = hashlib.sha256(
            json.dumps(request, ensure_ascii=False,
                       sort_keys=True).encode("utf-8")
        ).hexdigest()
        if response["request_sha256"] != expected_hash:
            raise ValueError(f"request hash mismatch for {task_id}")
        memory = validate_memory(response["memory"])
        if set(memory) != MEMORY_V1_KEYS:
            raise ValueError(f"non-v1 memory for {task_id}")
        if compact_target:
            memory = compact_memory_target(
                memory,
                maximum_stable=maximum_stable,
                maximum_current=maximum_current,
                maximum_changes=maximum_changes,
                maximum_fact_characters=maximum_fact_characters,
                maximum_change_field_characters=maximum_change_field_characters,
            )
        messages = [dict(message) for message in request["messages"]]
        if [message["role"] for message in messages] != ["system", "user"]:
            raise ValueError(f"unexpected request messages for {task_id}")
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(memory, ensure_ascii=False, separators=(",", ":")),
            }
        )
        records.append(
            {
                "id": task_id,
                "split": "train",
                "messages": messages,
                "metadata": {
                    "shared_context_id": request["shared_context_id"],
                    "history_end_index": request["history_end_index"],
                    "history_sha256": request["history_sha256"],
                    "prompt_version": request["prompt_version"],
                    "target_compaction": (
                        {
                            "maximum_stable": maximum_stable,
                            "maximum_current": maximum_current,
                            "maximum_changes": maximum_changes,
                            "maximum_fact_characters": maximum_fact_characters,
                            "maximum_change_field_characters": maximum_change_field_characters,
                        }
                        if compact_target
                        else None
                    ),
                },
            }
        )
    if set(response_by_id) != {str(row["task_id"]) for row in requests}:
        raise ValueError("responses contain task IDs absent from requests")
    return records


def encode_sft_record(tokenizer, record: Mapping[str, Any], max_length: int):
    """Tokenize one chat and mask everything except the assistant memory target."""

    prompt = tokenizer.apply_chat_template(
        record["messages"][:-1], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    cut = max(0, len(full_ids) - max_length)
    input_ids = full_ids[cut:]
    boundary = max(0, len(prompt_ids) - cut)
    labels = [-100] * min(boundary, len(input_ids)) + \
        input_ids[min(boundary, len(input_ids)):]
    target_tokens = sum(token != -100 for token in labels)
    if target_tokens == 0:
        raise ValueError(f"assistant target was truncated: {record['id']}")
    return input_ids, labels, {
        "tokens": len(input_ids),
        "target_tokens": target_tokens,
        "left_truncated": cut,
    }


def encode_weighted_multiturn_record(tokenizer, record: Mapping[str, Any], max_length: int):
    """Mask user text and assign independent weights to every assistant response."""

    messages = record["messages"]
    assistant_weights = list(record.get(
        "metadata", {}).get("assistant_loss_weights", []))
    assistant_count = sum(
        message["role"] == "assistant" for message in messages)
    if len(assistant_weights) != assistant_count or any(weight <= 0 for weight in assistant_weights):
        raise ValueError(
            "assistant_loss_weights must contain one positive value per assistant message")
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(full, add_special_tokens=False,
                        return_offsets_mapping=True)
    full_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    labels = [-100] * len(full_ids)
    loss_weights = [0.0] * len(full_ids)
    assistant_index = 0
    character_cursor = 0
    for message_index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        content = str(message["content"])
        content_start = full.find(content, character_cursor)
        if content_start < 0:
            raise ValueError(
                f"assistant content is absent from rendered chat: {record['id']}")
        content_end = content_start + len(content)
        weight = float(assistant_weights[assistant_index])
        selected = [
            token_index
            for token_index, (start, end) in enumerate(offsets)
            if end > content_start and start < content_end
        ]
        if not selected:
            raise ValueError(
                f"assistant content produced no supervised tokens: {record['id']}")
        for token_index in selected:
            labels[token_index] = full_ids[token_index]
            loss_weights[token_index] = weight
        character_cursor = content_end
        assistant_index += 1
    cut = max(0, len(full_ids) - max_length)
    input_ids, labels, loss_weights = full_ids[cut:
                                               ], labels[cut:], loss_weights[cut:]
    target_tokens = sum(label != -100 for label in labels)
    if target_tokens == 0:
        raise ValueError(
            f"all assistant targets were truncated: {record['id']}")
    return input_ids, labels, loss_weights, {
        "tokens": len(input_ids), "target_tokens": target_tokens, "left_truncated": cut,
        "weighted_multiturn": True,
    }


def ordered_record_indices(
    target_lengths: Sequence[int], *, seed: int, epoch_index: int, short_first: bool
) -> list[int]:
    """Return a deterministic epoch order, optionally using a length curriculum."""

    indices = list(range(len(target_lengths)))
    if short_first and epoch_index == 0:
        return sorted(indices, key=lambda index: (target_lengths[index], index))
    random.Random(seed + epoch_index).shuffle(indices)
    return indices
