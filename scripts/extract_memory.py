#!/usr/bin/env python3
"""Extract answer-blind PersonaMem memories tailored to a target Qwen backbone."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


SCHEMA_KEYS = ("evidence", "temporal_relations", "derived_facts")

COMMON = """You extract question-conditioned memory from a long chronological dialogue.
The memory will later be read by a Qwen language model to answer the CURRENT QUESTION.

Use only facts present in HISTORY up to the supplied endpoint. The CURRENT QUESTION may be used
only to select relevant history. Never answer the question. Never infer or mention answer options,
option labels, a gold answer, a question type, or future dialogue. Resolve speaker references and
write standalone facts using "The user" as the subject. Preserve exact entities, polarity, event
order, preference changes, reversals, and the latest state. Do not invent facts.

Return exactly one JSON object, with no markdown or surrounding prose:
{"evidence":["..."],"temporal_relations":["..."],"derived_facts":["..."]}
"""

BACKBONE_PROMPTS = {
    "qwen25-3b": COMMON + """
TARGET READER: Qwen2.5-3B-Instruct. Optimize for a smaller reader.
- Use short, literal, unambiguous sentences; one fact per sentence.
- Put the most decisive question-relevant facts first.
- Include 3-7 evidence facts, 0-3 explicit temporal relations, and 1-3 conservative derived facts.
- State a changed preference explicitly as earlier state -> later/current state and include its reason.
- Avoid decorative details, vague summaries, pronouns, duplication, and implicit multi-hop reasoning.
- Keep the complete JSON below 320 English words.
""",
    "qwen25-7b": COMMON + """
TARGET READER: Qwen2.5-7B-Instruct. Optimize for balanced coverage and explicit chronology.
- Include 5-10 concise evidence facts, 1-5 temporal relations, and 1-4 derived facts.
- Preserve competing or contradictory evidence when relevant, then identify the latest supported state.
- Make multi-event comparisons explicit instead of expecting the reader to reconstruct the timeline.
- Retain exact reasons and distinctive details that separate plausible alternatives.
- Avoid irrelevant biography, verbose atmosphere, duplication, and unsupported generalization.
- Keep the complete JSON below 500 English words.
""",
    "qwen3-4b": COMMON + """
TARGET READER: Qwen3-4B-Instruct. Optimize for compact evidence-grounded multi-hop reasoning.
- Include 4-9 decisive evidence facts, 1-5 temporal relations, and 1-4 conservative derived facts.
- Separate observations from derived conclusions and link multi-hop evidence explicitly.
- Preserve preference trajectories, reversals, causes, and the current state without redundant wording.
- Retain details that distinguish near-neighbor events or concepts; omit unrelated background.
- Keep the complete JSON below 450 English words.
""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", choices=sorted(BACKBONE_PROMPTS), required=True)
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key_from_args(args: argparse.Namespace) -> str:
    if args.api_key_file:
        key = args.api_key_file.read_text(encoding="utf-8").strip()
    else:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("API key missing: pass --api-key-file or set OPENAI_API_KEY")
    return key


def validate_memory(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict) or set(value) != set(SCHEMA_KEYS):
        raise ValueError(f"expected exactly keys {SCHEMA_KEYS}")
    normalized: dict[str, list[str]] = {}
    for key in SCHEMA_KEYS:
        items = value[key]
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise ValueError(f"{key} must be an array of strings")
        normalized[key] = [item.strip() for item in items if item.strip()]
    if not normalized["evidence"]:
        raise ValueError("evidence must not be empty")
    return normalized


def parse_memory(text: str) -> dict[str, list[str]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contains no JSON object")
    return validate_memory(json.loads(cleaned[start : end + 1]))


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return {name: getattr(usage, name) for name in (
        "input_tokens", "output_tokens", "total_tokens"
    ) if getattr(usage, name, None) is not None}


def writer_content(row: dict[str, Any]) -> str:
    messages = row.get("writer_messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {row.get('task_id')} has no writer_messages")
    user_parts = [str(message.get("content", "")) for message in messages if message.get("role") == "user"]
    if not user_parts:
        raise ValueError(f"row {row.get('task_id')} has no user writer message")
    content = "\n\n".join(user_parts)
    # Gold labels and shuffled options live in sibling fields and are deliberately never serialized.
    return content


def request_one(
    client: OpenAI,
    row: dict[str, Any],
    *,
    backbone: str,
    model: str,
    max_output_tokens: int,
    max_attempts: int,
) -> dict[str, Any]:
    content = writer_content(row)
    prompt = BACKBONE_PROMPTS[backbone] + "\n\nSOURCE DIALOGUE AND QUESTION:\n" + content
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )
            memory = parse_memory(response.output_text)
            memory_text = json.dumps(memory, ensure_ascii=False, separators=(",", ":"))
            return {
                "schema_version": "personamem-backbone-memory-gpt51-v1",
                "question_id": str(row["task_id"]),
                "task_id": str(row["task_id"]),
                "split": str(row.get("split", "unknown")),
                "target_backbone": backbone,
                "extractor_model": model,
                "prompt_sha256": prompt_hash,
                "memory": memory,
                "memory_text": memory_text,
                "usage": usage_dict(response),
                "student_saw_answer": False,
                "student_saw_options": False,
            }
        except Exception as error:  # network, upstream, or strict parsing failure
            last_error = error
            if attempt < max_attempts:
                time.sleep(min(20.0, (2 ** (attempt - 1)) + random.random()))
    raise RuntimeError(f"failed after {max_attempts} attempts: {last_error}") from last_error


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.max_attempts <= 0:
        raise SystemExit("--workers and --max-attempts must be positive")
    rows = read_jsonl(args.inputs)
    if args.limit is not None:
        rows = rows[: args.limit]
    ids = [str(row["task_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate task_id in inputs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    completed_rows = read_jsonl(args.output) if args.output.exists() else []
    completed = {str(row["task_id"]) for row in completed_rows}
    pending = [row for row in rows if str(row["task_id"]) not in completed]
    client = OpenAI(api_key=key_from_args(args), base_url=args.base_url)

    failures: list[dict[str, str]] = []
    with args.output.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    request_one,
                    client,
                    row,
                    backbone=args.backbone,
                    model=args.model,
                    max_output_tokens=args.max_output_tokens,
                    max_attempts=args.max_attempts,
                ): row
                for row in pending
            }
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    failures.append({"task_id": str(row["task_id"]), "error": str(error)})
                    print(json.dumps({"finished": index, "total": len(pending), "failed": str(row["task_id"])}), flush=True)
                    continue
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({"finished": index, "total": len(pending), "task_id": result["task_id"]}), flush=True)

    failure_path = args.output.with_suffix(args.output.suffix + ".failures.json")
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "personamem-backbone-memory-extraction-manifest-v1",
        "inputs": str(args.inputs.resolve()),
        "output": str(args.output.resolve()),
        "target_backbone": args.backbone,
        "extractor_model": args.model,
        "requested": len(rows),
        "previously_completed": len(completed),
        "newly_completed": len(pending) - len(failures),
        "failures": len(failures),
        "workers": args.workers,
        "answer_blind": True,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} requests failed; rerun to resume missing rows")


if __name__ == "__main__":
    main()
