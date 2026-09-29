#!/usr/bin/env python3
"""Extract question-blind atomic memories from PersonaMem and LoCoMo.

The script deliberately never loads PersonaMem questions/options/answers into an
API request and never loads LoCoMo QA annotations into an API request. PersonaMem
histories are split at the unique train-prefix cutoffs; LoCoMo is split by
session. Consequently, source text is sent once rather than once per question.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import json
from pathlib import Path
import time
from typing import Any, Iterable

import httpx


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.1"

INSTRUCTIONS = """You build a question-independent long-term memory bank.
Extract only atomic, user- or speaker-specific information from the supplied
history segment that could be useful in future interactions: identity,
preferences, goals, constraints, experiences, relationships, skills, and
explicit changes. The input contains no current question, options, or answer.

Rules:
- Ground every memory in one to four exact source IDs from this segment.
- Do not copy generic assistant advice or social pleasantries.
- Do not infer stereotypes or invent unstated causes, dates, or preferences.
- Keep names, negation, quantities, and temporal qualifiers precise.
- When this segment explicitly changes a state, record the change and the new
  state; do not silently erase the old state.
- Each statement must be self-contained and atomic.
- Return an empty memories array when the segment has no durable information.
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["memories"],
    "properties": {
        "memories": {
            "type": "array",
            "maxItems": 96,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "kind",
                    "subject",
                    "statement",
                    "temporal_status",
                    "event_time",
                    "evidence_ids",
                    "confidence",
                ],
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [
                            "identity",
                            "preference",
                            "goal",
                            "constraint",
                            "experience",
                            "relationship",
                            "skill",
                            "change",
                            "other",
                        ],
                    },
                    "subject": {"type": "string", "maxLength": 120},
                    "statement": {"type": "string", "maxLength": 360},
                    "temporal_status": {
                        "type": "string",
                        "enum": [
                            "stable",
                            "current_at_segment_end",
                            "historical",
                            "changed_within_segment",
                            "uncertain",
                        ],
                    },
                    "event_time": {"type": "string", "maxLength": 120},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        }
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_shared_contexts(path: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for row in read_jsonl(path):
        overlap = set(result).intersection(row)
        if overlap:
            raise ValueError(f"duplicate shared context IDs: {sorted(overlap)[:3]}")
        result.update(row)
    return result


def personamem_jobs(
    *, scale: str, questions: Path, contexts_path: Path
) -> list[dict[str, Any]]:
    rows = read_jsonl(questions)
    contexts = load_shared_contexts(contexts_path)
    cutoffs: dict[str, set[int]] = {}
    for row in rows:
        context_id = str(row["shared_context_id"])
        cutoff = int(row["end_index_in_shared_context"])
        if cutoff < 0:
            cutoff = len(contexts[context_id])
        cutoffs.setdefault(context_id, set()).add(cutoff)

    jobs: list[dict[str, Any]] = []
    for context_id in sorted(cutoffs):
        messages = contexts[context_id]
        start = 0
        for end in sorted(cutoffs[context_id]):
            if not 0 < end <= len(messages):
                raise ValueError(f"invalid cutoff {end} for {context_id}")
            events = []
            evidence_ids = []
            for index, message in enumerate(messages[start:end], start):
                evidence_id = f"m{index:04d}"
                evidence_ids.append(evidence_id)
                events.append(
                    f"[{evidence_id}] {str(message['role']).upper()}: "
                    f"{str(message['content']).strip()}"
                )
            job_id = f"personamem-{scale}-{context_id}-{start:04d}-{end:04d}"
            jobs.append(
                {
                    "job_id": job_id,
                    "dataset": f"PersonaMem-{scale}",
                    "context_id": context_id,
                    "segment_id": f"messages[{start}:{end}]",
                    "prefix_end": end,
                    "evidence_ids": evidence_ids,
                    "input": "\n".join(events),
                }
            )
            start = end
    return jobs


def _session_number(key: str) -> int:
    return int(key.removeprefix("session_"))


def locomo_jobs(path: Path) -> list[dict[str, Any]]:
    conversations = json.loads(path.read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = []
    for conversation in sorted(conversations, key=lambda item: item["sample_id"]):
        context_id = str(conversation["sample_id"])
        body = conversation["conversation"]
        session_keys = sorted(
            (
                key
                for key, value in body.items()
                if key.startswith("session_")
                and not key.endswith("_date_time")
                and isinstance(value, list)
            ),
            key=_session_number,
        )
        for session_key in session_keys:
            timestamp = str(body.get(f"{session_key}_date_time", "")).strip()
            events = [f"[SESSION_TIME] {timestamp}"]
            evidence_ids = []
            for turn in body[session_key]:
                evidence_id = str(turn["dia_id"])
                evidence_ids.append(evidence_id)
                line = (
                    f"[{evidence_id}] {str(turn['speaker']).strip()}: "
                    f"{str(turn['text']).strip()}"
                )
                caption = str(turn.get("blip_caption", "")).strip()
                if caption:
                    line += f" [Shared image: {caption}]"
                events.append(line)
            jobs.append(
                {
                    "job_id": f"locomo-{context_id}-{session_key}",
                    "dataset": "LoCoMo",
                    "context_id": context_id,
                    "segment_id": session_key,
                    "prefix_end": _session_number(session_key),
                    "evidence_ids": evidence_ids,
                    "input": "\n".join(events),
                }
            )
    return jobs


def response_text(body: dict[str, Any]) -> str:
    pieces = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        pieces.extend(
            content.get("text", "")
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        )
    return "".join(pieces)


def request_memory(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    model: str,
    job: dict[str, Any],
    max_output_tokens: int,
    retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": job["input"],
        "reasoning": {"effort": "none"},
        "max_output_tokens": max_output_tokens,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "question_blind_atomic_memory",
                "strict": True,
                "schema": SCHEMA,
            }
        },
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.post(
                base_url.rstrip("/") + "/responses", headers=headers, json=payload
            )
            response.raise_for_status()
            body = response.json()
            parsed = json.loads(response_text(body))
            allowed = set(job["evidence_ids"])
            invalid = sorted(
                {
                    evidence_id
                    for memory in parsed["memories"]
                    for evidence_id in memory["evidence_ids"]
                    if evidence_id not in allowed
                }
            )
            if invalid:
                raise ValueError(f"invalid evidence IDs: {invalid[:8]}")
            return parsed, body
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            last_error = error
            if attempt >= retries:
                break
            time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"request failed after {retries + 1} attempts: {last_error}")


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("personamem32k", "personamem128k", "locomo"),
        default=("personamem32k", "personamem128k", "locomo"),
    )
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--locomo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace
    jobs: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    if "personamem32k" in args.datasets:
        questions = workspace / "prepared/32k/personamem_32k_train.jsonl"
        contexts = workspace / "personamem_v1_raw/shared_contexts_32k.jsonl"
        jobs.extend(personamem_jobs(scale="32K", questions=questions, contexts_path=contexts))
        sources[str(questions)] = sha256_file(questions)
        sources[str(contexts)] = sha256_file(contexts)
    if "personamem128k" in args.datasets:
        questions = workspace / "prepared/128k/personamem_128k_train.jsonl"
        contexts = workspace / "personamem_v1_raw/shared_contexts_128k.jsonl"
        jobs.extend(personamem_jobs(scale="128K", questions=questions, contexts_path=contexts))
        sources[str(questions)] = sha256_file(questions)
        sources[str(contexts)] = sha256_file(contexts)
    if "locomo" in args.datasets:
        locomo = args.locomo or workspace / "datasets/locomo/locomo10.json"
        jobs.extend(locomo_jobs(locomo))
        sources[str(locomo)] = sha256_file(locomo)

    summary = {
        "jobs": len(jobs),
        "by_dataset": {
            name: sum(job["dataset"] == name for job in jobs)
            for name in sorted({job["dataset"] for job in jobs})
        },
        "input_characters": sum(len(job["input"]) for job in jobs),
        "sources": sources,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        completed = {row["job_id"]: row for row in read_jsonl(args.output)}
    pending = [job for job in jobs if job["job_id"] not in completed]
    if args.max_jobs is not None:
        pending = pending[: args.max_jobs]
    api_key = (args.api_key_file.read_text(encoding="utf-8").strip() if args.api_key_file else os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        raise ValueError("API key file is empty")

    with httpx.Client(timeout=args.timeout) as client:
        for number, job in enumerate(pending, 1):
            parsed, body = request_memory(
                client,
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                job=job,
                max_output_tokens=args.max_output_tokens,
                retries=args.retries,
            )
            record = {
                key: value for key, value in job.items() if key not in {"input", "evidence_ids"}
            }
            record.update(
                {
                    "model": args.model,
                    "prompt_version": sha256_bytes(INSTRUCTIONS.encode())[:16],
                    "input_sha256": sha256_bytes(job["input"].encode()),
                    "source_evidence_count": len(job["evidence_ids"]),
                    "memories": parsed["memories"],
                    "response_id": body.get("id"),
                    "response_status": body.get("status"),
                    "usage": body.get("usage", {}),
                    "completed_unix_time": time.time(),
                }
            )
            completed[job["job_id"]] = record
            ordered = [completed[item["job_id"]] for item in jobs if item["job_id"] in completed]
            atomic_write_jsonl(args.output, ordered)
            print(
                json.dumps(
                    {
                        "completed": number,
                        "selected_pending": len(pending),
                        "job_id": job["job_id"],
                        "memories": len(parsed["memories"]),
                        "usage": body.get("usage", {}),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    manifest = {
        "schema_version": "question_blind_atomic_memory_v1",
        "model": args.model,
        "base_url_host": args.base_url.split("//", 1)[-1].split("/", 1)[0],
        "prompt_sha256": sha256_bytes(INSTRUCTIONS.encode()),
        "question_answer_fields_sent_to_api": False,
        "jobs_total": len(jobs),
        "jobs_completed": len(completed),
        "sources": sources,
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
