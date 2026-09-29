#!/usr/bin/env python3
"""Question-blind LoCoMo memory extraction with GLM's JSON API.

The extractor sends one timestamped session per request.  QA annotations and
gold answers are never included.  Output records retain exact dialogue IDs so
that extraction quality can later be checked against LoCoMo evidence labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable

import httpx


GLM_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
SCHEMA_VERSION = "ood_atomic_memory_v3"

SYSTEM_PROMPT = """You extract a dataset-independent long-term memory bank from a conversation segment.
The input contains only a timestamped dialogue session. It never contains a test question or answer.

Return exactly one JSON object with a `memories` array. Each memory must contain exactly:
- memory_type: one of fact, event, preference, goal, relationship, state, state_change, constraint
- subject: canonical person or entity name
- facet: one of identity, family, relationship, location, education, career, health, preference,
  activity, possession, plan, life_event, belief, constraint, other
- attribute: a short snake_case property within the facet, such as favorite_color or residence
- value: the shortest answer-like value that preserves names, quantities, and negation
- statement: one self-contained natural-language proposition
- polarity: positive or negative
- temporal: an object with source_expression, start, end, granularity, certainty; start/end must be
  ISO-8601 dates or empty strings. For a point date set start=end. For a week/month/year interval,
  use the actual boundary dates when deterministic.
- state_status: one of stable, current, historical, planned, superseded, uncertain
- explicitness: direct if directly stated, or strict_entailment only if logically necessary
- evidence_ids: one to four exact dialogue IDs copied from the input
- confidence: a number from 0 to 1

Extraction rules:
1. Extract only durable personal facts, preferences, goals, constraints, relationships, experiences,
   dated events, and explicit state changes that could matter in a future conversation.
2. Be question-blind: do not guess future questions and do not optimize for benchmark answer wording.
3. Every memory must be directly grounded in its evidence IDs. Do not infer stereotypes, unstated
   motivations, causal explanations, or missing values.
4. Do not infer friendship merely because two people are chatting. Do not infer a person's identity
   from an organization they mention, support, visit, or select. Do not convert assistant suggestions
   or acknowledgements into facts about either speaker.
5. Resolve relative dates only when the session timestamp makes the resolution deterministic.
   Preserve the original phrase in temporal.source_expression. Represent "last week" as a week
   interval, not an arbitrarily selected day. If unresolved, leave start and end empty.
6. Keep negation, quantities, names, ownership, and who said what exact.
7. Prefer direct self-reports. Use strict_entailment sparingly and only for logically necessary facts.
8. Prefer atomic propositions, but do not emit a weaker duplicate when a stronger memory already
   captures the same fact in this segment (for example, has_kids plus has_children).
9. A state change must explicitly state both a change and the new state. A newly felt emotion alone
   is not a durable state change. Use superseded only when this segment says an earlier state ended.
10. Use an empty memories array if the segment has no durable grounded information.

Allowed temporal.granularity values: instant, day, week, month, year, interval, none.
Allowed temporal.certainty values: exact, resolved, approximate, unknown.
"""

REQUIRED_MEMORY_KEYS = {
    "memory_type",
    "subject",
    "facet",
    "attribute",
    "value",
    "statement",
    "polarity",
    "temporal",
    "state_status",
    "explicitness",
    "evidence_ids",
    "confidence",
}
REQUIRED_TEMPORAL_KEYS = {
    "source_expression",
    "start",
    "end",
    "granularity",
    "certainty",
}
MEMORY_TYPES = {
    "fact",
    "event",
    "preference",
    "goal",
    "relationship",
    "state",
    "state_change",
    "constraint",
}
MEMORY_TYPE_ALIASES = {
    "activity": "event",
    "experience": "event",
    "life_event": "event",
    "plan": "goal",
    "routine": "state",
    "profile_fact": "fact",
    "attribute": "fact",
    "belief": "preference",
    "career": "fact",
    "education": "fact",
    "family": "relationship",
    "health": "state",
    "identity": "fact",
    "location": "fact",
    "possession": "fact",
    "other": "fact",
}
FACETS = {
    "identity", "family", "relationship", "location", "education", "career", "health",
    "preference", "activity", "possession", "plan", "life_event", "belief", "constraint", "other",
}
POLARITIES = {"positive", "negative"}
STATE_STATUSES = {"stable", "current", "historical", "planned", "superseded", "uncertain"}
EXPLICITNESS = {"direct", "strict_entailment"}
GRANULARITIES = {"instant", "day", "week", "month", "year", "interval", "none"}
CERTAINTIES = {"exact", "resolved", "approximate", "unknown"}
ATTRIBUTE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_key_line(path: Path, line_number: int) -> str:
    if line_number < 1:
        raise ValueError("key line must be at least 1")
    lines = path.read_text(encoding="utf-8").splitlines()
    if line_number > len(lines) or not lines[line_number - 1].strip():
        raise ValueError(f"API key file has no non-empty line {line_number}")
    return lines[line_number - 1].strip()


def session_number(key: str) -> int:
    return int(key.removeprefix("session_"))


def locomo_jobs(path: Path) -> list[dict[str, Any]]:
    conversations = json.loads(path.read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = []
    for conversation in sorted(conversations, key=lambda row: str(row["sample_id"])):
        body = conversation["conversation"]
        session_keys = sorted(
            (
                key
                for key, value in body.items()
                if key.startswith("session_")
                and not key.endswith("_date_time")
                and isinstance(value, list)
            ),
            key=session_number,
        )
        for session_key in session_keys:
            timestamp = str(body.get(f"{session_key}_date_time", "")).strip()
            lines = [f"[SESSION_TIME] {timestamp}"]
            evidence_ids: list[str] = []
            for turn in body[session_key]:
                evidence_id = str(turn["dia_id"])
                evidence_ids.append(evidence_id)
                line = f"[{evidence_id}] {str(turn['speaker']).strip()}: {str(turn['text']).strip()}"
                caption = str(turn.get("blip_caption", "")).strip()
                if caption:
                    line += f" [Shared image: {caption}]"
                lines.append(line)
            jobs.append(
                {
                    "job_id": f"locomo-{conversation['sample_id']}-{session_key}",
                    "context_id": str(conversation["sample_id"]),
                    "segment_id": session_key,
                    "session_time": timestamp,
                    "evidence_ids": evidence_ids,
                    "input": "\n".join(lines),
                }
            )
    return jobs


def canonical_component(value: str) -> str:
    component = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return component or "unknown"


def validate_memory(payload: object, allowed_evidence: set[str]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) != {"memories"}:
        raise ValueError("response must contain exactly the memories key")
    memories = payload["memories"]
    if not isinstance(memories, list):
        raise ValueError("memories must be an array")
    for index, memory in enumerate(memories):
        if not isinstance(memory, dict) or set(memory) != REQUIRED_MEMORY_KEYS:
            raise ValueError(f"memory {index} has invalid keys")
        raw_memory_type = memory["memory_type"]
        if isinstance(raw_memory_type, str):
            memory["memory_type"] = MEMORY_TYPE_ALIASES.get(raw_memory_type, raw_memory_type)
        if memory["memory_type"] not in MEMORY_TYPES:
            raise ValueError(
                f"memory {index} has invalid memory_type {raw_memory_type!r}"
            )
        if memory["facet"] not in FACETS:
            raise ValueError(f"memory {index} has invalid facet")
        if memory["polarity"] not in POLARITIES:
            raise ValueError(f"memory {index} has invalid polarity")
        if memory["state_status"] not in STATE_STATUSES:
            raise ValueError(f"memory {index} has invalid state_status")
        if memory["explicitness"] not in EXPLICITNESS:
            raise ValueError(f"memory {index} has invalid explicitness")
        for field in ("subject", "attribute", "value", "statement"):
            if not isinstance(memory[field], str) or not memory[field].strip():
                raise ValueError(f"memory {index} has invalid {field}")
        if not ATTRIBUTE_PATTERN.fullmatch(memory["attribute"]):
            raise ValueError(f"memory {index} attribute must be short snake_case")
        temporal = memory["temporal"]
        if not isinstance(temporal, dict) or set(temporal) != REQUIRED_TEMPORAL_KEYS:
            raise ValueError(f"memory {index} has invalid temporal object")
        if temporal["granularity"] not in GRANULARITIES:
            raise ValueError(f"memory {index} has invalid temporal granularity")
        if temporal["certainty"] not in CERTAINTIES:
            raise ValueError(f"memory {index} has invalid temporal certainty")
        if not all(isinstance(temporal[field], str) for field in REQUIRED_TEMPORAL_KEYS):
            raise ValueError(f"memory {index} has non-string temporal field")
        for field in ("start", "end"):
            value = temporal[field]
            if value and not re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", value):
                raise ValueError(f"memory {index} has invalid temporal {field}")
        if temporal["granularity"] == "none" and (temporal["start"] or temporal["end"]):
            raise ValueError(f"memory {index} has dates with temporal granularity none")
        evidence_ids = memory["evidence_ids"]
        if not isinstance(evidence_ids, list) or not 1 <= len(evidence_ids) <= 4:
            raise ValueError(f"memory {index} has invalid evidence_ids")
        invalid = set(evidence_ids) - allowed_evidence
        if invalid:
            raise ValueError(f"memory {index} invented evidence IDs: {sorted(invalid)}")
        confidence = memory["confidence"]
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            raise ValueError(f"memory {index} has invalid confidence")
        memory["memory_key"] = "::".join(
            (
                canonical_component(memory["subject"]),
                memory["facet"],
                memory["attribute"],
            )
        )
    return memories


def request_memories(
    *,
    client: httpx.Client,
    api_key: str,
    model: str,
    job: dict[str, Any],
    max_tokens: int,
    retries: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": job["input"]},
        ],
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.post(GLM_ENDPOINT, headers=headers, json=body)
            response.raise_for_status()
            raw = response.json()
            if raw['choices'][0].get('finish_reason') != 'stop':
                raise ValueError('non-stop/truncated API response')
            content = raw["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            memories = validate_memory(parsed, set(job["evidence_ids"]))
            return memories, raw
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as error:
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
    parser.add_argument("--locomo", type=Path, default=Path("datasets/locomo/locomo10.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=Path("secrets/zhipu_api_key"))
    parser.add_argument("--api-key-line", type=int, default=4)
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument(
        "--jobs-per-conversation",
        type=int,
        help="select the first N sessions from every conversation",
    )
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = locomo_jobs(args.locomo)
    if args.max_jobs is not None and args.jobs_per_conversation is not None:
        raise ValueError("use only one of --max-jobs and --jobs-per-conversation")
    if args.jobs_per_conversation is not None:
        if args.jobs_per_conversation < 1:
            raise ValueError("jobs per conversation must be at least 1")
        counts: dict[str, int] = {}
        selected = []
        for job in jobs:
            context_id = job["context_id"]
            if counts.get(context_id, 0) < args.jobs_per_conversation:
                selected.append(job)
                counts[context_id] = counts.get(context_id, 0) + 1
    else:
        selected = jobs[: args.max_jobs] if args.max_jobs is not None else jobs
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "jobs_total": len(jobs),
                "jobs_selected": len(selected),
                "input_characters": sum(len(job["input"]) for job in selected),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    api_key = read_key_line(args.api_key_file, args.api_key_line)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            completed = {
                row["job_id"]: row
                for line in handle
                if line.strip()
                for row in [json.loads(line)]
            }

    with httpx.Client(timeout=args.timeout) as client:
        for ordinal, job in enumerate(selected, 1):
            if job["job_id"] in completed:
                continue
            memories, raw = request_memories(
                client=client,
                api_key=api_key,
                model=args.model,
                job=job,
                max_tokens=args.max_tokens,
                retries=args.retries,
            )
            record = {
                "job_id": job["job_id"],
                "context_id": job["context_id"],
                "segment_id": job["segment_id"],
                "session_time": job["session_time"],
                "schema_version": SCHEMA_VERSION,
                "model": raw.get("model", args.model),
                "prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode()),
                "input_sha256": sha256_bytes(job["input"].encode()),
                "source_evidence_count": len(job["evidence_ids"]),
                "memories": memories,
                "request_id": raw.get("id"),
                "finish_reason": raw.get("choices", [{}])[0].get("finish_reason"),
                "usage": raw.get("usage", {}),
                "completed_unix_time": time.time(),
            }
            completed[job["job_id"]] = record
            ordered = [completed[item["job_id"]] for item in selected if item["job_id"] in completed]
            atomic_write_jsonl(args.output, ordered)
            print(
                json.dumps(
                    {
                        "completed": ordinal,
                        "selected": len(selected),
                        "job_id": job["job_id"],
                        "memories": len(memories),
                        "usage": raw.get("usage", {}),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "endpoint_host": "open.bigmodel.cn",
        "api_key_line": args.api_key_line,
        "question_answer_fields_sent_to_api": False,
        "prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode()),
        "jobs_total": len(jobs),
        "jobs_selected": len(selected),
        "jobs_completed": len(completed),
        "source": {str(args.locomo): sha256_file(args.locomo)},
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
