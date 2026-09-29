"""Leakage-safe adapters for the four-dataset OPSD/GRPO training matrix."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from memory_opd.mem_r1.datasets import load_locomo


REWARD_TYPES = frozenset({"strict_choice", "normalized_token_f1"})
LAMP_SYSTEM = (
    "Answer the current question in a personalized way using the user's past posts. "
    "Return only the personalized answer, without commentary about the profile."
)
LOCOMO_SYSTEM = (
    "Answer the question using only the supplied conversation history. "
    "Return only the concise answer."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_record(record: Mapping[str, Any], *, expected_split: str | None = None) -> None:
    required = ("dataset", "split", "prompt", "answer", "question_id", "reward_type")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"training record lacks fields: {missing}")
    if expected_split is not None and record["split"] != expected_split:
        raise ValueError(f"expected split {expected_split!r}, found {record['split']!r}")
    if record["split"] not in {"train", "validation"}:
        raise ValueError("only train/validation records may enter the trainer")
    if record["reward_type"] not in REWARD_TYPES:
        raise ValueError(f"unsupported reward_type: {record['reward_type']!r}")
    if not all(str(record[key]).strip() for key in ("prompt", "answer", "question_id")):
        raise ValueError("prompt, answer, and question_id must be non-empty")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    rows = list(records)
    for row in rows:
        validate_record(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def _lamp_profile(profile: Sequence[Mapping[str, Any]]) -> str:
    return "\n\n".join(
        f"[Past post {index}]\n{str(item.get('text', '')).strip()}"
        for index, item in enumerate(profile, start=1)
    )


def _lamp_record(row: Mapping[str, Any], split: str, source: str) -> dict[str, Any]:
    answer = row.get("narrative", row.get("target"))
    if answer is None:
        raise ValueError(f"LaMP-QA {split} row {row.get('id')!r} has no narrative target")
    profile = row.get("profile")
    if not isinstance(profile, list):
        raise ValueError(f"LaMP-QA row {row.get('id')!r} has invalid profile")
    prompt = (
        f"{LAMP_SYSTEM}\n\n# Past posts\n{_lamp_profile(profile)}\n\n"
        f"# Current question\n{str(row.get('question', '')).strip()}\n\n# Personalized answer\n"
    )
    return {
        "dataset": "lampqa",
        "split": split,
        "prompt": prompt,
        "answer": str(answer).strip(),
        "question_id": f"{Path(source).stem}:{row['id']}",
        "reward_type": "normalized_token_f1",
        "metadata": {"category": row.get("category"), "source_file": source},
    }


def prepare_lampqa(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Combine domain files without ever copying test labels into trainer JSONL."""

    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    sources: dict[str, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("*.json")):
        match = re.search(r"_(train|validation|test)$", path.stem)
        if match is None:
            continue
        split = match.group(1)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{path} must contain a JSON list")
        sources[path.name] = {"sha256": sha256(path), "rows": len(raw), "split": split}
        if split == "test":
            continue
        grouped[split].extend(_lamp_record(row, split, path.name) for row in raw)
    if not grouped["train"] or not grouped["validation"]:
        raise ValueError("LaMP-QA requires non-empty train and validation domain files")
    counts = {
        split: write_jsonl(output_dir / f"{split}.jsonl", rows)
        for split, rows in grouped.items()
    }
    manifest = {
        "schema_version": "opsd-grpo-qa-v1",
        "dataset": "lampqa",
        "counts": counts,
        "sources": sources,
        "test_label_firewall": True,
        "reward_type": "normalized_token_f1",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _session_number(name: str) -> int:
    match = re.fullmatch(r"session_(\d+)", name)
    return int(match.group(1)) if match else 10**9


def render_locomo_history(conversation: Mapping[str, Any]) -> str:
    body = conversation["conversation"]
    sessions = sorted(
        (
            key
            for key, value in body.items()
            if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
        ),
        key=_session_number,
    )
    rendered: list[str] = []
    for session in sessions:
        timestamp = str(body.get(f"{session}_date_time", "")).strip()
        rendered.append(f"## {session} ({timestamp})")
        for turn in body[session]:
            speaker = str(turn.get("speaker", turn.get("role", "unknown"))).strip()
            text = str(turn.get("text", turn.get("content", ""))).strip()
            rendered.append(f"{speaker}: {text}")
    return "\n".join(rendered)


def _locomo_records(conversations: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for conversation in conversations:
        history = render_locomo_history(conversation)
        sample_id = str(conversation["sample_id"])
        for index, question in enumerate(conversation["qa"]):
            if str(question.get("category")) == "5":
                continue
            prompt = (
                f"{LOCOMO_SYSTEM}\n\n# Conversation history\n{history}\n\n"
                f"# Question\n{str(question['question']).strip()}\n\n# Answer\n"
            )
            records.append(
                {
                    "dataset": "longmemeval_transfer",
                    "split": split,
                    "prompt": prompt,
                    "answer": str(question["answer"]).strip(),
                    "question_id": f"{sample_id}-qa-{index:04d}",
                    "reward_type": "normalized_token_f1",
                    "metadata": {
                        "train_source": "locomo",
                        "sample_id": sample_id,
                        "category": str(question.get("category", "")),
                    },
                }
            )
    return records


def prepare_longmemeval_transfer(
    locomo_path: Path,
    output_dir: Path,
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Prepare LoCoMo-only training data for frozen LongMemEval evaluation."""

    splits = load_locomo(locomo_path, verify_hash=verify_hash)
    counts: dict[str, int] = {}
    conversation_ids: dict[str, list[str]] = {}
    for split in ("train", "validation"):
        selected = splits[split]
        rows = _locomo_records(selected.conversations, split)
        counts[split] = write_jsonl(output_dir / f"{split}.jsonl", rows)
        conversation_ids[split] = [str(item["sample_id"]) for item in selected.conversations]
    if set(conversation_ids["train"]) & set(conversation_ids["validation"]):
        raise AssertionError("LoCoMo train/validation conversation leakage")
    manifest = {
        "schema_version": "opsd-grpo-qa-v1",
        "dataset": "longmemeval_transfer",
        "training_source": "locomo",
        "held_out_evaluation": "longmemeval_s_cleaned_500",
        "longmemeval_enters_training": False,
        "source_sha256": sha256(locomo_path),
        "counts": counts,
        "conversation_ids": conversation_ids,
        "reward_type": "normalized_token_f1",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
