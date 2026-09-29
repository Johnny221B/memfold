#!/usr/bin/env python3
"""Build audited extraction-only PersonaMem SFT JSONL from GLM requests/responses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from memory_opd.writer.training import build_sft_records


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-target", action="store_true")
    parser.add_argument("--maximum-stable", type=int, default=14)
    parser.add_argument("--maximum-current", type=int, default=18)
    parser.add_argument("--maximum-changes", type=int, default=8)
    parser.add_argument("--maximum-fact-characters", type=int, default=176)
    parser.add_argument("--maximum-change-field-characters", type=int, default=144)
    args = parser.parse_args()
    records = build_sft_records(
        load(args.requests),
        load(args.responses),
        compact_target=args.compact_target,
        maximum_stable=args.maximum_stable,
        maximum_current=args.maximum_current,
        maximum_changes=args.maximum_changes,
        maximum_fact_characters=args.maximum_fact_characters,
        maximum_change_field_characters=args.maximum_change_field_characters,
    )
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps({"records": len(records), "sha256": hashlib.sha256(payload.encode()).hexdigest(), "output": str(args.output)}))


if __name__ == "__main__":
    main()
