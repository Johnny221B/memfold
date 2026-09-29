"""Validated expansion of the 2 x 3 x 4 OPSD/GRPO training matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPECTED_METHODS = ("opsd", "grpo")
EXPECTED_BACKBONES = ("Qwen3-4B", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct")
EXPECTED_DATASETS = ("personamem32k", "personamem128k", "longmemeval", "lampqa")


@dataclass(frozen=True)
class TrainingCell:
    method: str
    backbone: str
    dataset: str
    train: Path
    validation: Path
    max_prompt_length: int
    max_completion_length: int
    reward_type: str
    model_variant: str
    record_dataset: str | None


def load_matrix(config_path: Path, data_root: Path) -> tuple[dict[str, Any], tuple[TrainingCell, ...]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    matrix = config.get("matrix", {})
    methods = tuple(matrix.get("methods", ()))
    backbones = tuple(matrix.get("backbones", ()))
    datasets: Mapping[str, Mapping[str, Any]] = matrix.get("datasets", {})
    if methods != EXPECTED_METHODS:
        raise ValueError(f"matrix methods must be {EXPECTED_METHODS}")
    if backbones != EXPECTED_BACKBONES:
        raise ValueError(f"matrix backbones must be {EXPECTED_BACKBONES}")
    if tuple(datasets) != EXPECTED_DATASETS:
        raise ValueError(f"matrix datasets must be {EXPECTED_DATASETS}")
    if datasets["longmemeval"].get("held_out_test_enters_training") is not False:
        raise ValueError("LongMemEval held-out test must not enter training")
    expected_rewards = {
        "personamem32k": "strict_choice",
        "personamem128k": "strict_choice",
        "longmemeval": "normalized_token_f1",
        "lampqa": "normalized_token_f1",
    }
    if {name: row.get("reward_type") for name, row in datasets.items()} != expected_rewards:
        raise ValueError("dataset reward contracts do not match the registered matrix")
    if datasets["personamem128k"].get("model_variant") != "long_context":
        raise ValueError("PersonaMem-128K must use a long-context model view")
    opsd = config.get("training", {}).get("opsd", {})
    required_opsd = {"fixed_teacher": True, "jsd_beta": 0.0, "pointwise_clip": 0.05, "use_task_reward": False}
    if any(opsd.get(key) != value for key, value in required_opsd.items()):
        raise ValueError("OPSD method contract does not match the pinned baseline")
    if config.get("training", {}).get("grpo", {}).get("beta") != 0.0:
        raise ValueError("Vanilla GRPO beta must be zero")

    cells: list[TrainingCell] = []
    for method in methods:
        for backbone in backbones:
            for dataset, settings in datasets.items():
                cells.append(
                    TrainingCell(
                        method=method,
                        backbone=backbone,
                        dataset=dataset,
                        train=data_root / str(settings["train"]),
                        validation=data_root / str(settings["validation"]),
                        max_prompt_length=int(settings["max_prompt_length"]),
                        max_completion_length=int(settings["max_completion_length"]),
                        reward_type=str(settings["reward_type"]),
                        model_variant=str(settings.get("model_variant", "native")),
                        record_dataset=settings.get("record_dataset"),
                    )
                )
    if len(cells) != 24 or len(set(cells)) != 24:
        raise AssertionError("the OPSD/GRPO training matrix must contain 24 unique cells")
    return config, tuple(cells)
