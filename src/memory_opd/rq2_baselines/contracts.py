"""Fail-closed configuration checks for the two RQ2 baseline methods."""

from __future__ import annotations

from typing import Any, Mapping


def validate_baseline_config(config: Mapping[str, Any]) -> None:
    method = config.get("method")
    if method not in {"vanilla_grpo", "opsd"}:
        raise ValueError("method must be vanilla_grpo or opsd")
    if config.get("backbone") != "Qwen/Qwen2.5-3B-Instruct":
        raise ValueError("RQ2 primary baseline backbone must be Qwen2.5-3B-Instruct")
    evaluation = config.get("evaluation", {})
    if evaluation.get("decoding") != "greedy" or evaluation.get("max_new_tokens") != 5:
        raise ValueError("RQ2 evaluation must use greedy decoding with max_new_tokens=5")
    training = config.get("training", {})
    if training.get("rollout_max_new_tokens") != 5:
        raise ValueError("PersonaMem baseline rollouts must be capped at 5 tokens")
    if method == "vanilla_grpo":
        if training.get("adv_estimator") != "grpo":
            raise ValueError("Vanilla GRPO must use adv_estimator=grpo")
        if training.get("task_reward") != "strict_mc_exact_match":
            raise ValueError("Vanilla GRPO must use strict MC exact-match reward")
    else:
        required = {
            "teacher": "fixed",
            "divergence": "jsd",
            "jsd_beta": 0.0,
            "pointwise_clip": 0.05,
            "use_task_reward": False,
        }
        mismatches = {key: (training.get(key), value) for key, value in required.items() if training.get(key) != value}
        if mismatches:
            raise ValueError(f"OPSD contract mismatch: {mismatches}")
