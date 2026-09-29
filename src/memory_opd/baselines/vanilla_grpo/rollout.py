"""Data-only closed-loop rollout records and reward semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Decision:
    prompt_ids: list[int]
    completion_ids: list[int]
    allowed_token_ids: list[list[int]]
    behavior_logps: list[float]
    raw_generation: str
    parsed_action: str
    executed_action: str
    raw_exact_valid: bool
    snapping_distance: float


@dataclass
class Episode:
    trajectory_id: str
    rollout_seed: int
    decisions: list[Decision] = field(default_factory=list)
    cumulative_reward: float = 0.0
    environment_done: bool = False

    @property
    def reward(self) -> float:
        return float(self.cumulative_reward >= 1.0)

    def to_log(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "rollout_seed": self.rollout_seed,
            "reward": self.reward,
            "cumulative_environment_reward": self.cumulative_reward,
            "environment_done": self.environment_done,
            "step_count": len(self.decisions),
            "decisions": [asdict(item) for item in self.decisions],
        }


def validate_group(episodes: list[Episode], expected_size: int) -> None:
    if len(episodes) != expected_size:
        raise ValueError(f"expected {expected_size} episodes, got {len(episodes)}")
    if expected_size < 2:
        raise ValueError("GRPO group size must be at least two")
    identities = {episode.trajectory_id for episode in episodes}
    if len(identities) != 1:
        raise ValueError("all group members must start from the same ALFWorld game")
    seeds = [episode.rollout_seed for episode in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError("rollout seeds must be unique within a group")
