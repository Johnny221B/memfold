"""Replay ALFWorld games using the official TextWorld planner expert."""

from __future__ import annotations

import traceback
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import textworld

from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

from .inventory import GameRecord


@dataclass(frozen=True)
class ReplayResult:
    """Serializable outcome of replaying one game."""

    trajectory: dict[str, Any] | None
    failure: dict[str, Any] | None

    @property
    def success(self) -> bool:
        return self.trajectory is not None and bool(self.trajectory["success"])


def _to_plain_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(item) for item in value]


def replay_game(record: GameRecord, data_root: Path, max_steps: int = 50) -> ReplayResult:
    """Replay a game to completion, retaining failure context instead of raising."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    gamefile = data_root.resolve() / record.relative_gamefile
    try:
        game_data = json.loads(gamefile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read game walkthrough from {gamefile}: {exc}") from exc
    walkthrough = _to_plain_list(game_data.get("walkthrough"))
    if not walkthrough:
        raise ValueError(f"Game has no oracle walkthrough: {gamefile}")

    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        score=True,
        max_score=True,
    )
    env = None
    steps: list[dict[str, Any]] = []
    try:
        env = textworld.start(
            str(gamefile),
            request_infos=request_infos,
            wrappers=[AlfredDemangler()],
        )
        state = env.reset()
        observation = str(state.feedback)
        done = False

        for step_index, action in enumerate(walkthrough[:max_steps]):
            admissible = _to_plain_list(state.get("admissible_commands"))
            if action not in admissible:
                raise RuntimeError(
                    f"walkthrough action is not admissible at step {step_index}: {action!r}"
                )

            next_state, reward, done = env.step(action)
            next_observation = str(next_state.feedback)
            steps.append(
                {
                    "t": step_index,
                    "observation": observation,
                    "admissible_actions": admissible,
                    "action": action,
                    "next_observation": next_observation,
                    "reward": float(reward),
                    "done": bool(done),
                }
            )
            state = next_state
            observation = next_observation
            if done:
                break

        won = bool(state.get("won", False))
        trajectory = {
            "schema_version": "1.0",
            "trajectory_id": record.trajectory_id,
            "split": "train",
            "task_id": record.task_id,
            "task_type": record.task_type,
            "objective": record.objective,
            "success": won,
            "steps": steps,
            "metadata": {
                "source": "official_alfworld_planner_replay",
                "expert_source": "game.tw-pddl:walkthrough",
                "gamefile": record.relative_gamefile,
                "traj_data": record.relative_traj_data,
                "max_steps": max_steps,
                "terminated": bool(done),
                "walkthrough_steps": len(walkthrough),
            },
        }
        if not won:
            return ReplayResult(
                trajectory=trajectory,
                failure={
                    "trajectory_id": record.trajectory_id,
                    "kind": "not_won",
                    "message": f"episode ended without success after {len(steps)} steps",
                },
            )
        return ReplayResult(trajectory=trajectory, failure=None)
    except Exception as exc:
        return ReplayResult(
            trajectory=None,
            failure={
                "trajectory_id": record.trajectory_id,
                "gamefile": record.relative_gamefile,
                "kind": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "steps_completed": len(steps),
            },
        )
    finally:
        if env is not None:
            env.close()
