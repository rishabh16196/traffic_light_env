# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Grading rubrics for the Traffic Light Environment.

Each task has specific grading criteria based on:
- Average waiting vehicles per step (lower is better)
- Total throughput (higher is better)
- Emergency vehicle clearance speed (emergency_vehicle task only)

Scores are normalized to 0.0-1.0 using per-task thresholds that
reflect increasing difficulty across tasks.
"""

from typing import Any, Dict, List, Tuple

from openenv.core.rubrics.trajectory import TrajectoryRubric


def _linear_score(value: float, perfect: float, fail: float) -> float:
    """Linearly interpolate a score between 1.0 (at perfect) and 0.0 (at fail).

    Works for both "lower is better" (perfect < fail) and "higher is better"
    (perfect > fail) metrics.
    """
    if perfect == fail:
        return 1.0 if value == perfect else 0.0
    score = (value - fail) / (perfect - fail)
    return max(0.0, min(1.0, score))


# Per-task grading thresholds.
# avg_waiting: (perfect, fail) — average total_waiting per step; lower is better.
# throughput:  (perfect, fail) — total vehicles cleared; higher is better.
TASK_THRESHOLDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "balanced": {
        "avg_waiting": (15.0, 70.0),
        "throughput": (700.0, 250.0),
    },
    "rush_hour_ns": {
        "avg_waiting": (15.0, 60.0),
        "throughput": (800.0, 300.0),
    },
    "rush_hour_ew": {
        "avg_waiting": (15.0, 60.0),
        "throughput": (800.0, 300.0),
    },
    "alternating_surge": {
        "avg_waiting": (30.0, 120.0),
        "throughput": (800.0, 300.0),
    },
    "random_spikes": {
        "avg_waiting": (15.0, 60.0),
        "throughput": (600.0, 200.0),
    },
    "gridlock": {
        "avg_waiting": (100.0, 500.0),
        "throughput": (800.0, 350.0),
    },
    "emergency_vehicle": {
        "avg_waiting": (15.0, 70.0),
        "throughput": (700.0, 250.0),
    },
}

# Emergency clearance grading: steps waited → score component
EMERGENCY_CLEARANCE_THRESHOLDS: List[Tuple[int, float]] = [
    (3, 1.0),
    (8, 0.7),
    (15, 0.4),
    (30, 0.1),
]
EMERGENCY_NOT_CLEARED_SCORE = 0.0


class TrafficLightRubric(TrajectoryRubric):
    """Grades agent performance on each traffic light task.

    Accumulates (action, observation) pairs over the episode and computes
    a 0.0-1.0 score at episode end based on task-specific criteria:

    - **Waiting score** (weight 0.40): How well the agent minimized average
      queue length across the episode.
    - **Throughput score** (weight 0.40): How many vehicles the agent cleared.
    - **Safety score** (weight 0.20): How well the agent avoided dilemma-zone
      incidents (vehicles that can't stop safely on phase switch).
    - For **emergency_vehicle** task, the weights shift to
      0.25 / 0.20 / 0.15 / 0.40 to account for emergency clearance speed.
    """

    def __init__(self) -> None:
        super().__init__(intermediate_reward=0.0)

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """Compute final grade from the full episode trajectory."""
        if not trajectory:
            return 0.0

        _, final_obs = trajectory[-1]
        task_name = getattr(final_obs, "task_name", "balanced")
        thresholds = TASK_THRESHOLDS.get(task_name, TASK_THRESHOLDS["balanced"])

        # --- Compute metrics ---
        waiting_sum = 0.0
        for _, obs in trajectory:
            waiting_sum += getattr(obs, "total_waiting", 0)
        avg_waiting = waiting_sum / len(trajectory)

        total_throughput = float(getattr(final_obs, "total_throughput", 0))
        total_dilemma = float(getattr(final_obs, "total_dilemma_vehicles", 0.0))

        # --- Score components ---
        perf_w, fail_w = thresholds["avg_waiting"]
        waiting_score = _linear_score(avg_waiting, perf_w, fail_w)

        perf_t, fail_t = thresholds["throughput"]
        throughput_score = _linear_score(total_throughput, perf_t, fail_t)

        # Safety: fewer dilemma vehicles = better
        # Perfect: 0 dilemma vehicles; Fail: 50+ dilemma vehicles
        safety_score = _linear_score(total_dilemma, 0.0, 50.0)

        # --- Task-specific weighting ---
        if task_name == "emergency_vehicle":
            emergency_score = self._grade_emergency(trajectory)
            final_score = (
                0.25 * waiting_score
                + 0.20 * throughput_score
                + 0.15 * safety_score
                + 0.40 * emergency_score
            )
            self._grade_details = {
                "task": task_name,
                "score": round(final_score, 4),
                "waiting_score": round(waiting_score, 4),
                "throughput_score": round(throughput_score, 4),
                "safety_score": round(safety_score, 4),
                "emergency_score": round(emergency_score, 4),
                "avg_waiting": round(avg_waiting, 2),
                "total_throughput": int(total_throughput),
                "total_dilemma_vehicles": round(total_dilemma, 2),
                "passed": final_score >= 0.5,
            }
        else:
            final_score = (
                0.40 * waiting_score
                + 0.40 * throughput_score
                + 0.20 * safety_score
            )
            self._grade_details = {
                "task": task_name,
                "score": round(final_score, 4),
                "waiting_score": round(waiting_score, 4),
                "throughput_score": round(throughput_score, 4),
                "safety_score": round(safety_score, 4),
                "avg_waiting": round(avg_waiting, 2),
                "total_throughput": int(total_throughput),
                "total_dilemma_vehicles": round(total_dilemma, 2),
                "passed": final_score >= 0.5,
            }

        return final_score

    def compute_step_rewards(self) -> List[float]:
        """Uniform credit assignment — every step shares the final score equally."""
        if not self._trajectory:
            return []
        score = self.score_trajectory(self._trajectory)
        return [score] * len(self._trajectory)

    @property
    def grade_details(self) -> Dict[str, Any]:
        """Detailed breakdown of the last grading result."""
        return getattr(self, "_grade_details", {})

    def reset(self) -> None:
        """Clear trajectory and grade details on episode reset."""
        super().reset()
        self._grade_details = {}

    def _grade_emergency(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """Score emergency vehicle clearance speed."""
        emergency_appeared = False
        emergency_wait_steps = 0
        emergency_cleared = False

        for _, obs in trajectory:
            e_dir = getattr(obs, "emergency_direction", -1)
            e_wait = getattr(obs, "emergency_wait", 0)

            if e_dir >= 0:
                emergency_appeared = True
                emergency_wait_steps = e_wait
            elif emergency_appeared and not emergency_cleared:
                emergency_cleared = True

        if not emergency_appeared:
            return 1.0

        if not emergency_cleared:
            return EMERGENCY_NOT_CLEARED_SCORE

        for max_wait, score in EMERGENCY_CLEARANCE_THRESHOLDS:
            if emergency_wait_steps <= max_wait:
                return score
        return 0.05
