# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Traffic Light Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TrafficLightAction, TrafficLightObservation


class TrafficLightEnv(
    EnvClient[TrafficLightAction, TrafficLightObservation, State]
):
    """
    Client for the Traffic Light Environment.

    Controls a single 4-way intersection traffic light via WebSocket.
    Observes vehicle counts at 100m and 500m per lane, plus per-lane
    light states (red/yellow/green).

    Use reset(task="task_name") to select a scenario:
        balanced, rush_hour_ns, rush_hour_ew, alternating_surge,
        random_spikes, gridlock, emergency_vehicle, or "random".

    Example:
        >>> with TrafficLightEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task="emergency_vehicle")
        ...     print(f"Task: {result.observation.task_name}")
        ...
        ...     result = client.step(TrafficLightAction(phase=1))
        ...     print(f"Emergency lane: {result.observation.emergency_lane}")
    """

    def _step_payload(self, action: TrafficLightAction) -> Dict:
        return {
            "phase": action.phase,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TrafficLightObservation]:
        obs_data = payload.get("observation", {})
        observation = TrafficLightObservation(
            task_name=obs_data.get("task_name", "balanced"),
            north_100m=obs_data.get("north_100m", 0),
            south_100m=obs_data.get("south_100m", 0),
            east_100m=obs_data.get("east_100m", 0),
            west_100m=obs_data.get("west_100m", 0),
            north_500m=obs_data.get("north_500m", 0),
            south_500m=obs_data.get("south_500m", 0),
            east_500m=obs_data.get("east_500m", 0),
            west_500m=obs_data.get("west_500m", 0),
            light_north=obs_data.get("light_north", 0),
            light_south=obs_data.get("light_south", 0),
            light_east=obs_data.get("light_east", 0),
            light_west=obs_data.get("light_west", 0),
            emergency_lane=obs_data.get("emergency_lane", -1),
            emergency_wait=obs_data.get("emergency_wait", 0),
            active_phase=obs_data.get("active_phase", 0),
            yellow_remaining=obs_data.get("yellow_remaining", 0),
            time_in_phase=obs_data.get("time_in_phase", 0),
            step_number=obs_data.get("step_number", 0),
            total_waiting=obs_data.get("total_waiting", 0),
            total_throughput=obs_data.get("total_throughput", 0),
            arrivals=obs_data.get("arrivals", [0, 0, 0, 0]),
            departures=obs_data.get("departures", [0, 0, 0, 0]),
            grade_score=obs_data.get("grade_score"),
            grade_details=obs_data.get("grade_details"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
