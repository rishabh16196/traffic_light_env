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

from .models import NUM_DIRECTIONS, NUM_LANES, TrafficLightAction, TrafficLightObservation


class TrafficLightEnv(
    EnvClient[TrafficLightAction, TrafficLightObservation, State]
):
    """
    Client for the Traffic Light Environment.

    Controls a single 4-way intersection traffic light via WebSocket.
    Observes per-direction vehicle counts at 100 m and 500 m (4 directions,
    2 lanes each), plus per-direction light states.

    Use reset(task="task_name") to select a scenario:
        balanced, rush_hour_ns, rush_hour_ew, alternating_surge,
        random_spikes, gridlock, emergency_vehicle, or "random".

    Example:
        >>> async with TrafficLightEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset(task="emergency_vehicle")
        ...     print(f"Task: {result.observation.task_name}")
        ...
        ...     result = await client.step(TrafficLightAction(phase=0))
        ...     print(f"NS 100m: {result.observation.ns_100m}")
    """

    def _step_payload(self, action: TrafficLightAction) -> Dict:
        return {
            "phase": action.phase,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TrafficLightObservation]:
        obs_data = payload.get("observation", {})
        observation = TrafficLightObservation(
            task_name=obs_data.get("task_name", "balanced"),
            # Per-direction 100 m
            ns_100m=obs_data.get("ns_100m", 0),
            sn_100m=obs_data.get("sn_100m", 0),
            ew_100m=obs_data.get("ew_100m", 0),
            we_100m=obs_data.get("we_100m", 0),
            # Per-direction 500 m
            ns_500m=obs_data.get("ns_500m", 0),
            sn_500m=obs_data.get("sn_500m", 0),
            ew_500m=obs_data.get("ew_500m", 0),
            we_500m=obs_data.get("we_500m", 0),
            # Lights
            light_ns=obs_data.get("light_ns", 0),
            light_sn=obs_data.get("light_sn", 0),
            light_ew=obs_data.get("light_ew", 0),
            light_we=obs_data.get("light_we", 0),
            # Emergency
            emergency_direction=obs_data.get("emergency_direction", -1),
            emergency_lane=obs_data.get("emergency_lane", -1),
            emergency_wait=obs_data.get("emergency_wait", 0),
            # Phase / timing
            active_phase=obs_data.get("active_phase", 0),
            yellow_remaining=obs_data.get("yellow_remaining", 0),
            time_in_phase=obs_data.get("time_in_phase", 0),
            step_number=obs_data.get("step_number", 0),
            # Aggregates
            total_waiting=obs_data.get("total_waiting", 0),
            total_throughput=obs_data.get("total_throughput", 0),
            arrivals=obs_data.get("arrivals", [0] * NUM_DIRECTIONS),
            departures=obs_data.get("departures", [0] * NUM_DIRECTIONS),
            # Per-lane detail
            lanes_100m=obs_data.get("lanes_100m", [0] * NUM_LANES),
            lanes_500m=obs_data.get("lanes_500m", [0] * NUM_LANES),
            # Grading
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
