# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Traffic Light Environment.

A single intersection with 4 lanes (North, South, East, West).
Each lane has two observation zones: 100m and 500m from the intersection.
Traffic lights have 3 states: red (0), yellow (1), green (2).

Supports multiple tasks/scenarios selected at reset time.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# Light state constants
LIGHT_RED = 0
LIGHT_YELLOW = 1
LIGHT_GREEN = 2

# Available task names
TASK_NAMES = [
    "balanced",
    "rush_hour_ns",
    "rush_hour_ew",
    "alternating_surge",
    "random_spikes",
    "gridlock",
    "emergency_vehicle",
]


class TrafficLightAction(Action):
    """Action: choose the desired green direction.

    phase: 0 = North-South green (East-West red)
           1 = East-West green (North-South red)

    If this differs from the current active direction, a mandatory yellow
    transition period is triggered before the new direction turns green.
    """

    phase: int = Field(
        ...,
        description="Desired green direction: 0=NS-green/EW-red, 1=EW-green/NS-red",
        ge=0,
        le=1,
    )


class TrafficLightObservation(Observation):
    """Observation from the intersection with 100m and 500m views per lane."""

    # Task info
    task_name: str = Field(default="balanced", description="Current task/scenario name")

    # 100m zone — vehicles near the intersection, ready to depart on green
    north_100m: int = Field(default=0, description="Vehicles within 100m in North lane")
    south_100m: int = Field(default=0, description="Vehicles within 100m in South lane")
    east_100m: int = Field(default=0, description="Vehicles within 100m in East lane")
    west_100m: int = Field(default=0, description="Vehicles within 100m in West lane")

    # 500m zone — vehicles approaching, between 100m and 500m
    north_500m: int = Field(default=0, description="Vehicles between 100-500m in North lane")
    south_500m: int = Field(default=0, description="Vehicles between 100-500m in South lane")
    east_500m: int = Field(default=0, description="Vehicles between 100-500m in East lane")
    west_500m: int = Field(default=0, description="Vehicles between 100-500m in West lane")

    # Per-lane light state: 0=red, 1=yellow, 2=green
    light_north: int = Field(default=0, description="North lane light: 0=red, 1=yellow, 2=green")
    light_south: int = Field(default=0, description="South lane light: 0=red, 1=yellow, 2=green")
    light_east: int = Field(default=0, description="East lane light: 0=red, 1=yellow, 2=green")
    light_west: int = Field(default=0, description="West lane light: 0=red, 1=yellow, 2=green")

    # Emergency vehicle info
    emergency_lane: int = Field(
        default=-1,
        description="Lane with emergency vehicle: 0=N, 1=S, 2=E, 3=W, -1=none",
    )
    emergency_wait: int = Field(
        default=0, description="Steps the emergency vehicle has been waiting"
    )

    # Phase and timing
    active_phase: int = Field(
        default=0,
        description="Active green direction: 0=NS, 1=EW (-1 during yellow transition)",
    )
    yellow_remaining: int = Field(
        default=0, description="Steps remaining in yellow transition (0 if not transitioning)"
    )
    time_in_phase: int = Field(default=0, description="Steps since last phase change completed")
    step_number: int = Field(default=0, description="Current simulation step")

    # Aggregate stats
    total_waiting: int = Field(
        default=0, description="Total vehicles across all lanes and zones"
    )
    total_throughput: int = Field(
        default=0, description="Cumulative vehicles that have passed through"
    )
    arrivals: List[int] = Field(
        default_factory=lambda: [0, 0, 0, 0],
        description="Vehicles that arrived this step (500m zone) [N, S, E, W]",
    )
    departures: List[int] = Field(
        default_factory=lambda: [0, 0, 0, 0],
        description="Vehicles that departed this step (from 100m zone) [N, S, E, W]",
    )

    # Grading (populated on final step when done=True)
    grade_score: Optional[float] = Field(
        default=None,
        description="Final grade 0.0-1.0 (only set on terminal observation when done=True)",
    )
    grade_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Breakdown of grading components (only set on terminal observation when done=True)",
    )
