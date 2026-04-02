# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Traffic Light Environment.

A single 4-way intersection with four traffic-flow directions:
  NS (north-to-south), SN (south-to-north),
  EW (east-to-west),   WE (west-to-east).

Each direction has 2 lanes, giving 8 lanes total.
Each lane has two observation zones: 100 m and 500 m from the intersection.
Traffic lights have 3 states per direction: red (0), yellow (1), green (2).

The agent selects one of 6 phases that determine which directions get green.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Light state constants
LIGHT_RED = 0
LIGHT_YELLOW = 1
LIGHT_GREEN = 2

# Direction indices
DIR_NS = 0  # North → South
DIR_SN = 1  # South → North
DIR_EW = 2  # East → West
DIR_WE = 3  # West → East

NUM_DIRECTIONS = 4
LANES_PER_DIRECTION = 2
NUM_LANES = NUM_DIRECTIONS * LANES_PER_DIRECTION  # 8

DIRECTION_NAMES = ["ns", "sn", "ew", "we"]

# Phase definitions — which directions get green
NUM_PHASES = 6
PHASE_NAMES = [
    "ns_sn_corridor",  # 0: full north-south corridor
    "ew_we_corridor",  # 1: full east-west corridor
    "ns_only",         # 2: north-to-south only
    "sn_only",         # 3: south-to-north only
    "ew_only",         # 4: east-to-west only
    "we_only",         # 5: west-to-east only
]

# ---------------------------------------------------------------------------
# Vehicle types with real-world physics
# ---------------------------------------------------------------------------
# Stopping distance = d_reaction + d_braking
#   d_reaction = speed_ms × reaction_time_s
#   d_braking  = speed_ms² / (2 × deceleration_ms2)
# Assumes dry urban road (friction coefficient ≈ 0.7).

VEHICLE_TYPES: Dict[str, Dict[str, float]] = {
    "car": {
        "speed_kmh": 50.0,
        "reaction_time_s": 1.0,
        "deceleration_ms2": 6.8,   # standard passenger car, dry road
        "spawn_weight": 0.40,
    },
    "suv": {
        "speed_kmh": 50.0,
        "reaction_time_s": 1.2,
        "deceleration_ms2": 6.0,   # higher center of gravity, slightly worse braking
        "spawn_weight": 0.25,
    },
    "bus": {
        "speed_kmh": 40.0,
        "reaction_time_s": 1.5,
        "deceleration_ms2": 4.5,   # heavy, pneumatic brakes, longer reaction
        "spawn_weight": 0.10,
    },
    "truck": {
        "speed_kmh": 45.0,
        "reaction_time_s": 1.4,
        "deceleration_ms2": 4.0,   # heaviest, longest stopping distance
        "spawn_weight": 0.15,
    },
    "motorcycle": {
        "speed_kmh": 55.0,
        "reaction_time_s": 0.8,
        "deceleration_ms2": 7.5,   # light, good brakes, alert rider
        "spawn_weight": 0.10,
    },
}

VEHICLE_TYPE_NAMES: List[str] = list(VEHICLE_TYPES.keys())


def stopping_distance(vtype: str) -> float:
    """Compute total stopping distance in metres for a vehicle type."""
    props = VEHICLE_TYPES[vtype]
    v = props["speed_kmh"] / 3.6  # km/h → m/s
    d_react = v * props["reaction_time_s"]
    d_brake = v ** 2 / (2.0 * props["deceleration_ms2"])
    return d_react + d_brake


# Pre-computed stopping distances (metres) and dilemma-zone fractions
# Dilemma fraction = stopping_distance / 100 m (clamped to [0, 1])
# Vehicles within this fraction of the 100 m zone can't stop safely.
STOPPING_DISTANCES: Dict[str, float] = {
    vt: round(stopping_distance(vt), 1) for vt in VEHICLE_TYPE_NAMES
}
DILEMMA_FRACTIONS: Dict[str, float] = {
    vt: min(STOPPING_DISTANCES[vt] / 100.0, 1.0) for vt in VEHICLE_TYPE_NAMES
}

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


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class TrafficLightAction(Action):
    """Action: choose the desired green phase.

    phase:
      0 = NS+SN corridor (both north-south directions green)
      1 = EW+WE corridor (both east-west directions green)
      2 = NS only (north-to-south green)
      3 = SN only (south-to-north green)
      4 = EW only (east-to-west green)
      5 = WE only (west-to-east green)

    Switching to a different phase triggers a mandatory yellow transition
    period before the new phase activates.
    """

    phase: int = Field(
        ...,
        description=(
            "Desired green phase: "
            "0=NS+SN corridor, 1=EW+WE corridor, "
            "2=NS only, 3=SN only, 4=EW only, 5=WE only"
        ),
        ge=0,
        le=NUM_PHASES - 1,
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TrafficLightObservation(Observation):
    """Observation from the intersection with per-direction queue totals."""

    # Task info
    task_name: str = Field(default="balanced", description="Current task/scenario name")

    # Per-direction 100 m zone totals (sum of both lanes in each direction)
    ns_100m: int = Field(default=0, description="Vehicles within 100 m — NS direction (2 lanes)")
    sn_100m: int = Field(default=0, description="Vehicles within 100 m — SN direction (2 lanes)")
    ew_100m: int = Field(default=0, description="Vehicles within 100 m — EW direction (2 lanes)")
    we_100m: int = Field(default=0, description="Vehicles within 100 m — WE direction (2 lanes)")

    # Per-direction 500 m zone totals
    ns_500m: int = Field(default=0, description="Vehicles 100-500 m — NS direction (2 lanes)")
    sn_500m: int = Field(default=0, description="Vehicles 100-500 m — SN direction (2 lanes)")
    ew_500m: int = Field(default=0, description="Vehicles 100-500 m — EW direction (2 lanes)")
    we_500m: int = Field(default=0, description="Vehicles 100-500 m — WE direction (2 lanes)")

    # Per-direction light state: 0=red, 1=yellow, 2=green
    light_ns: int = Field(default=0, description="NS direction light: 0=red, 1=yellow, 2=green")
    light_sn: int = Field(default=0, description="SN direction light: 0=red, 1=yellow, 2=green")
    light_ew: int = Field(default=0, description="EW direction light: 0=red, 1=yellow, 2=green")
    light_we: int = Field(default=0, description="WE direction light: 0=red, 1=yellow, 2=green")

    # Emergency vehicle info
    emergency_direction: int = Field(
        default=-1,
        description="Direction with emergency vehicle: 0=NS, 1=SN, 2=EW, 3=WE, -1=none",
    )
    emergency_lane: int = Field(
        default=-1,
        description="Specific lane with emergency vehicle (0-7), -1=none",
    )
    emergency_wait: int = Field(
        default=0, description="Steps the emergency vehicle has been waiting"
    )

    # Phase and timing
    active_phase: int = Field(
        default=0,
        description="Active phase 0-5 (-1 during yellow transition)",
    )
    yellow_remaining: int = Field(
        default=0, description="Steps remaining in yellow transition (0 = not transitioning)"
    )
    time_in_phase: int = Field(default=0, description="Steps since last phase change completed")
    step_number: int = Field(default=0, description="Current simulation step")

    # Aggregate stats
    total_waiting: int = Field(default=0, description="Total vehicles across all lanes and zones")
    total_throughput: int = Field(default=0, description="Cumulative vehicles that have passed through")
    arrivals: List[int] = Field(
        default_factory=lambda: [0, 0, 0, 0],
        description="Vehicles arrived this step per direction [NS, SN, EW, WE]",
    )
    departures: List[int] = Field(
        default_factory=lambda: [0, 0, 0, 0],
        description="Vehicles departed this step per direction [NS, SN, EW, WE]",
    )

    # Per-lane detail (8 values: NS_L0, NS_L1, SN_L0, SN_L1, EW_L0, EW_L1, WE_L0, WE_L1)
    lanes_100m: List[int] = Field(
        default_factory=lambda: [0] * NUM_LANES,
        description="Per-lane 100 m queue counts (8 lanes)",
    )
    lanes_500m: List[int] = Field(
        default_factory=lambda: [0] * NUM_LANES,
        description="Per-lane 500 m queue counts (8 lanes)",
    )

    # Vehicle type composition per direction at 100 m
    # Keys: vehicle type name, Values: list of 4 ints [NS, SN, EW, WE]
    vehicles_100m: Dict[str, List[int]] = Field(
        default_factory=lambda: {vt: [0] * NUM_DIRECTIONS for vt in VEHICLE_TYPE_NAMES},
        description="Per-type, per-direction vehicle counts at 100 m zone",
    )
    vehicles_500m: Dict[str, List[int]] = Field(
        default_factory=lambda: {vt: [0] * NUM_DIRECTIONS for vt in VEHICLE_TYPE_NAMES},
        description="Per-type, per-direction vehicle counts at 500 m zone",
    )

    # Dilemma zone — physics-based safety metric
    dilemma_risk: float = Field(
        default=0.0,
        description=(
            "Number of vehicles in the dilemma zone this step (can't stop safely "
            "after phase switch). 0.0 when no switch occurred."
        ),
    )
    total_dilemma_vehicles: float = Field(
        default=0.0,
        description="Cumulative dilemma-zone vehicles across the episode",
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
