# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Traffic Light Environment Implementation.

Simulates a single 4-way intersection where an RL agent controls
the traffic light to minimize total vehicle waiting time.

Directions: NS (north→south), SN (south→north),
            EW (east→west),   WE (west→east).
Each direction has 2 lanes, giving 8 lanes total.
Each lane has two zones:
  - 100 m zone: vehicles near the stop line, can depart when green
  - 500 m zone: vehicles approaching, migrate toward 100 m each step

Light states per direction: red (0), yellow (1), green (2).
When the agent switches phase, a mandatory yellow transition
(YELLOW_DURATION steps) occurs — no departures during yellow.

Vehicle types (car, suv, bus, truck, motorcycle) have real-world
physics properties. When a phase switch triggers yellow, vehicles
in the 100 m zone whose stopping distance exceeds their position
are in the **dilemma zone** — they cannot safely stop.

Phases (6):
  0  NS+SN corridor    1  EW+WE corridor
  2  NS only           3  SN only
  4  EW only           5  WE only
"""

import math
import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        DILEMMA_FRACTIONS,
        DIR_EW,
        DIR_NS,
        DIR_SN,
        DIR_WE,
        LANES_PER_DIRECTION,
        LIGHT_GREEN,
        LIGHT_RED,
        LIGHT_YELLOW,
        NUM_DIRECTIONS,
        NUM_LANES,
        NUM_PHASES,
        TASK_NAMES,
        VEHICLE_TYPE_NAMES,
        VEHICLE_TYPES,
        TrafficLightAction,
        TrafficLightObservation,
    )
    from .rubrics import TrafficLightRubric
except ImportError:
    from models import (
        DILEMMA_FRACTIONS,
        DIR_EW,
        DIR_NS,
        DIR_SN,
        DIR_WE,
        LANES_PER_DIRECTION,
        LIGHT_GREEN,
        LIGHT_RED,
        LIGHT_YELLOW,
        NUM_DIRECTIONS,
        NUM_LANES,
        NUM_PHASES,
        TASK_NAMES,
        VEHICLE_TYPE_NAMES,
        VEHICLE_TYPES,
        TrafficLightAction,
        TrafficLightObservation,
    )
    from server.rubrics import TrafficLightRubric

# ---------------------------------------------------------------------------
# Direction / lane / phase mappings
# ---------------------------------------------------------------------------

DIR_LANES: list[list[int]] = [
    [0, 1],  # NS
    [2, 3],  # SN
    [4, 5],  # EW
    [6, 7],  # WE
]

PHASE_GREEN_DIRS: dict[int, list[int]] = {
    0: [DIR_NS, DIR_SN],
    1: [DIR_EW, DIR_WE],
    2: [DIR_NS],
    3: [DIR_SN],
    4: [DIR_EW],
    5: [DIR_WE],
}

LANE_TO_DIR: list[int] = []
for _d in range(NUM_DIRECTIONS):
    for _ in range(LANES_PER_DIRECTION):
        LANE_TO_DIR.append(_d)

# Spawn weights for random.choices
_SPAWN_WEIGHTS: list[float] = [VEHICLE_TYPES[vt]["spawn_weight"] for vt in VEHICLE_TYPE_NAMES]

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

MAX_STEPS = 200
MIGRATION_RATE = 0.4
GREEN_THROUGHPUT = 3
YELLOW_DURATION = 2
SWITCH_PENALTY = -2.0
MAX_QUEUE_100M = 30
MAX_QUEUE_500M = 40
EMERGENCY_PENALTY = -5.0
DILEMMA_PENALTY = -1.5   # per dilemma-zone vehicle

# ---------------------------------------------------------------------------
# Per-task arrival rates [NS, SN, EW, WE]
# ---------------------------------------------------------------------------

TASK_CONFIGS: dict[str, dict] = {
    "balanced": {
        "arrival_rates": [1.0, 1.0, 1.0, 1.0],
    },
    "rush_hour_ns": {
        "arrival_rates": [2.0, 1.8, 0.4, 0.4],
    },
    "rush_hour_ew": {
        "arrival_rates": [0.4, 0.4, 1.8, 2.0],
    },
    "alternating_surge": {
        "arrival_rates": [0.8, 0.8, 0.8, 0.8],
        "surge_period": 30,
        "surge_boost": 1.2,
    },
    "random_spikes": {
        "arrival_rates": [0.8, 0.8, 0.8, 0.8],
        "spike_prob": 0.08,
        "spike_duration": 5,
        "spike_rate": 3.0,
    },
    "gridlock": {
        "arrival_rates": [2.0, 2.0, 2.0, 2.0],
    },
    "emergency_vehicle": {
        "arrival_rates": [1.0, 1.0, 1.0, 1.0],
        "emergency_appear_step": 10,
    },
}


def _empty_type_counts() -> list[dict[str, int]]:
    """Create empty per-lane vehicle-type count dicts."""
    return [{vt: 0 for vt in VEHICLE_TYPE_NAMES} for _ in range(NUM_LANES)]


class TrafficLightEnvironment(Environment):
    """
    RL environment for a single 4-way intersection traffic light.

    Tracks individual vehicle types per lane with physics-based stopping
    distances. Phase switches incur dilemma-zone risk when heavy or fast
    vehicles cannot stop safely within the 100 m zone.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__(rubric=TrafficLightRubric())
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Per-lane, per-type vehicle counts
        self._veh_100m: list[dict[str, int]] = _empty_type_counts()
        self._veh_500m: list[dict[str, int]] = _empty_type_counts()

        # Per-direction light states
        self._lights: list[int] = [LIGHT_RED] * NUM_DIRECTIONS

        # Phase state
        self._active_phase: int = 0
        self._yellow_remaining: int = 0
        self._pending_phase: int = -1
        self._time_in_phase: int = 0
        self._total_throughput: int = 0
        self._total_dilemma: float = 0.0
        self._rng = random.Random()

        # Task state
        self._task_name: str = "balanced"
        self._task_config: dict = TASK_CONFIGS["balanced"]

        # Emergency vehicle state
        self._emergency_lane: int = -1
        self._emergency_direction: int = -1
        self._emergency_wait: int = 0
        self._emergency_cleared: bool = False

        # Random spikes state
        self._spike_direction: int = -1
        self._spike_remaining: int = 0

    # ------------------------------------------------------------------
    # Convenience: lane totals from type dicts
    # ------------------------------------------------------------------

    def _lane_total(self, zone: list[dict[str, int]], lane: int) -> int:
        return sum(zone[lane].values())

    def _dir_total(self, zone: list[dict[str, int]], d: int) -> int:
        return sum(self._lane_total(zone, ln) for ln in DIR_LANES[d])

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
    ) -> TrafficLightObservation:
        self._state = State(
            episode_id=episode_id or str(uuid4()), step_count=0
        )
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        # Select task
        if task == "random":
            self._task_name = self._rng.choice(TASK_NAMES)
        elif task is not None and task in TASK_NAMES:
            self._task_name = task
        else:
            self._task_name = "balanced"
        self._task_config = TASK_CONFIGS[self._task_name]

        # Reset queues
        self._veh_100m = _empty_type_counts()
        self._veh_500m = _empty_type_counts()

        # Start with NS+SN corridor green (phase 0)
        self._active_phase = 0
        self._lights = [LIGHT_RED] * NUM_DIRECTIONS
        self._set_lights_for_phase(0)
        self._yellow_remaining = 0
        self._pending_phase = -1
        self._time_in_phase = 0
        self._total_throughput = 0
        self._total_dilemma = 0.0

        # Reset emergency
        self._emergency_lane = -1
        self._emergency_direction = -1
        self._emergency_wait = 0
        self._emergency_cleared = False

        # Reset spike
        self._spike_direction = -1
        self._spike_remaining = 0

        self._reset_rubric()

        return self._build_observation(
            arrivals=[0] * NUM_DIRECTIONS,
            departures=[0] * NUM_DIRECTIONS,
            done=False,
            reward=0.0,
            switched=False,
            dilemma_risk=0.0,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: TrafficLightAction) -> TrafficLightObservation:  # type: ignore[override]
        self._state.step_count += 1
        desired_phase = action.phase
        switched = False
        dilemma_risk = 0.0

        # --- 1. Phase transition logic ---
        if self._yellow_remaining > 0:
            self._yellow_remaining -= 1
            if self._yellow_remaining == 0:
                self._active_phase = self._pending_phase
                self._pending_phase = -1
                self._time_in_phase = 1
                self._set_lights_for_phase(self._active_phase)
        elif desired_phase != self._active_phase:
            switched = True
            # Compute dilemma-zone risk BEFORE changing lights
            prev_green_dirs = PHASE_GREEN_DIRS.get(self._active_phase, [])
            dilemma_risk = self._compute_dilemma_risk(prev_green_dirs)
            self._total_dilemma += dilemma_risk

            self._yellow_remaining = YELLOW_DURATION
            self._pending_phase = desired_phase
            self._time_in_phase = 0
            for d in prev_green_dirs:
                self._lights[d] = LIGHT_YELLOW
            self._active_phase = -1
        else:
            self._time_in_phase += 1

        # --- 2. Departures (green directions only) ---
        departures_per_lane = [0] * NUM_LANES
        for d in range(NUM_DIRECTIONS):
            if self._lights[d] == LIGHT_GREEN:
                for lane in DIR_LANES[d]:
                    lane_total = self._lane_total(self._veh_100m, lane)
                    depart = min(lane_total, GREEN_THROUGHPUT)
                    departures_per_lane[lane] = depart
                    self._remove_vehicles(self._veh_100m, lane, depart)
                    self._total_throughput += depart

        departures = self._aggregate_per_dir(departures_per_lane)

        # --- 3. Migration: 500 m → 100 m (per type) ---
        for lane in range(NUM_LANES):
            cap = MAX_QUEUE_100M - self._lane_total(self._veh_100m, lane)
            if cap <= 0:
                continue
            for vt in VEHICLE_TYPE_NAMES:
                count_500 = self._veh_500m[lane][vt]
                if count_500 > 0 and cap > 0:
                    migrate = min(self._binomial(count_500, MIGRATION_RATE), cap)
                    self._veh_500m[lane][vt] -= migrate
                    self._veh_100m[lane][vt] += migrate
                    cap -= migrate

        # --- 4. Arrival rates ---
        dir_rates = self._get_arrival_rates()

        # --- 5. Arrivals at 500 m zone ---
        arrivals_per_lane = [0] * NUM_LANES
        for d in range(NUM_DIRECTIONS):
            lane_rate = dir_rates[d] / LANES_PER_DIRECTION
            for lane in DIR_LANES[d]:
                cap = MAX_QUEUE_500M - self._lane_total(self._veh_500m, lane)
                n_arrive = min(self._poisson(lane_rate), cap)
                arrivals_per_lane[lane] = n_arrive
                # Assign vehicle types
                if n_arrive > 0:
                    types = self._rng.choices(
                        VEHICLE_TYPE_NAMES, weights=_SPAWN_WEIGHTS, k=n_arrive
                    )
                    for vt in types:
                        self._veh_500m[lane][vt] += 1

        arrivals = self._aggregate_per_dir(arrivals_per_lane)

        # --- 6. Emergency vehicle ---
        self._update_emergency(departures_per_lane)

        # --- 7. Reward ---
        q100 = sum(self._lane_total(self._veh_100m, ln) for ln in range(NUM_LANES))
        q500 = sum(self._lane_total(self._veh_500m, ln) for ln in range(NUM_LANES))
        reward = -float(q100) - 0.3 * float(q500)
        if switched:
            reward += SWITCH_PENALTY
        if dilemma_risk > 0:
            reward += DILEMMA_PENALTY * dilemma_risk
        if self._emergency_lane >= 0:
            reward += EMERGENCY_PENALTY

        done = self._state.step_count >= MAX_STEPS

        obs = self._build_observation(
            arrivals=arrivals,
            departures=departures,
            done=done,
            reward=reward,
            switched=switched,
            dilemma_risk=dilemma_risk,
        )

        # Rubric grading on terminal step
        grade_reward = self._apply_rubric(action, obs)
        if done and self.rubric is not None:
            obs.grade_score = round(grade_reward, 4)
            obs.grade_details = self.rubric.grade_details

        return obs

    # ------------------------------------------------------------------
    # Dilemma zone computation
    # ------------------------------------------------------------------

    def _compute_dilemma_risk(self, green_dirs: list[int]) -> float:
        """Compute dilemma-zone risk for directions turning from green to yellow.

        For each vehicle type in each green lane's 100 m zone, the fraction
        at risk equals stopping_distance / 100 m.  We assume vehicles are
        uniformly distributed within the zone.

        Returns total expected dilemma-zone vehicles (float).
        """
        risk = 0.0
        for d in green_dirs:
            for lane in DIR_LANES[d]:
                for vt in VEHICLE_TYPE_NAMES:
                    count = self._veh_100m[lane][vt]
                    if count > 0:
                        risk += count * DILEMMA_FRACTIONS[vt]
        return round(risk, 2)

    # ------------------------------------------------------------------
    # Task-specific arrival rate logic
    # ------------------------------------------------------------------

    def _get_arrival_rates(self) -> list[float]:
        base = list(self._task_config["arrival_rates"])
        step = self._state.step_count

        if self._task_name == "alternating_surge":
            period = self._task_config["surge_period"]
            boost = self._task_config["surge_boost"]
            in_first_half = (step // period) % 2 == 0
            if in_first_half:
                base[DIR_NS] += boost
                base[DIR_SN] += boost
            else:
                base[DIR_EW] += boost
                base[DIR_WE] += boost

        elif self._task_name == "random_spikes":
            if self._spike_remaining > 0:
                self._spike_remaining -= 1
                base[self._spike_direction] = self._task_config["spike_rate"]
            else:
                self._spike_direction = -1
                if self._rng.random() < self._task_config["spike_prob"]:
                    self._spike_direction = self._rng.randint(0, NUM_DIRECTIONS - 1)
                    self._spike_remaining = self._task_config["spike_duration"]
                    base[self._spike_direction] = self._task_config["spike_rate"]

        return base

    # ------------------------------------------------------------------
    # Emergency vehicle logic
    # ------------------------------------------------------------------

    def _update_emergency(self, departures_per_lane: list[int]) -> None:
        if self._task_name != "emergency_vehicle":
            return

        appear_step = self._task_config["emergency_appear_step"]

        if (
            self._state.step_count == appear_step
            and self._emergency_lane < 0
            and not self._emergency_cleared
        ):
            self._emergency_lane = self._rng.randint(0, NUM_LANES - 1)
            self._emergency_direction = LANE_TO_DIR[self._emergency_lane]
            self._emergency_wait = 0
            # Emergency vehicle is always a car (trained driver, fast reaction)
            self._veh_100m[self._emergency_lane]["car"] += 1

        if self._emergency_lane >= 0:
            if departures_per_lane[self._emergency_lane] > 0:
                self._emergency_cleared = True
                self._emergency_lane = -1
                self._emergency_direction = -1
                self._emergency_wait = 0
            else:
                self._emergency_wait += 1

    # ------------------------------------------------------------------
    # Vehicle removal (proportional to type distribution)
    # ------------------------------------------------------------------

    def _remove_vehicles(
        self, zone: list[dict[str, int]], lane: int, count: int
    ) -> None:
        """Remove `count` vehicles from a lane, proportional to type mix."""
        total = self._lane_total(zone, lane)
        if total == 0 or count == 0:
            return
        remaining = count
        # First pass: proportional removal
        for vt in VEHICLE_TYPE_NAMES:
            if remaining <= 0:
                break
            type_count = zone[lane][vt]
            remove = min(type_count, round(count * type_count / total))
            remove = min(remove, remaining)
            zone[lane][vt] -= remove
            remaining -= remove
        # Second pass: remove any remainder
        for vt in VEHICLE_TYPE_NAMES:
            if remaining <= 0:
                break
            avail = zone[lane][vt]
            take = min(avail, remaining)
            zone[lane][vt] -= take
            remaining -= take

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _aggregate_per_dir(self, per_lane: list[int]) -> list[int]:
        result = [0] * NUM_DIRECTIONS
        for d in range(NUM_DIRECTIONS):
            for lane in DIR_LANES[d]:
                result[d] += per_lane[lane]
        return result

    def _set_lights_for_phase(self, phase: int) -> None:
        green_dirs = PHASE_GREEN_DIRS.get(phase, [])
        for d in range(NUM_DIRECTIONS):
            self._lights[d] = LIGHT_GREEN if d in green_dirs else LIGHT_RED

    def _dir_type_counts(
        self, zone: list[dict[str, int]]
    ) -> dict[str, list[int]]:
        """Aggregate per-type counts per direction."""
        result: dict[str, list[int]] = {
            vt: [0] * NUM_DIRECTIONS for vt in VEHICLE_TYPE_NAMES
        }
        for d in range(NUM_DIRECTIONS):
            for lane in DIR_LANES[d]:
                for vt in VEHICLE_TYPE_NAMES:
                    result[vt][d] += zone[lane][vt]
        return result

    def _build_observation(
        self,
        arrivals: list[int],
        departures: list[int],
        done: bool,
        reward: float,
        switched: bool,
        dilemma_risk: float,
    ) -> TrafficLightObservation:
        lanes_100m = [self._lane_total(self._veh_100m, ln) for ln in range(NUM_LANES)]
        lanes_500m = [self._lane_total(self._veh_500m, ln) for ln in range(NUM_LANES)]
        total_waiting = sum(lanes_100m) + sum(lanes_500m)

        return TrafficLightObservation(
            task_name=self._task_name,
            ns_100m=self._dir_total(self._veh_100m, DIR_NS),
            sn_100m=self._dir_total(self._veh_100m, DIR_SN),
            ew_100m=self._dir_total(self._veh_100m, DIR_EW),
            we_100m=self._dir_total(self._veh_100m, DIR_WE),
            ns_500m=self._dir_total(self._veh_500m, DIR_NS),
            sn_500m=self._dir_total(self._veh_500m, DIR_SN),
            ew_500m=self._dir_total(self._veh_500m, DIR_EW),
            we_500m=self._dir_total(self._veh_500m, DIR_WE),
            light_ns=self._lights[DIR_NS],
            light_sn=self._lights[DIR_SN],
            light_ew=self._lights[DIR_EW],
            light_we=self._lights[DIR_WE],
            emergency_direction=self._emergency_direction,
            emergency_lane=self._emergency_lane,
            emergency_wait=self._emergency_wait,
            active_phase=self._active_phase,
            yellow_remaining=self._yellow_remaining,
            time_in_phase=self._time_in_phase,
            step_number=self._state.step_count,
            total_waiting=total_waiting,
            total_throughput=self._total_throughput,
            arrivals=arrivals,
            departures=departures,
            lanes_100m=lanes_100m,
            lanes_500m=lanes_500m,
            vehicles_100m=self._dir_type_counts(self._veh_100m),
            vehicles_500m=self._dir_type_counts(self._veh_500m),
            dilemma_risk=dilemma_risk,
            total_dilemma_vehicles=round(self._total_dilemma, 2),
            done=done,
            reward=reward,
            metadata={
                "switched": switched,
                "dilemma_risk": dilemma_risk,
                "lights": list(self._lights),
                "emergency_direction": self._emergency_direction,
                "emergency_lane": self._emergency_lane,
                "emergency_wait": self._emergency_wait,
            },
        )

    def _poisson(self, lam: float) -> int:
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self._rng.random()
            if p <= L:
                return k - 1

    def _binomial(self, n: int, p: float) -> int:
        count = 0
        for _ in range(n):
            if self._rng.random() < p:
                count += 1
        return count

    @property
    def state(self) -> State:
        return self._state
