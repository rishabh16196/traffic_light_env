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

Phases (6):
  0  NS+SN corridor — both north-south directions green
  1  EW+WE corridor — both east-west directions green
  2  NS only
  3  SN only
  4  EW only
  5  WE only

Tasks:
  balanced, rush_hour_ns, rush_hour_ew, alternating_surge,
  random_spikes, gridlock, emergency_vehicle
"""

import math
import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
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
        TrafficLightAction,
        TrafficLightObservation,
    )
    from .rubrics import TrafficLightRubric
except ImportError:
    from models import (
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
        TrafficLightAction,
        TrafficLightObservation,
    )
    from server.rubrics import TrafficLightRubric

# ---------------------------------------------------------------------------
# Direction / lane / phase mappings
# ---------------------------------------------------------------------------

# Direction index → lane indices (2 lanes per direction)
DIR_LANES: list[list[int]] = [
    [0, 1],  # NS
    [2, 3],  # SN
    [4, 5],  # EW
    [6, 7],  # WE
]

# Phase → which directions get green
PHASE_GREEN_DIRS: dict[int, list[int]] = {
    0: [DIR_NS, DIR_SN],  # NS+SN corridor
    1: [DIR_EW, DIR_WE],  # EW+WE corridor
    2: [DIR_NS],           # NS only
    3: [DIR_SN],           # SN only
    4: [DIR_EW],           # EW only
    5: [DIR_WE],           # WE only
}

# Lane → direction (derived)
LANE_TO_DIR: list[int] = []
for _d in range(NUM_DIRECTIONS):
    for _ in range(LANES_PER_DIRECTION):
        LANE_TO_DIR.append(_d)

# Direction → which phases make it green
DIR_TO_PHASES: dict[int, list[int]] = {d: [] for d in range(NUM_DIRECTIONS)}
for _ph, _dirs in PHASE_GREEN_DIRS.items():
    for _d in _dirs:
        DIR_TO_PHASES[_d].append(_ph)

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

MAX_STEPS = 200
MIGRATION_RATE = 0.4    # fraction of 500 m vehicles that move to 100 m each step
GREEN_THROUGHPUT = 3     # max vehicles departing a green lane's 100 m zone per step
YELLOW_DURATION = 2      # steps of yellow before new green activates
SWITCH_PENALTY = -2.0    # reward penalty when a yellow transition starts
MAX_QUEUE_100M = 30      # cap per lane at 100 m
MAX_QUEUE_500M = 40      # cap per lane at 500 m
EMERGENCY_PENALTY = -5.0 # per-step penalty while emergency vehicle is waiting

# ---------------------------------------------------------------------------
# Per-task arrival rates [NS, SN, EW, WE] (per direction, split across lanes)
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
        "arrival_rates": [0.8, 0.8, 0.8, 0.8],  # base; modified in step
        "surge_period": 30,
        "surge_boost": 1.2,
    },
    "random_spikes": {
        "arrival_rates": [0.8, 0.8, 0.8, 0.8],  # base; modified in step
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


class TrafficLightEnvironment(Environment):
    """
    RL environment for a single 4-way intersection traffic light.

    The agent observes per-direction vehicle counts at 100 m and 500 m,
    per-direction light states, then picks one of 6 green phases.
    The goal is to minimize cumulative waiting.

    Use reset(task="task_name") to select a scenario, or task="random"
    to sample one uniformly.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__(rubric=TrafficLightRubric())
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Per-lane queues (8 lanes)
        self._queues_100m: list[int] = [0] * NUM_LANES
        self._queues_500m: list[int] = [0] * NUM_LANES

        # Per-direction light states (4 directions)
        self._lights: list[int] = [LIGHT_RED] * NUM_DIRECTIONS

        # Phase state
        self._active_phase: int = 0
        self._yellow_remaining: int = 0
        self._pending_phase: int = -1
        self._time_in_phase: int = 0
        self._total_throughput: int = 0
        self._rng = random.Random()

        # Task state
        self._task_name: str = "balanced"
        self._task_config: dict = TASK_CONFIGS["balanced"]

        # Emergency vehicle state
        self._emergency_lane: int = -1       # specific lane (0-7), -1 = none
        self._emergency_direction: int = -1  # direction (0-3), -1 = none
        self._emergency_wait: int = 0
        self._emergency_cleared: bool = False

        # Random spikes state
        self._spike_direction: int = -1
        self._spike_remaining: int = 0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
    ) -> TrafficLightObservation:
        """Reset the intersection."""
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
        self._queues_100m = [0] * NUM_LANES
        self._queues_500m = [0] * NUM_LANES

        # Start with NS+SN corridor green (phase 0)
        self._active_phase = 0
        self._lights = [LIGHT_RED] * NUM_DIRECTIONS
        self._set_lights_for_phase(0)
        self._yellow_remaining = 0
        self._pending_phase = -1
        self._time_in_phase = 0
        self._total_throughput = 0

        # Reset emergency state
        self._emergency_lane = -1
        self._emergency_direction = -1
        self._emergency_wait = 0
        self._emergency_cleared = False

        # Reset spike state
        self._spike_direction = -1
        self._spike_remaining = 0

        # Reset rubric for new episode
        self._reset_rubric()

        return self._build_observation(
            arrivals=[0] * NUM_DIRECTIONS,
            departures=[0] * NUM_DIRECTIONS,
            done=False,
            reward=0.0,
            switched=False,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: TrafficLightAction) -> TrafficLightObservation:  # type: ignore[override]
        """
        Advance one timestep.

        Order of operations:
        1. Process phase/yellow transition logic based on agent action.
        2. Depart vehicles from 100 m zone of green directions.
        3. Migrate vehicles from 500 m zone to 100 m zone.
        4. Compute arrival rates for current task/step.
        5. Arrive new vehicles at 500 m zone.
        6. Update emergency vehicle state.
        7. Compute reward.
        """
        self._state.step_count += 1
        desired_phase = action.phase
        switched = False

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
            self._yellow_remaining = YELLOW_DURATION
            self._pending_phase = desired_phase
            self._time_in_phase = 0
            # Set currently-green directions to yellow
            for d in PHASE_GREEN_DIRS.get(self._active_phase, []):
                self._lights[d] = LIGHT_YELLOW
            self._active_phase = -1
        else:
            self._time_in_phase += 1

        # --- 2. Departures (only from green directions' lanes) ---
        departures_per_lane = [0] * NUM_LANES
        for d in range(NUM_DIRECTIONS):
            if self._lights[d] == LIGHT_GREEN:
                for lane in DIR_LANES[d]:
                    depart = min(self._queues_100m[lane], GREEN_THROUGHPUT)
                    departures_per_lane[lane] = depart
                    self._queues_100m[lane] -= depart
                    self._total_throughput += depart

        # Aggregate departures per direction
        departures = [0] * NUM_DIRECTIONS
        for d in range(NUM_DIRECTIONS):
            for lane in DIR_LANES[d]:
                departures[d] += departures_per_lane[lane]

        # --- 3. Migration: 500 m → 100 m ---
        for lane in range(NUM_LANES):
            if self._queues_500m[lane] > 0:
                migrate = self._binomial(self._queues_500m[lane], MIGRATION_RATE)
                migrate = min(migrate, MAX_QUEUE_100M - self._queues_100m[lane])
                self._queues_500m[lane] -= migrate
                self._queues_100m[lane] += migrate

        # --- 4. Compute effective arrival rates for this step ---
        dir_rates = self._get_arrival_rates()

        # --- 5. Arrivals at 500 m zone (Poisson, split across lanes) ---
        arrivals_per_lane = [0] * NUM_LANES
        for d in range(NUM_DIRECTIONS):
            lane_rate = dir_rates[d] / LANES_PER_DIRECTION
            for lane in DIR_LANES[d]:
                n_arrive = self._poisson(lane_rate)
                arrivals_per_lane[lane] = n_arrive
                self._queues_500m[lane] = min(
                    self._queues_500m[lane] + n_arrive, MAX_QUEUE_500M
                )

        # Aggregate arrivals per direction
        arrivals = [0] * NUM_DIRECTIONS
        for d in range(NUM_DIRECTIONS):
            for lane in DIR_LANES[d]:
                arrivals[d] += arrivals_per_lane[lane]

        # --- 6. Emergency vehicle logic ---
        self._update_emergency(departures_per_lane)

        # --- 7. Reward ---
        reward = -float(sum(self._queues_100m)) - 0.3 * float(sum(self._queues_500m))
        if switched:
            reward += SWITCH_PENALTY
        if self._emergency_lane >= 0:
            reward += EMERGENCY_PENALTY

        done = self._state.step_count >= MAX_STEPS

        obs = self._build_observation(
            arrivals=arrivals,
            departures=departures,
            done=done,
            reward=reward,
            switched=switched,
        )

        # Apply rubric — accumulates trajectory and grades on terminal step
        grade_reward = self._apply_rubric(action, obs)
        if done and self.rubric is not None:
            obs.grade_score = round(grade_reward, 4)
            obs.grade_details = self.rubric.grade_details

        return obs

    # ------------------------------------------------------------------
    # Task-specific arrival rate logic
    # ------------------------------------------------------------------

    def _get_arrival_rates(self) -> list[float]:
        """Return per-direction arrival rates for the current step."""
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
        """Manage emergency vehicle appearance and clearance."""
        if self._task_name != "emergency_vehicle":
            return

        appear_step = self._task_config["emergency_appear_step"]

        # Spawn the emergency vehicle at the designated step
        if (
            self._state.step_count == appear_step
            and self._emergency_lane < 0
            and not self._emergency_cleared
        ):
            self._emergency_lane = self._rng.randint(0, NUM_LANES - 1)
            self._emergency_direction = LANE_TO_DIR[self._emergency_lane]
            self._emergency_wait = 0
            self._queues_100m[self._emergency_lane] += 1

        # If emergency vehicle is active, check if it departed
        if self._emergency_lane >= 0:
            if departures_per_lane[self._emergency_lane] > 0:
                self._emergency_cleared = True
                self._emergency_lane = -1
                self._emergency_direction = -1
                self._emergency_wait = 0
            else:
                self._emergency_wait += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_lights_for_phase(self, phase: int) -> None:
        green_dirs = PHASE_GREEN_DIRS.get(phase, [])
        for d in range(NUM_DIRECTIONS):
            self._lights[d] = LIGHT_GREEN if d in green_dirs else LIGHT_RED

    def _dir_queue_100m(self, d: int) -> int:
        return sum(self._queues_100m[lane] for lane in DIR_LANES[d])

    def _dir_queue_500m(self, d: int) -> int:
        return sum(self._queues_500m[lane] for lane in DIR_LANES[d])

    def _build_observation(
        self,
        arrivals: list[int],
        departures: list[int],
        done: bool,
        reward: float,
        switched: bool,
    ) -> TrafficLightObservation:
        total_waiting = sum(self._queues_100m) + sum(self._queues_500m)
        return TrafficLightObservation(
            task_name=self._task_name,
            # Per-direction 100 m totals
            ns_100m=self._dir_queue_100m(DIR_NS),
            sn_100m=self._dir_queue_100m(DIR_SN),
            ew_100m=self._dir_queue_100m(DIR_EW),
            we_100m=self._dir_queue_100m(DIR_WE),
            # Per-direction 500 m totals
            ns_500m=self._dir_queue_500m(DIR_NS),
            sn_500m=self._dir_queue_500m(DIR_SN),
            ew_500m=self._dir_queue_500m(DIR_EW),
            we_500m=self._dir_queue_500m(DIR_WE),
            # Per-direction lights
            light_ns=self._lights[DIR_NS],
            light_sn=self._lights[DIR_SN],
            light_ew=self._lights[DIR_EW],
            light_we=self._lights[DIR_WE],
            # Emergency
            emergency_direction=self._emergency_direction,
            emergency_lane=self._emergency_lane,
            emergency_wait=self._emergency_wait,
            # Phase / timing
            active_phase=self._active_phase,
            yellow_remaining=self._yellow_remaining,
            time_in_phase=self._time_in_phase,
            step_number=self._state.step_count,
            # Aggregates
            total_waiting=total_waiting,
            total_throughput=self._total_throughput,
            arrivals=arrivals,
            departures=departures,
            # Per-lane detail
            lanes_100m=list(self._queues_100m),
            lanes_500m=list(self._queues_500m),
            done=done,
            reward=reward,
            metadata={
                "switched": switched,
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
