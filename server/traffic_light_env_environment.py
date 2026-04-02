# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Traffic Light Environment Implementation.

Simulates a single 4-way intersection where an RL agent controls
the traffic light to minimize total vehicle waiting time.

Lanes: North, South, East, West (indices 0, 1, 2, 3).
Each lane has two zones:
  - 100m zone: vehicles near the stop line, can depart when green
  - 500m zone: vehicles approaching, migrate toward 100m each step

Light states per lane: red (0), yellow (1), green (2).
When the agent switches direction, a mandatory yellow transition
(YELLOW_DURATION steps) occurs — no departures during yellow.

Tasks:
  - balanced: equal traffic all directions
  - rush_hour_ns: heavy N/S, light E/W
  - rush_hour_ew: heavy E/W, light N/S
  - alternating_surge: traffic surges alternate directions every 30 steps
  - random_spikes: random bursts on random lanes
  - gridlock: all lanes heavy
  - emergency_vehicle: normal traffic + an emergency vehicle on a random lane
"""

import math
import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        LIGHT_GREEN,
        LIGHT_RED,
        LIGHT_YELLOW,
        TASK_NAMES,
        TrafficLightAction,
        TrafficLightObservation,
    )
    from .rubrics import TrafficLightRubric
except ImportError:
    from models import (
        LIGHT_GREEN,
        LIGHT_RED,
        LIGHT_YELLOW,
        TASK_NAMES,
        TrafficLightAction,
        TrafficLightObservation,
    )
    from server.rubrics import TrafficLightRubric

# Lane indices
NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
LANE_NAMES = ["north", "south", "east", "west"]

# Which lanes are green in each phase
PHASE_GREEN_LANES = {
    0: [NORTH, SOUTH],  # NS green
    1: [EAST, WEST],    # EW green
}

# Lane-to-phase mapping: which phase makes this lane green
LANE_TO_PHASE = {NORTH: 0, SOUTH: 0, EAST: 1, WEST: 1}

# Global config
MAX_STEPS = 200
MIGRATION_RATE = 0.4   # fraction of 500m vehicles that move to 100m each step
GREEN_THROUGHPUT = 3    # max vehicles departing a green lane's 100m zone per step
YELLOW_DURATION = 2     # steps of yellow before new green activates
SWITCH_PENALTY = -2.0   # reward penalty when a yellow transition starts
MAX_QUEUE_100M = 30     # cap per lane at 100m
MAX_QUEUE_500M = 40     # cap per lane at 500m
EMERGENCY_PENALTY = -5.0  # per-step penalty while emergency vehicle is waiting

# Per-task arrival rates [N, S, E, W]
TASK_CONFIGS = {
    "balanced": {
        "arrival_rates": [0.5, 0.5, 0.5, 0.5],
    },
    "rush_hour_ns": {
        "arrival_rates": [1.2, 1.0, 0.2, 0.2],
    },
    "rush_hour_ew": {
        "arrival_rates": [0.2, 0.2, 1.0, 1.2],
    },
    "alternating_surge": {
        "arrival_rates": [0.4, 0.4, 0.4, 0.4],  # base rates, modified in step()
        "surge_period": 30,   # steps per half-cycle
        "surge_boost": 0.8,   # added to surging direction
    },
    "random_spikes": {
        "arrival_rates": [0.4, 0.4, 0.4, 0.4],  # base rates, modified in step()
        "spike_prob": 0.08,   # probability a spike starts on any step
        "spike_duration": 5,  # how long a spike lasts
        "spike_rate": 2.0,    # arrival rate during spike
    },
    "gridlock": {
        "arrival_rates": [1.0, 1.0, 1.0, 1.0],
    },
    "emergency_vehicle": {
        "arrival_rates": [0.5, 0.5, 0.5, 0.5],  # normal background traffic
        "emergency_appear_step": 10,  # when the emergency vehicle shows up
    },
}


class TrafficLightEnvironment(Environment):
    """
    RL environment for a single 4-way intersection traffic light.

    The agent observes vehicle counts at 100m and 500m per lane plus
    per-lane light states (red/yellow/green), then picks the desired
    green direction. The goal is to minimize cumulative waiting.

    Use reset(task="task_name") to select a scenario, or task="random"
    to sample one uniformly.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__(rubric=TrafficLightRubric())
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._queues_100m = [0, 0, 0, 0]
        self._queues_500m = [0, 0, 0, 0]
        self._lights = [LIGHT_GREEN, LIGHT_GREEN, LIGHT_RED, LIGHT_RED]
        self._active_phase = 0
        self._yellow_remaining = 0
        self._pending_phase = -1
        self._time_in_phase = 0
        self._total_throughput = 0
        self._rng = random.Random()

        # Task state
        self._task_name = "balanced"
        self._task_config: dict = TASK_CONFIGS["balanced"]

        # Emergency vehicle state
        self._emergency_lane = -1    # -1 = no emergency vehicle
        self._emergency_wait = 0     # steps it has been waiting
        self._emergency_cleared = False

        # Random spikes state
        self._spike_lane = -1        # which lane is spiking (-1 = none)
        self._spike_remaining = 0    # steps left in current spike

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
    ) -> TrafficLightObservation:
        """Reset the intersection.

        Args:
            seed: Random seed for reproducibility.
            episode_id: Custom episode ID.
            task: Task name (one of TASK_NAMES) or "random" to sample one.
                  Defaults to "balanced".
        """
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

        # Reset intersection state
        self._queues_100m = [0, 0, 0, 0]
        self._queues_500m = [0, 0, 0, 0]
        self._lights = [LIGHT_GREEN, LIGHT_GREEN, LIGHT_RED, LIGHT_RED]
        self._active_phase = 0
        self._yellow_remaining = 0
        self._pending_phase = -1
        self._time_in_phase = 0
        self._total_throughput = 0

        # Reset emergency state
        self._emergency_lane = -1
        self._emergency_wait = 0
        self._emergency_cleared = False

        # Reset spike state
        self._spike_lane = -1
        self._spike_remaining = 0

        # Reset rubric for new episode
        self._reset_rubric()

        return self._build_observation(
            arrivals=[0, 0, 0, 0],
            departures=[0, 0, 0, 0],
            done=False,
            reward=0.0,
            switched=False,
        )

    def step(self, action: TrafficLightAction) -> TrafficLightObservation:  # type: ignore[override]
        """
        Advance one timestep.

        Order of operations:
        1. Process phase/yellow transition logic based on agent action.
        2. Depart vehicles from 100m zone of green lanes.
        3. Migrate vehicles from 500m zone to 100m zone.
        4. Compute arrival rates for current task/step.
        5. Arrive new vehicles at 500m zone.
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
            for lane in PHASE_GREEN_LANES[self._active_phase]:
                self._lights[lane] = LIGHT_YELLOW
            self._active_phase = -1
        else:
            self._time_in_phase += 1

        # --- 2. Departures (only from green lanes' 100m zone) ---
        departures = [0, 0, 0, 0]
        for lane in range(4):
            if self._lights[lane] == LIGHT_GREEN:
                depart = min(self._queues_100m[lane], GREEN_THROUGHPUT)
                departures[lane] = depart
                self._queues_100m[lane] -= depart
                self._total_throughput += depart

        # --- 3. Migration: 500m -> 100m ---
        for lane in range(4):
            if self._queues_500m[lane] > 0:
                migrate = self._binomial(self._queues_500m[lane], MIGRATION_RATE)
                migrate = min(migrate, MAX_QUEUE_100M - self._queues_100m[lane])
                self._queues_500m[lane] -= migrate
                self._queues_100m[lane] += migrate

        # --- 4. Compute effective arrival rates for this step ---
        arrival_rates = self._get_arrival_rates()

        # --- 5. Arrivals at 500m zone (Poisson) ---
        arrivals = [0, 0, 0, 0]
        for lane in range(4):
            n_arrive = self._poisson(arrival_rates[lane])
            arrivals[lane] = n_arrive
            self._queues_500m[lane] = min(
                self._queues_500m[lane] + n_arrive, MAX_QUEUE_500M
            )

        # --- 6. Emergency vehicle logic ---
        self._update_emergency(departures)

        # --- 7. Reward ---
        total_waiting = sum(self._queues_100m) + sum(self._queues_500m)
        reward = -float(sum(self._queues_100m)) - 0.3 * float(sum(self._queues_500m))
        if switched:
            reward += SWITCH_PENALTY
        # Emergency penalty: large per-step cost while emergency vehicle waits
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
        """Return per-lane arrival rates for the current step, accounting for task dynamics."""
        base = list(self._task_config["arrival_rates"])
        step = self._state.step_count

        if self._task_name == "alternating_surge":
            period = self._task_config["surge_period"]
            boost = self._task_config["surge_boost"]
            # First half of cycle: surge NS, second half: surge EW
            in_first_half = (step // period) % 2 == 0
            if in_first_half:
                base[NORTH] += boost
                base[SOUTH] += boost
            else:
                base[EAST] += boost
                base[WEST] += boost

        elif self._task_name == "random_spikes":
            # Manage spike lifecycle
            if self._spike_remaining > 0:
                self._spike_remaining -= 1
                base[self._spike_lane] = self._task_config["spike_rate"]
            else:
                self._spike_lane = -1
                # Possibly start a new spike
                if self._rng.random() < self._task_config["spike_prob"]:
                    self._spike_lane = self._rng.randint(0, 3)
                    self._spike_remaining = self._task_config["spike_duration"]
                    base[self._spike_lane] = self._task_config["spike_rate"]

        return base

    # ------------------------------------------------------------------
    # Emergency vehicle logic
    # ------------------------------------------------------------------

    def _update_emergency(self, departures: list[int]) -> None:
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
            self._emergency_lane = self._rng.randint(0, 3)
            self._emergency_wait = 0
            # Place the emergency vehicle in the 100m zone (it's right at the light)
            self._queues_100m[self._emergency_lane] += 1

        # If emergency vehicle is active, check if it departed
        if self._emergency_lane >= 0:
            if departures[self._emergency_lane] > 0:
                # Emergency vehicle was in the departures — it's cleared
                self._emergency_cleared = True
                self._emergency_lane = -1
                self._emergency_wait = 0
            else:
                self._emergency_wait += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_lights_for_phase(self, phase: int) -> None:
        green_lanes = PHASE_GREEN_LANES[phase]
        for lane in range(4):
            self._lights[lane] = LIGHT_GREEN if lane in green_lanes else LIGHT_RED

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
            north_100m=self._queues_100m[NORTH],
            south_100m=self._queues_100m[SOUTH],
            east_100m=self._queues_100m[EAST],
            west_100m=self._queues_100m[WEST],
            north_500m=self._queues_500m[NORTH],
            south_500m=self._queues_500m[SOUTH],
            east_500m=self._queues_500m[EAST],
            west_500m=self._queues_500m[WEST],
            light_north=self._lights[NORTH],
            light_south=self._lights[SOUTH],
            light_east=self._lights[EAST],
            light_west=self._lights[WEST],
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
            done=done,
            reward=reward,
            metadata={
                "switched": switched,
                "queues_100m": list(self._queues_100m),
                "queues_500m": list(self._queues_500m),
                "lights": list(self._lights),
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
