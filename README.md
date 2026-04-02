---
title: Traffic Light Env Environment Server
emoji: 🚦
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Traffic Light Env Environment

An RL environment simulating a 4-way traffic intersection where an AI agent controls the traffic light to minimize vehicle waiting time. The intersection has 4 traffic-flow directions (NS, SN, EW, WE), each with 2 lanes (8 lanes total), and 6 selectable green phases. Features 7 task scenarios with increasing difficulty and automated grading rubrics.

## Quick Start

```python
from traffic_light_env import TrafficLightAction, TrafficLightEnv

async with TrafficLightEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task="balanced")
    obs = result.observation

    for step in range(200):
        # Choose phase based on which axis has more waiting vehicles
        ns_sn = obs.ns_100m + obs.sn_100m
        ew_we = obs.ew_100m + obs.we_100m
        phase = 0 if ns_sn >= ew_we else 1  # corridor phases

        result = await env.step(TrafficLightAction(phase=phase))
        obs = result.observation

        if obs.done:
            print(f"Grade: {obs.grade_score}")
            print(f"Passed: {obs.grade_details['passed']}")
            break
```

### From Docker

```python
env = await TrafficLightEnv.from_docker_image("traffic_light_env-env:latest")
result = await env.reset(task="gridlock")
# ... run episode ...
await env.close()
```

## Building the Docker Image

```bash
docker build -t traffic_light_env-env:latest -f server/Dockerfile .
```

## How It Works

### Intersection Model

A single 4-way intersection with **4 traffic-flow directions**, each with **2 lanes**:

| Direction | Code | Description |
|---|---|---|
| NS | `DIR_NS = 0` | North → South (lanes 0, 1) |
| SN | `DIR_SN = 1` | South → North (lanes 2, 3) |
| EW | `DIR_EW = 2` | East → West (lanes 4, 5) |
| WE | `DIR_WE = 3` | West → East (lanes 6, 7) |

Each of the 8 lanes has two observation zones:
- **100 m zone**: Vehicles near the stop line, ready to depart on green (max 30 per lane)
- **500 m zone**: Vehicles approaching, migrating toward 100 m each step (max 40 per lane)

Each direction has its own traffic light (red/yellow/green).

### Agent Actions — 6 Phases

The agent selects one of 6 phases each step:

| Phase | Green Directions | Lanes Green | Description |
|---|---|---|---|
| **0** | NS + SN | 4 | Full north-south corridor |
| **1** | EW + WE | 4 | Full east-west corridor |
| **2** | NS only | 2 | North-to-south only |
| **3** | SN only | 2 | South-to-north only |
| **4** | EW only | 2 | East-to-west only |
| **5** | WE only | 2 | West-to-east only |

Corridor phases (0, 1) green 4 lanes for maximum throughput. Single-direction phases (2-5) are useful when one direction is much busier than its opposite. Switching phases triggers a mandatory **2-step yellow transition**.

### Vehicle Types and Stopping Physics

Each vehicle is randomly assigned a type with real-world physics properties:

| Type | Speed | Reaction Time | Deceleration | Stopping Distance | Dilemma Fraction | Spawn Weight |
|---|---|---|---|---|---|---|
| Car | 50 km/h | 1.0 s | 6.8 m/s² | 28.1 m | 28.1% | 40% |
| SUV | 50 km/h | 1.2 s | 6.0 m/s² | 32.7 m | 32.7% | 25% |
| Bus | 40 km/h | 1.5 s | 4.5 m/s² | 30.4 m | 30.4% | 10% |
| Truck | 45 km/h | 1.4 s | 4.0 m/s² | 37.0 m | 37.0% | 15% |
| Motorcycle | 55 km/h | 0.8 s | 7.5 m/s² | 27.8 m | 27.8% | 10% |

**Stopping distance** = reaction distance + braking distance:
- `d_reaction = speed × reaction_time`
- `d_braking = speed² / (2 × deceleration)`

Assumes dry urban road conditions (friction coefficient ~0.7).

### Dilemma Zone

When the agent switches phases (green → yellow), vehicles in the 100 m zone whose stopping distance exceeds their position from the intersection are in the **dilemma zone** — they cannot safely stop in time. The risk is computed assuming vehicles are uniformly distributed within the zone:

`dilemma_vehicles = Σ (vehicle_count × stopping_distance / 100)`

This creates a key strategic tension: switching phases clears different queues but puts vehicles at risk. The agent must balance **throughput** (clearing queues) against **safety** (avoiding dilemma-zone incidents), especially when heavy vehicles (trucks, buses) are in the green lanes.

### Step Mechanics

Each timestep:
1. Process phase/yellow transition based on agent's action
2. **If switching**: compute dilemma-zone risk for previously-green directions
3. Depart up to 3 vehicles per green lane from the 100 m zone
4. Migrate 40% of 500 m vehicles to 100 m zone
5. Arrive new vehicles at 500 m zone (Poisson-distributed per lane, random type)
6. Update emergency vehicle state (if applicable)
7. Compute reward

### Reward Signal

Per-step reward (used for RL training):
- `-1.0` per vehicle in any 100 m zone
- `-0.3` per vehicle in any 500 m zone
- `-2.0` penalty when switching phases
- `-1.5` per dilemma-zone vehicle (on phase switch)
- `-5.0` per step while an emergency vehicle is waiting

## Tasks

Seven scenarios with increasing difficulty, selected at reset:

```python
result = await env.reset(task="balanced")    # or "random" for a random task
```

| Task | Difficulty | Arrival Rates [NS, SN, EW, WE] | Notes |
|---|---|---|---|
| `balanced` | Easy | [1.0, 1.0, 1.0, 1.0] | Equal traffic all directions |
| `rush_hour_ns` | Medium | [2.0, 1.8, 0.4, 0.4] | Heavy north-south corridor |
| `rush_hour_ew` | Medium | [0.4, 0.4, 1.8, 2.0] | Heavy east-west corridor |
| `alternating_surge` | Hard | [0.8, 0.8, 0.8, 0.8] + surges | NS/SN and EW/WE surge alternately every 30 steps |
| `random_spikes` | Hard | [0.8, 0.8, 0.8, 0.8] + spikes | Random bursts on random directions |
| `gridlock` | Very Hard | [2.0, 2.0, 2.0, 2.0] | All directions heavy |
| `emergency_vehicle` | Hard | [1.0, 1.0, 1.0, 1.0] + emergency | Emergency vehicle spawns at step 10 |

Arrival rates are per direction; each lane receives half the direction rate.

## Grading

Each episode is automatically graded on a **0.0-1.0 scale** at step 200. The grade is returned in the terminal observation:

```python
obs.grade_score    # 0.0 - 1.0
obs.grade_details  # Full breakdown dict
```

### Grading Components

**Standard tasks** (40/40/20 weighting):

| Component | Weight | Metric |
|---|---|---|
| Waiting score | 40% | Average vehicles waiting per step (lower is better) |
| Throughput score | 40% | Total vehicles cleared over episode (higher is better) |
| Safety score | 20% | Cumulative dilemma-zone vehicles (lower is better, 0=perfect, 50+=fail) |

**Emergency vehicle task** (25/20/15/40 weighting):

| Component | Weight | Metric |
|---|---|---|
| Waiting score | 25% | Average vehicles waiting per step |
| Throughput score | 20% | Total vehicles cleared |
| Safety score | 15% | Cumulative dilemma-zone vehicles |
| Emergency score | 40% | How quickly the emergency vehicle was cleared |

### Per-Task Thresholds

| Task | Perfect avg_waiting | Fail avg_waiting | Perfect throughput | Fail throughput |
|---|---|---|---|---|
| `balanced` | <= 15 | >= 70 | >= 700 | <= 250 |
| `rush_hour_ns` | <= 15 | >= 60 | >= 800 | <= 300 |
| `rush_hour_ew` | <= 15 | >= 60 | >= 800 | <= 300 |
| `alternating_surge` | <= 30 | >= 120 | >= 800 | <= 300 |
| `random_spikes` | <= 15 | >= 60 | >= 600 | <= 200 |
| `gridlock` | <= 100 | >= 500 | >= 800 | <= 350 |
| `emergency_vehicle` | <= 15 | >= 70 | >= 700 | <= 250 |

A score >= **0.5** is considered **passed**.

### Emergency Clearance Scoring

| Cleared within | Score |
|---|---|
| 3 steps | 1.0 |
| 8 steps | 0.7 |
| 15 steps | 0.4 |
| 30 steps | 0.1 |
| Not cleared | 0.0 |

## Baseline Scores

Two baselines compared — fixed-timer (switches every 10 steps) and smart heuristic (adaptive). Seed=42, 200 steps per episode.

### Fixed 10-step timer (best overall baseline)

| Task | Score | Waiting | Throughput | Safety | Emergency | Dilemma | Result |
|---|---|---|---|---|---|---|---|
| `balanced` | 0.8314 | 0.671 | 1.000 | 0.815 | — | 9.25 | PASS |
| `rush_hour_ns` | 0.6906 | 0.409 | 1.000 | 0.636 | — | 18.22 | PASS |
| `rush_hour_ew` | 0.7710 | 0.534 | 1.000 | 0.786 | — | 10.68 | PASS |
| `alternating_surge` | 0.8103 | 0.791 | 1.000 | 0.469 | — | 26.54 | PASS |
| `random_spikes` | 0.8132 | 0.630 | 1.000 | 0.806 | — | 9.72 | PASS |
| `gridlock` | 0.8482 | 1.000 | 1.000 | 0.241 | — | 37.96 | PASS |
| `emergency_vehicle` | 0.8845 | 0.701 | 1.000 | 0.729 | 1.000 | 13.54 | PASS |

**Average: 0.807**

### Smart heuristic (adaptive, switches on demand)

| Task | Score | Waiting | Throughput | Safety | Emergency | Dilemma | Result |
|---|---|---|---|---|---|---|---|
| `balanced` | 0.6032 | 0.508 | 1.000 | 0.000 | — | 208.16 | PASS |
| `rush_hour_ns` | 0.7545 | 0.727 | 1.000 | 0.319 | — | 34.05 | PASS |
| `rush_hour_ew` | 0.7772 | 0.754 | 1.000 | 0.378 | — | 31.09 | PASS |
| `alternating_surge` | 0.5906 | 0.477 | 1.000 | 0.000 | — | 409.59 | PASS |
| `random_spikes` | 0.6448 | 0.612 | 1.000 | 0.000 | — | 125.39 | PASS |
| `gridlock` | 0.5334 | 0.334 | 1.000 | 0.000 | — | 1989.23 | PASS |
| `emergency_vehicle` | 0.6932 | 0.373 | 1.000 | 0.000 | 1.000 | 297.64 | PASS |

**Average: 0.657**

The smart heuristic switches too often, causing massive dilemma-zone incidents (up to 1989 vehicles on gridlock). The fixed timer is safer but can't adapt to asymmetric traffic. An LLM agent that reasons about both traffic patterns and vehicle composition should outperform both.

## Observation

The `TrafficLightObservation` provides:

| Field | Description |
|---|---|
| `ns_100m`, `sn_100m`, `ew_100m`, `we_100m` | Per-direction 100 m queue totals (sum of 2 lanes) |
| `ns_500m`, `sn_500m`, `ew_500m`, `we_500m` | Per-direction 500 m queue totals |
| `light_ns`, `light_sn`, `light_ew`, `light_we` | Per-direction light state (0=red, 1=yellow, 2=green) |
| `active_phase` | Current phase 0-5 (-1 during yellow) |
| `yellow_remaining` | Steps left in yellow transition |
| `time_in_phase` | Steps since last phase change |
| `emergency_direction` | Direction with emergency vehicle (0-3, -1=none) |
| `emergency_lane` | Specific lane (0-7, -1=none) |
| `emergency_wait` | Steps the emergency vehicle has waited |
| `total_waiting` | Total vehicles across all zones |
| `total_throughput` | Cumulative vehicles cleared |
| `arrivals` | Vehicles arrived this step per direction [NS, SN, EW, WE] |
| `departures` | Vehicles departed this step per direction |
| `lanes_100m` | Per-lane 100 m queues (8 values) |
| `lanes_500m` | Per-lane 500 m queues (8 values) |
| `vehicles_100m` | Per-type, per-direction counts at 100 m (`{"car": [ns,sn,ew,we], ...}`) |
| `vehicles_500m` | Per-type, per-direction counts at 500 m |
| `dilemma_risk` | Dilemma-zone vehicles this step (0.0 if no switch) |
| `total_dilemma_vehicles` | Cumulative dilemma-zone vehicles this episode |
| `step_number` | Current step (0-200) |
| `done` | Whether the episode is over |
| `reward` | Per-step reward signal |
| `grade_score` | Final grade 0.0-1.0 (only on terminal step) |
| `grade_details` | Grading breakdown dict (only on terminal step) |

## Deploying to Hugging Face Spaces

```bash
openenv push
openenv push --repo-id my-org/traffic-light-env --private
```

The deployed space includes:
- **Web Interface** at `/web`
- **API Documentation** at `/docs`
- **Health Check** at `/health`
- **WebSocket** at `/ws`

## Development

### Running Locally

```bash
uvicorn server.app:app --reload
```

### Concurrent Sessions

The server supports multiple concurrent WebSocket connections (configured in `server/app.py` via `max_concurrent_envs`).

## Project Structure

```
traffic_light_env/
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies
├── inference.py           # Baseline inference script (LLM agent)
├── __init__.py            # Module exports
├── client.py              # TrafficLightEnv client (HTTP/WebSocket)
├── models.py              # Action, Observation, constants
└── server/
    ├── app.py             # FastAPI application
    ├── traffic_light_env_environment.py  # Core simulation logic
    ├── rubrics.py         # Grading rubrics (per-task evaluation)
    └── Dockerfile         # Container image definition
```
