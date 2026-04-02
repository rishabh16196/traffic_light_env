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

An RL environment simulating a 4-way traffic intersection where an AI agent controls the traffic light to minimize vehicle waiting time. Features 7 task scenarios with increasing difficulty, realistic traffic dynamics, and automated grading rubrics.

## Quick Start

```python
from traffic_light_env import TrafficLightAction, TrafficLightEnv

with TrafficLightEnv(base_url="http://localhost:8000") as env:
    obs = env.reset(task="balanced")

    for step in range(200):
        # Choose phase based on which direction has more waiting vehicles
        ns_load = obs.north_100m + obs.south_100m
        ew_load = obs.east_100m + obs.west_100m
        phase = 0 if ns_load >= ew_load else 1

        result = env.step(TrafficLightAction(phase=phase))
        obs = result.observation

        if obs.done:
            print(f"Episode finished!")
            print(f"  Grade: {obs.grade_score}")
            print(f"  Passed: {obs.grade_details['passed']}")
            print(f"  Avg waiting: {obs.grade_details['avg_waiting']}")
            print(f"  Throughput: {obs.grade_details['total_throughput']}")
            break
```

### From Docker

```python
from traffic_light_env import TrafficLightAction, TrafficLightEnv

env = TrafficLightEnv.from_docker_image("traffic_light_env-env:latest")
obs = env.reset(task="gridlock")
# ... run episode ...
env.close()
```

## Building the Docker Image

```bash
docker build -t traffic_light_env-env:latest -f server/Dockerfile .
```

## How It Works

### Intersection Model

A single 4-way intersection with lanes: North, South, East, West.

Each lane has two observation zones:
- **100m zone**: Vehicles near the stop line, ready to depart on green (max 30 per lane)
- **500m zone**: Vehicles approaching, migrating toward 100m each step (max 40 per lane)

### Agent Actions

The agent picks one of two phases each step:
- **Phase 0**: North-South green, East-West red
- **Phase 1**: East-West green, North-South red

Switching phases triggers a mandatory **2-step yellow transition** where no vehicles can depart.

### Step Mechanics

Each timestep:
1. Process phase/yellow transition based on agent's action
2. Depart up to 3 vehicles per green lane from the 100m zone
3. Migrate 40% of 500m vehicles to 100m zone
4. Arrive new vehicles at 500m zone (Poisson-distributed)
5. Update emergency vehicle state (if applicable)
6. Compute reward

### Reward Signal

Per-step reward (used for RL training):
- `-1.0` per vehicle in the 100m zone
- `-0.3` per vehicle in the 500m zone
- `-2.0` penalty when switching phases
- `-5.0` per step while an emergency vehicle is waiting

## Tasks

Seven scenarios with increasing difficulty, selected at reset:

```python
obs = env.reset(task="balanced")    # or "random" for a random task
```

| Task | Difficulty | Traffic Pattern |
|---|---|---|
| `balanced` | Easy | Equal rates (0.5) all directions |
| `rush_hour_ns` | Medium | Heavy N/S (1.2, 1.0), light E/W (0.2, 0.2) |
| `rush_hour_ew` | Medium | Heavy E/W (1.0, 1.2), light N/S (0.2, 0.2) |
| `alternating_surge` | Hard | Surges alternate N/S and E/W every 30 steps |
| `random_spikes` | Hard | Random bursts on random lanes |
| `gridlock` | Very Hard | All directions heavy (1.0) |
| `emergency_vehicle` | Hard | Normal traffic + emergency vehicle at step 10 |

## Grading

Each episode is automatically graded on a **0.0-1.0 scale** when the episode ends (step 200). The grade is returned in the terminal observation:

```python
obs.grade_score    # 0.0 - 1.0
obs.grade_details  # Full breakdown dict
```

### Grading Components

**Standard tasks** (50/50 weighting):

| Component | Weight | Metric |
|---|---|---|
| Waiting score | 50% | Average vehicles waiting per step (lower is better) |
| Throughput score | 50% | Total vehicles cleared over episode (higher is better) |

**Emergency vehicle task** (35/25/40 weighting):

| Component | Weight | Metric |
|---|---|---|
| Waiting score | 35% | Average vehicles waiting per step |
| Throughput score | 25% | Total vehicles cleared |
| Emergency score | 40% | How quickly the emergency vehicle was cleared |

### Per-Task Thresholds

Each task has difficulty-appropriate thresholds for a perfect (1.0) vs failing (0.0) score:

| Task | Perfect avg_waiting | Fail avg_waiting | Perfect throughput | Fail throughput |
|---|---|---|---|---|
| `balanced` | <= 4 | >= 20 | >= 180 | <= 60 |
| `rush_hour_ns` | <= 8 | >= 25 | >= 200 | <= 80 |
| `rush_hour_ew` | <= 8 | >= 25 | >= 200 | <= 80 |
| `alternating_surge` | <= 10 | >= 30 | >= 180 | <= 60 |
| `random_spikes` | <= 10 | >= 30 | >= 160 | <= 50 |
| `gridlock` | <= 15 | >= 40 | >= 250 | <= 100 |
| `emergency_vehicle` | <= 4 | >= 20 | >= 180 | <= 60 |

A score >= **0.5** is considered **passed**.

### Emergency Clearance Scoring

| Cleared within | Score |
|---|---|
| 3 steps | 1.0 |
| 8 steps | 0.7 |
| 15 steps | 0.4 |
| 30 steps | 0.1 |
| Not cleared | 0.0 |

## Observation

The `TrafficLightObservation` provides:

| Field | Description |
|---|---|
| `north_100m`, `south_100m`, `east_100m`, `west_100m` | Vehicles within 100m per lane |
| `north_500m`, `south_500m`, `east_500m`, `west_500m` | Vehicles between 100-500m per lane |
| `light_north`, `light_south`, `light_east`, `light_west` | Per-lane light state (0=red, 1=yellow, 2=green) |
| `active_phase` | Current green direction (0=NS, 1=EW, -1=yellow) |
| `yellow_remaining` | Steps left in yellow transition |
| `time_in_phase` | Steps since last phase change |
| `emergency_lane` | Lane with emergency vehicle (-1=none) |
| `emergency_wait` | Steps the emergency vehicle has waited |
| `total_waiting` | Total vehicles across all zones |
| `total_throughput` | Cumulative vehicles cleared |
| `arrivals` | Vehicles arrived this step [N, S, E, W] |
| `departures` | Vehicles departed this step [N, S, E, W] |
| `step_number` | Current step (0-200) |
| `done` | Whether the episode is over |
| `reward` | Per-step reward signal |
| `grade_score` | Final grade 0.0-1.0 (only on terminal step) |
| `grade_details` | Grading breakdown dict (only on terminal step) |

## Deploying to Hugging Face Spaces

```bash
# Push to your namespace
openenv push

# Push to a specific repo as private
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

```python
from concurrent.futures import ThreadPoolExecutor

def run_episode(task: str):
    with TrafficLightEnv(base_url="http://localhost:8000") as env:
        obs = env.reset(task=task)
        for _ in range(200):
            result = env.step(TrafficLightAction(phase=0))
            obs = result.observation
            if obs.done:
                return task, obs.grade_score

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, ["balanced", "gridlock", "rush_hour_ns", "emergency_vehicle"]))
```

## Project Structure

```
traffic_light_env/
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies
├── __init__.py            # Module exports
├── client.py              # TrafficLightEnv client (HTTP/WebSocket)
├── models.py              # Action, Observation, and constants
└── server/
    ├── app.py             # FastAPI application
    ├── traffic_light_env_environment.py  # Core simulation logic
    ├── rubrics.py         # Grading rubrics (per-task evaluation)
    └── Dockerfile         # Container image definition
```
