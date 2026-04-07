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

# Smart Traffic Light Control Environment

Anyone who has sat at a red light with zero cars on the cross street knows the frustration: dumb traffic signals waste millions of hours every day. This environment exists to change that. It provides a realistic 4-way intersection simulator where AI agents learn to control traffic lights intelligently — minimizing waiting time, maximizing throughput, and keeping vehicles safe.

## The Problem

Traditional traffic lights run on fixed timers or simple sensors. They can't anticipate surges, adapt to rush-hour asymmetry, or reason about the safety cost of switching. The result: needless idling, wasted fuel, and preventable dilemma-zone accidents. This environment lets us train and evaluate models that do better.

## What We Built

- A physics-based intersection simulator with 5 real-world vehicle types, dilemma-zone safety modeling, and 7 task scenarios ranging from balanced traffic to gridlock
- A hybrid LLM + heuristic inference agent (`inference.py`) that uses task-specific strategies with periodic LLM consultation
- A FastAPI server exposing the environment over HTTP/WebSocket, deployable via Docker or Hugging Face Spaces
- Automated grading rubrics that score agents on waiting time, throughput, and safety

### Our Agent's Results (avg 0.83, beating the 0.807 fixed-timer baseline)

| Task | Our Score | Fixed Timer |
|---|---|---|
| balanced | 0.82 | 0.83 |
| rush_hour_ns | 0.79 | 0.69 |
| rush_hour_ew | 0.82 | 0.77 |
| alternating_surge | 0.87 | 0.81 |
| random_spikes | 0.83 | 0.81 |
| gridlock | 0.88 | 0.85 |
| emergency_vehicle | 0.88 | 0.88 |

## Quick Start

```python
from traffic_light_env import TrafficLightAction, TrafficLightEnv

async with TrafficLightEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task="balanced")
    obs = result.observation

    for step in range(200):
        ns_sn = obs.ns_100m + obs.sn_100m
        ew_we = obs.ew_100m + obs.we_100m
        phase = 0 if ns_sn >= ew_we else 1

        result = await env.step(TrafficLightAction(phase=phase))
        obs = result.observation

        if obs.done:
            print(f"Grade: {obs.grade_score}")
            break
```

## Intersection Model

A 4-way intersection with **4 directions** (NS, SN, EW, WE), **2 lanes each** (8 total). Each lane has a **100 m zone** (vehicles ready to depart) and a **500 m zone** (vehicles approaching).

The agent picks one of **6 phases** each step:

| Phase | Green Directions | Description |
|---|---|---|
| 0 | NS + SN | Full north-south corridor (4 lanes) |
| 1 | EW + WE | Full east-west corridor (4 lanes) |
| 2 | NS only | North-to-south only (2 lanes) |
| 3 | SN only | South-to-north only (2 lanes) |
| 4 | EW only | East-to-west only (2 lanes) |
| 5 | WE only | West-to-east only (2 lanes) |

Switching phases triggers a **2-step yellow transition** with no departures.

## Vehicle Types

Five vehicle types with real-world stopping physics. Heavier vehicles are harder to stop, creating higher dilemma-zone risk when switching phases.

| Type | Speed | Stopping Distance | Dilemma Risk | Spawn Rate |
|---|---|---|---|---|
| Car | 50 km/h | 28.1 m | 28% | 40% |
| SUV | 50 km/h | 32.7 m | 33% | 25% |
| Truck | 45 km/h | 37.0 m | 37% | 15% |
| Bus | 40 km/h | 30.4 m | 30% | 10% |
| Motorcycle | 55 km/h | 27.8 m | 28% | 10% |

When a phase switch occurs, vehicles in the 100 m zone that can't stop safely are in the **dilemma zone**. Each dilemma-zone vehicle incurs a -1.5 reward penalty. Trucks and buses are the riskiest.

## Tasks

Seven scenarios with increasing difficulty:

| Task | Difficulty | Arrival Rates [NS, SN, EW, WE] | Notes |
|---|---|---|---|
| `balanced` | Easy | [1.0, 1.0, 1.0, 1.0] | Equal traffic all directions |
| `rush_hour_ns` | Medium | [2.0, 1.8, 0.4, 0.4] | Heavy north-south corridor |
| `rush_hour_ew` | Medium | [0.4, 0.4, 1.8, 2.0] | Heavy east-west corridor |
| `alternating_surge` | Hard | [0.8, 0.8, 0.8, 0.8] + surges | NS/SN and EW/WE surge alternately every 30 steps |
| `random_spikes` | Hard | [0.8, 0.8, 0.8, 0.8] + spikes | Random bursts on random directions |
| `gridlock` | Very Hard | [2.0, 2.0, 2.0, 2.0] | All directions heavy |
| `emergency_vehicle` | Hard | [1.0, 1.0, 1.0, 1.0] + emergency | Emergency vehicle spawns at step 10 |

## Grading

Episodes are graded 0.0-1.0 at step 200. Standard tasks: 40% waiting + 40% throughput + 20% safety. Emergency task: 25% waiting + 20% throughput + 15% safety + 40% emergency clearance speed. Score >= 0.5 = pass.

## Our Inference Strategy

The `inference.py` agent uses a **hybrid heuristic + LLM approach**:

1. **Task-specific heuristics** handle most steps (fast, no API cost, avoids over-switching)
2. **Periodic LLM consultation** (via OpenAI-compatible API) provides strategic guidance at key decision points
3. **Per-task tuning**: different hold times, switch thresholds, and strategies for each scenario
4. **Dilemma-zone awareness**: factors in vehicle composition before switching to minimize safety penalties
5. **Pattern detection**: pre-emptively switches for alternating surge boundaries, uses fixed-timer for gridlock, and immediately overrides for emergency vehicles

## Running

```bash
# Start the server
uvicorn traffic_light_env.server.app:app --reload --port 8000

# Run inference (set your API key)
OPENAI_API_KEY="..." API_BASE_URL="https://api.openai.com/v1" MODEL_NAME="gpt-4o-mini" \
  python traffic_light_env/inference.py

# Docker
docker build -t traffic_light_env-env:latest -f server/Dockerfile .
```

## Project Structure

```
traffic_light_env/
├── inference.py           # Hybrid LLM + heuristic agent
├── models.py              # Action, Observation, vehicle physics constants
├── client.py              # TrafficLightEnv client (HTTP/WebSocket)
├── __init__.py            # Module exports
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
└── server/
    ├── app.py             # FastAPI application
    ├── traffic_light_env_environment.py  # Core simulation logic
    ├── rubrics.py         # Grading rubrics (per-task evaluation)
    └── Dockerfile         # Container image definition
```
