"""
Inference Script — Traffic Light Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    OPENAI_API_KEY   Your API key (also accepts HF_TOKEN or API_KEY as fallbacks).
    API_BASE_URL     The API endpoint for the LLM.
    MODEL_NAME       The model identifier to use for inference.
    IMAGE_NAME       The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Example:
    [START] task=balanced env=traffic_light_env model=Qwen2.5-72B-Instruct
    [STEP] step=1 action=phase(0) reward=-2.40 done=false error=null
    [STEP] step=2 action=phase(1) reward=-5.10 done=false error=null
    ...
    [END] success=true steps=200 score=0.624 rewards=-2.40,-5.10,...
"""

import asyncio
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from traffic_light_env import TrafficLightAction, TrafficLightEnv
from traffic_light_env.models import DIRECTION_NAMES, NUM_PHASES, TASK_NAMES

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "traffic_light_env"
MAX_STEPS = 200
TEMPERATURE = 0.3
MAX_TOKENS = 64

# Tasks to run. Override with TRAFFIC_LIGHT_TASKS env var (comma-separated).
TASKS = os.getenv("TRAFFIC_LIGHT_TASKS", ",".join(TASK_NAMES)).split(",")

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a traffic light at a 4-way intersection with 4 directions
    (NS=north-to-south, SN=south-to-north, EW=east-to-west, WE=west-to-east),
    each with 2 lanes (8 lanes total).

    Your goal: minimize total vehicle waiting time by choosing the optimal phase.

    Available phases (pick one number 0-5):
      0 = NS+SN corridor (both north-south directions green)
      1 = EW+WE corridor (both east-west directions green)
      2 = NS only green
      3 = SN only green
      4 = EW only green
      5 = WE only green

    Phase switching incurs a 2-step yellow transition (no departures) and a -2.0
    reward penalty. Avoid unnecessary switching.

    SAFETY: Each lane has a mix of vehicle types (car, suv, bus, truck, motorcycle)
    with different stopping distances based on real physics. When you switch phases,
    vehicles in the 100m zone that can't stop in time are in the "dilemma zone":
    - Trucks: 37m stopping distance (37% of 100m zone at risk)
    - SUVs: 33m (33% at risk)
    - Buses: 30m (30% at risk)
    - Cars: 28m (28% at risk)
    - Motorcycles: 28m (28% at risk)
    Each dilemma-zone vehicle incurs a -1.5 reward penalty. Avoid switching when
    many heavy vehicles (trucks, buses) are in the green lanes' 100m zones.

    Strategy tips:
    - Corridor phases (0, 1) green 4 lanes at once — high throughput.
    - Single-direction phases (2-5) useful when one direction is much busier.
    - Consider 500m vehicles: they migrate to 100m soon.
    - For emergency vehicles, prioritize the direction containing the emergency.
    - Avoid switching when trucks/buses are in the 100m zone (high dilemma risk).
    - Minimize total switches — each costs yellow time + dilemma risk + penalty.

    Respond with ONLY a single digit: 0, 1, 2, 3, 4, or 5
    """
).strip()


# ---------------------------------------------------------------------------
# Logging helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Observation → LLM prompt
# ---------------------------------------------------------------------------

def obs_to_summary(obs: Any) -> str:
    """Build a concise text summary of the current observation for the LLM."""
    phase_desc = {
        0: "NS+SN corridor", 1: "EW+WE corridor",
        2: "NS only", 3: "SN only", 4: "EW only", 5: "WE only",
    }
    lines = [
        f"Step: {obs.step_number}/{MAX_STEPS}",
        f"Task: {obs.task_name}",
        f"Active phase: {obs.active_phase} ({phase_desc.get(obs.active_phase, 'yellow transition')})",
        f"Yellow remaining: {obs.yellow_remaining}",
        f"Time in phase: {obs.time_in_phase}",
        f"100m queues — NS:{obs.ns_100m} SN:{obs.sn_100m} EW:{obs.ew_100m} WE:{obs.we_100m}",
        f"500m queues — NS:{obs.ns_500m} SN:{obs.sn_500m} EW:{obs.ew_500m} WE:{obs.we_500m}",
        f"Total waiting: {obs.total_waiting}",
        f"Throughput so far: {obs.total_throughput}",
    ]
    # Show heavy vehicle counts in 100m zone (dilemma risk factors)
    v100 = obs.vehicles_100m
    heavy = {d: 0 for d in range(4)}
    dir_labels = ["NS", "SN", "EW", "WE"]
    for vt in ("truck", "bus", "suv"):
        for d in range(4):
            heavy[d] += v100.get(vt, [0, 0, 0, 0])[d]
    heavy_str = " ".join(f"{dir_labels[d]}:{heavy[d]}" for d in range(4))
    lines.append(f"Heavy vehicles (truck+bus+suv) at 100m — {heavy_str}")
    lines.append(f"Cumulative dilemma-zone vehicles: {obs.total_dilemma_vehicles:.1f}")

    if obs.emergency_direction >= 0:
        dir_name = DIRECTION_NAMES[obs.emergency_direction].upper()
        if obs.emergency_direction <= 1:
            phases_help = "phase 0 (corridor) or phase " + str(obs.emergency_direction + 2)
        else:
            phases_help = "phase 1 (corridor) or phase " + str(obs.emergency_direction + 2)
        lines.append(
            f"EMERGENCY vehicle in {dir_name} direction (use {phases_help}), "
            f"waiting {obs.emergency_wait} steps"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM decision
# ---------------------------------------------------------------------------

def get_phase_from_llm(
    client: OpenAI,
    obs: Any,
    history: List[str],
) -> int:
    """Ask the LLM which phase to choose. Falls back to heuristic on failure."""
    user_prompt = obs_to_summary(obs)
    if history:
        user_prompt += "\n\nRecent history:\n" + "\n".join(history[-5:])
    user_prompt += "\n\nChoose phase (0-5):"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        for ch in text:
            if ch in "012345":
                return int(ch)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)

    return heuristic_phase(obs)


def heuristic_phase(obs: Any) -> int:
    """Heuristic baseline: corridor for the busier axis, or target emergency."""
    # Emergency: green the direction containing the emergency vehicle
    if obs.emergency_direction >= 0:
        d = obs.emergency_direction
        # Use corridor if possible (0 for NS/SN, 1 for EW/WE)
        if d <= 1:
            return 0  # NS+SN corridor
        else:
            return 1  # EW+WE corridor

    # Compare NS+SN axis vs EW+WE axis (100m weighted 1.0, 500m weighted 0.3)
    ns_sn = (obs.ns_100m + obs.sn_100m) + 0.3 * (obs.ns_500m + obs.sn_500m)
    ew_we = (obs.ew_100m + obs.we_100m) + 0.3 * (obs.ew_500m + obs.we_500m)
    return 0 if ns_sn >= ew_we else 1


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, env: TrafficLightEnv, task: str) -> Dict[str, Any]:
    """Run a single task episode and return results."""
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            phase = get_phase_from_llm(client, obs, history)
            action = TrafficLightAction(phase=phase)

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=f"phase({phase})",
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: phase={phase}, waiting={obs.total_waiting}, reward={reward:+.2f}"
            )

            if done:
                score = obs.grade_score if obs.grade_score is not None else 0.0
                success = score >= 0.5
                break

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task,
        "success": success,
        "score": score,
        "steps": steps_taken,
        "grade_details": obs.grade_details if hasattr(obs, "grade_details") else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await TrafficLightEnv.from_docker_image(IMAGE_NAME)
    else:
        base_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
        env = TrafficLightEnv(base_url=base_url)
        await env.connect()

    try:
        all_results = []
        for task in TASKS:
            task = task.strip()
            if task not in TASK_NAMES:
                print(f"[DEBUG] Skipping unknown task: {task}", flush=True)
                continue
            result = await run_task(client, env, task)
            all_results.append(result)

        # Summary
        print("\n=== SUMMARY ===", flush=True)
        for r in all_results:
            status = "PASS" if r["success"] else "FAIL"
            print(
                f"  [{status}] {r['task']:22s} score={r['score']:.4f} steps={r['steps']}",
                flush=True,
            )
        avg_score = (
            sum(r["score"] for r in all_results) / len(all_results)
            if all_results else 0.0
        )
        print(f"  Average score: {avg_score:.4f}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
