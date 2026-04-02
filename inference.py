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
    [STEP] step=1 action=phase(0) reward=-1.20 done=false error=null
    [STEP] step=2 action=phase(1) reward=-3.50 done=false error=null
    ...
    [END] success=true steps=200 score=0.624 rewards=-1.20,-3.50,...
"""

import asyncio
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from traffic_light_env import TrafficLightAction, TrafficLightEnv
from traffic_light_env.models import TASK_NAMES

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
    You are controlling a traffic light at a 4-way intersection. Your goal is to
    minimize total vehicle waiting time by choosing the optimal green phase each step.

    Actions:
      0 = North-South green (East-West red)
      1 = East-West green (North-South red)

    Switching phases incurs a 2-step yellow transition (no departures) and a -2.0
    reward penalty. Avoid unnecessary switching.

    Strategy tips:
    - Keep the phase green for the direction with more waiting vehicles.
    - Consider 500m vehicles: they will arrive at 100m soon.
    - For emergency vehicles, prioritize clearing the emergency lane quickly (-5.0 per step penalty).
    - Avoid rapid switching — the yellow transition wastes 2 steps.

    Respond with ONLY a single digit: 0 or 1
    """
).strip()


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


def obs_to_summary(obs: Any) -> str:
    """Build a concise text summary of the current observation for the LLM."""
    lines = [
        f"Step: {obs.step_number}/{MAX_STEPS}",
        f"Task: {obs.task_name}",
        f"Active phase: {obs.active_phase} ({'NS green' if obs.active_phase == 0 else 'EW green' if obs.active_phase == 1 else 'yellow transition'})",
        f"Yellow remaining: {obs.yellow_remaining}",
        f"Time in phase: {obs.time_in_phase}",
        f"100m queues — N:{obs.north_100m} S:{obs.south_100m} E:{obs.east_100m} W:{obs.west_100m}",
        f"500m queues — N:{obs.north_500m} S:{obs.south_500m} E:{obs.east_500m} W:{obs.west_500m}",
        f"Total waiting: {obs.total_waiting}",
        f"Throughput so far: {obs.total_throughput}",
    ]
    if obs.emergency_lane >= 0:
        lane_name = ["North", "South", "East", "West"][obs.emergency_lane]
        phase_needed = 0 if obs.emergency_lane in (0, 1) else 1
        lines.append(
            f"EMERGENCY vehicle in {lane_name} lane (needs phase {phase_needed}), waiting {obs.emergency_wait} steps"
        )
    return "\n".join(lines)


def get_phase_from_llm(
    client: OpenAI,
    obs: Any,
    history: List[str],
) -> int:
    """Ask the LLM which phase to choose. Falls back to heuristic on failure."""
    user_prompt = obs_to_summary(obs)
    if history:
        user_prompt += "\n\nRecent history:\n" + "\n".join(history[-5:])
    user_prompt += "\n\nChoose phase (0 or 1):"

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
        # Extract the first 0 or 1 from the response
        for ch in text:
            if ch in ("0", "1"):
                return int(ch)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)

    # Fallback heuristic: pick the direction with more 100m vehicles
    return heuristic_phase(obs)


def heuristic_phase(obs: Any) -> int:
    """Simple heuristic: green the direction with more waiting vehicles."""
    # If emergency vehicle is active, prioritize its lane
    if obs.emergency_lane >= 0:
        return 0 if obs.emergency_lane in (0, 1) else 1

    ns = obs.north_100m + obs.south_100m + 0.3 * (obs.north_500m + obs.south_500m)
    ew = obs.east_100m + obs.west_100m + 0.3 * (obs.east_500m + obs.west_500m)
    return 0 if ns >= ew else 1


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
                # Extract grade from terminal observation
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
        avg_score = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0
        print(f"  Average score: {avg_score:.4f}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
