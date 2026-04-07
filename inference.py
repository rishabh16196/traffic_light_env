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
from traffic_light_env.models import (
    DILEMMA_FRACTIONS,
    DIRECTION_NAMES,
    NUM_PHASES,
    TASK_NAMES,
    VEHICLE_TYPE_NAMES,
)

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "traffic_light_env"
MAX_STEPS = 200
TEMPERATURE = 0.2
MAX_TOKENS = 128

# Strategy parameters
MIN_HOLD_TIME = 8          # Minimum steps to hold a phase before considering switch
SWITCH_THRESHOLD = 1.8     # Opposing axis must be this many times busier to switch
LLM_CONSULT_INTERVAL = 10  # Ask LLM every N steps for strategic guidance
EMERGENCY_OVERRIDE = True  # Immediately switch for emergency vehicles

# Tasks to run. Override with TRAFFIC_LIGHT_TASKS env var (comma-separated).
TASKS = os.getenv("TRAFFIC_LIGHT_TASKS", ",".join(TASK_NAMES)).split(",")

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a traffic light controller at a 4-way intersection. 4 directions
    (NS, SN, EW, WE) with 2 lanes each (8 total). You pick one of 6 phases:

    0 = NS+SN corridor (4 lanes green — best throughput for N-S axis)
    1 = EW+WE corridor (4 lanes green — best throughput for E-W axis)
    2 = NS only   3 = SN only   4 = EW only   5 = WE only

    CRITICAL RULES — switching phases costs 2 dead steps (yellow) + dilemma-zone
    risk (vehicles that can't stop safely). Every unnecessary switch HURTS your score.

    DECISION FRAMEWORK:
    1. If currently in yellow transition → keep the pending phase (no choice).
    2. If emergency vehicle present → switch to its corridor ONCE, then hold.
    3. If held current phase < 8 steps → KEEP current phase (too early to switch).
    4. Only switch if opposing axis queue is >1.8× current axis queue.
    5. Prefer corridor phases (0 or 1) for maximum throughput.
    6. Use single-direction phases (2-5) ONLY if one direction has >3× its opposite.

    Scoring: 40% waiting (lower=better), 40% throughput (higher=better), 20% safety
    (fewer dilemma vehicles=better). The fixed-timer baseline scores 0.81 by switching
    every 10 steps. You should switch LESS often than that on balanced traffic.

    Respond: one line with the phase digit (0-5), then a brief reason.
    Format: <digit> <reason>
    Example: 0 NS+SN corridor has more vehicles, hold current phase
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
# Dilemma risk estimation
# ---------------------------------------------------------------------------

def estimate_dilemma_risk(obs: Any, green_dirs: List[int]) -> float:
    """Estimate how many vehicles would be in the dilemma zone if we switch now."""
    v100 = obs.vehicles_100m
    dir_labels = ["NS", "SN", "EW", "WE"]
    risk = 0.0
    for d in green_dirs:
        for vt in VEHICLE_TYPE_NAMES:
            count = v100.get(vt, [0, 0, 0, 0])[d]
            if count > 0:
                risk += count * DILEMMA_FRACTIONS[vt]
    return risk


def get_green_dirs(phase: int) -> List[int]:
    """Return which directions are green for a given phase."""
    mapping = {0: [0, 1], 1: [2, 3], 2: [0], 3: [1], 4: [2], 5: [3]}
    return mapping.get(phase, [])


# ---------------------------------------------------------------------------
# Smart heuristic (primary decision maker)
# ---------------------------------------------------------------------------

def smart_heuristic(obs: Any, current_phase: int, time_in_phase: int) -> int:
    """
    Heuristic that minimizes switching while maintaining good throughput.
    Key insight: the fixed-timer baseline (switch every 10 steps) scores 0.81.
    We can beat it by being smarter about WHEN to switch.
    """
    # During yellow, we can't do anything — return current pending or active
    if obs.yellow_remaining > 0:
        return obs.active_phase if obs.active_phase >= 0 else current_phase

    # Emergency override: immediately switch to emergency corridor
    if obs.emergency_direction >= 0:
        d = obs.emergency_direction
        target = 0 if d <= 1 else 1
        if current_phase != target:
            return target
        return current_phase

    # Compute axis loads (100m weighted heavily, 500m as future pressure)
    ns_sn_100 = obs.ns_100m + obs.sn_100m
    ew_we_100 = obs.ew_100m + obs.we_100m
    ns_sn_500 = obs.ns_500m + obs.sn_500m
    ew_we_500 = obs.ew_500m + obs.we_500m

    ns_sn_load = ns_sn_100 + 0.3 * ns_sn_500
    ew_we_load = ew_we_100 + 0.3 * ew_we_500

    # Determine which corridor the current phase serves
    current_green_dirs = get_green_dirs(current_phase)
    serves_ns = any(d in [0, 1] for d in current_green_dirs)
    serves_ew = any(d in [2, 3] for d in current_green_dirs)

    current_load = 0.0
    opposing_load = 0.0
    if serves_ns and not serves_ew:
        current_load = ns_sn_load
        opposing_load = ew_we_load
    elif serves_ew and not serves_ns:
        current_load = ew_we_load
        opposing_load = ns_sn_load
    else:
        # Phase serves both or neither — use corridor phases
        current_load = ns_sn_load
        opposing_load = ew_we_load

    # Don't switch if we haven't held long enough
    if time_in_phase < MIN_HOLD_TIME:
        return current_phase

    # Check if opposing axis is significantly busier
    if opposing_load > 0 and current_load > 0:
        ratio = opposing_load / max(current_load, 1.0)
    elif opposing_load > 0:
        ratio = 10.0  # Current axis is empty
    else:
        ratio = 0.0  # Opposing axis is empty

    # Also factor in dilemma risk — if many heavy vehicles in green lanes, don't switch
    dilemma_risk = estimate_dilemma_risk(obs, current_green_dirs)

    # Adaptive threshold: require higher ratio if dilemma risk is high
    effective_threshold = SWITCH_THRESHOLD + (dilemma_risk * 0.1)

    if ratio >= effective_threshold:
        # Switch to the opposing corridor
        if serves_ns or (not serves_ew and ns_sn_load < ew_we_load):
            # Check if one EW direction dominates — use single phase
            if obs.ew_100m > 3 * obs.we_100m and obs.ew_100m > 10:
                return 4  # EW only
            elif obs.we_100m > 3 * obs.ew_100m and obs.we_100m > 10:
                return 5  # WE only
            return 1  # EW+WE corridor
        else:
            if obs.ns_100m > 3 * obs.sn_100m and obs.ns_100m > 10:
                return 2  # NS only
            elif obs.sn_100m > 3 * obs.ns_100m and obs.sn_100m > 10:
                return 3  # SN only
            return 0  # NS+SN corridor

    # Check for very unbalanced single-direction loads within current axis
    if serves_ns and time_in_phase >= MIN_HOLD_TIME + 4:
        if obs.ns_100m > 3 * obs.sn_100m and obs.ns_100m > 15 and current_phase == 0:
            return 2  # Focus on NS only
        elif obs.sn_100m > 3 * obs.ns_100m and obs.sn_100m > 15 and current_phase == 0:
            return 3  # Focus on SN only
    elif serves_ew and time_in_phase >= MIN_HOLD_TIME + 4:
        if obs.ew_100m > 3 * obs.we_100m and obs.ew_100m > 15 and current_phase == 1:
            return 4
        elif obs.we_100m > 3 * obs.ew_100m and obs.we_100m > 15 and current_phase == 1:
            return 5

    return current_phase


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
    # Dilemma risk info
    green_dirs = get_green_dirs(obs.active_phase)
    dilemma = estimate_dilemma_risk(obs, green_dirs)
    lines.append(f"Dilemma risk if switching now: {dilemma:.1f} vehicles")
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

    # Heuristic recommendation
    heuristic_rec = smart_heuristic(obs, obs.active_phase, obs.time_in_phase)
    lines.append(f"\nHeuristic recommends: phase {heuristic_rec} ({phase_desc.get(heuristic_rec, '?')})")

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
        user_prompt += "\n\nRecent actions:\n" + "\n".join(history[-5:])
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

    return smart_heuristic(obs, obs.active_phase, obs.time_in_phase)


# ---------------------------------------------------------------------------
# Hybrid decision: heuristic + periodic LLM consultation
# ---------------------------------------------------------------------------

def decide_phase(
    client: OpenAI,
    obs: Any,
    history: List[str],
    step: int,
    current_phase: int,
    time_in_phase: int,
) -> int:
    """
    Hybrid approach:
    - Use heuristic for most steps (fast, no API cost, avoids over-switching)
    - Consult LLM every LLM_CONSULT_INTERVAL steps for strategic decisions
    - Always use heuristic for emergency overrides
    """
    # During yellow, just hold
    if obs.yellow_remaining > 0:
        return current_phase

    # Emergency: always use heuristic (fast, deterministic)
    if obs.emergency_direction >= 0:
        return smart_heuristic(obs, current_phase, time_in_phase)

    # Consult LLM at strategic intervals when we might need to switch
    if (step % LLM_CONSULT_INTERVAL == 0) and time_in_phase >= MIN_HOLD_TIME:
        return get_phase_from_llm(client, obs, history)

    # Default: use heuristic
    return smart_heuristic(obs, current_phase, time_in_phase)


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
        current_phase = 0  # Start at NS+SN corridor
        time_in_phase = 0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            phase = decide_phase(
                client, obs, history, step,
                current_phase, time_in_phase,
            )

            # Track phase timing locally
            if phase != current_phase:
                time_in_phase = 0
                current_phase = phase
            else:
                time_in_phase += 1

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
                f"Step {step}: phase={phase}, waiting={obs.total_waiting}, "
                f"throughput={obs.total_throughput}, reward={reward:+.2f}"
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
            if r.get("grade_details"):
                d = r["grade_details"]
                print(
                    f"           waiting={d.get('waiting_score', 0):.3f} "
                    f"throughput={d.get('throughput_score', 0):.3f} "
                    f"safety={d.get('safety_score', 0):.3f} "
                    f"dilemma={d.get('total_dilemma_vehicles', 0):.1f}",
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
