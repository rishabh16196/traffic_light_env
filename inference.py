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

# Per-task tuning parameters: (min_hold, switch_threshold, llm_interval)
# min_hold: minimum steps to hold a phase before considering switch
# switch_threshold: opposing axis must be this factor busier to trigger switch
# llm_interval: consult LLM every N steps (0 = never use LLM for this task)
TASK_PARAMS: Dict[str, Dict[str, Any]] = {
    "balanced":           {"min_hold": 8,  "switch_thresh": 1.6, "llm_interval": 15},
    "rush_hour_ns":       {"min_hold": 8,  "switch_thresh": 1.8, "llm_interval": 0},
    "rush_hour_ew":       {"min_hold": 8,  "switch_thresh": 1.8, "llm_interval": 0},
    "alternating_surge":  {"min_hold": 6,  "switch_thresh": 1.4, "llm_interval": 0},  # pattern-based
    "random_spikes":      {"min_hold": 8,  "switch_thresh": 1.5, "llm_interval": 15},
    "gridlock":           {"min_hold": 8,  "switch_thresh": 1.3, "llm_interval": 0},   # fixed timer
    "emergency_vehicle":  {"min_hold": 8,  "switch_thresh": 1.6, "llm_interval": 0},   # heuristic only
}
DEFAULT_PARAMS = {"min_hold": 8, "switch_thresh": 1.8, "llm_interval": 10}

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
# Task-specific strategies
# ---------------------------------------------------------------------------

def _alternating_surge_strategy(obs: Any, current_phase: int, time_in_phase: int) -> int:
    """
    Surge pattern: NS/SN surge when (step//30)%2==0, EW/WE surge otherwise.
    Pre-emptively switch 2 steps before surge boundary to absorb yellow transition.
    """
    step = obs.step_number
    period = 30

    # Which surge are we in now?
    ns_surge = (step // period) % 2 == 0
    # When does the next surge boundary hit?
    next_boundary = ((step // period) + 1) * period
    steps_to_boundary = next_boundary - step

    # Target corridor for current surge
    target = 0 if ns_surge else 1

    # Pre-emptive switch: 2 steps before boundary, switch to upcoming corridor
    if steps_to_boundary <= 2:
        upcoming_target = 1 if ns_surge else 0  # opposite of current surge
        if current_phase != upcoming_target:
            return upcoming_target
        return current_phase

    # During surge, ensure we're on the right corridor
    if current_phase != target and time_in_phase >= 6:
        return target

    # If we're on the right corridor, check for load imbalance within the axis
    if current_phase == target and time_in_phase >= 10:
        if target == 0:  # NS/SN corridor
            ns_sn_100 = obs.ns_100m + obs.sn_100m
            ew_we_100 = obs.ew_100m + obs.we_100m
            # If EW/WE is building up massively despite NS surge, give it some time
            if ew_we_100 > ns_sn_100 * 2.5 and ew_we_100 > 20:
                return 1
        else:  # EW/WE corridor
            ns_sn_100 = obs.ns_100m + obs.sn_100m
            ew_we_100 = obs.ew_100m + obs.we_100m
            if ns_sn_100 > ew_we_100 * 2.5 and ns_sn_100 > 20:
                return 0

    return current_phase


def _gridlock_strategy(obs: Any, current_phase: int, time_in_phase: int) -> int:
    """
    Gridlock: all directions have equal rate 2.0.
    Use fixed timer (~10 steps) switching between corridor 0 and 1.
    Matches the fixed-timer baseline approach which scores 0.848.
    Only use corridor phases for maximum throughput.
    """
    GRIDLOCK_CYCLE = 10

    # Ensure we only use corridor phases
    if current_phase not in (0, 1):
        return 0  # Reset to corridor

    if time_in_phase >= GRIDLOCK_CYCLE:
        # Check dilemma risk before switching
        green_dirs = get_green_dirs(current_phase)
        dilemma = estimate_dilemma_risk(obs, green_dirs)

        # Delay switch by 1-2 steps if dilemma risk is very high
        if dilemma > 8 and time_in_phase < GRIDLOCK_CYCLE + 2:
            return current_phase

        # Alternate between corridors
        return 1 if current_phase == 0 else 0

    return current_phase


def _emergency_strategy(obs: Any, current_phase: int, time_in_phase: int,
                         emergency_handled: bool) -> int:
    """
    Emergency vehicle task: prioritize clearing the emergency ASAP.
    Emergency clearance is 40% of the grade — must be within 3 steps for 1.0 score.
    Strategy: use corridor phase covering the emergency direction (greens 4 lanes,
    including the emergency lane, while maintaining throughput).
    """
    if obs.emergency_direction >= 0:
        d = obs.emergency_direction
        # Use corridor phase — it greens the emergency direction AND its opposite
        # for better throughput, while still clearing the emergency
        target = 0 if d <= 1 else 1
        if current_phase != target:
            return target
        return current_phase

    # Before emergency appears (step < 10), use balanced strategy but
    # position on phase 0 (NS+SN) to be ready for 50% of emergencies
    if not emergency_handled and obs.step_number < 10:
        # Pre-position: stay on phase 0 — if emergency is NS/SN, we're ready
        return _balanced_strategy(obs, current_phase, time_in_phase, "balanced")

    # After emergency cleared, use standard balanced strategy
    return _balanced_strategy(obs, current_phase, time_in_phase, "balanced")


def _rush_hour_strategy(obs: Any, current_phase: int, time_in_phase: int,
                         task_name: str) -> int:
    """
    Rush hour: one axis is much busier (rate ~2.0 vs ~0.4).
    Strategy: stay on the busy corridor most of the time.
    Give quiet axis brief windows (~6 steps) to prevent total starvation.
    Switch back to busy corridor as soon as quiet axis is drained.
    """
    if task_name == "rush_hour_ns":
        busy_corridor = 0  # NS+SN
    else:
        busy_corridor = 1  # EW+WE

    ns_sn_100 = obs.ns_100m + obs.sn_100m
    ew_we_100 = obs.ew_100m + obs.we_100m
    ns_sn_load = ns_sn_100 + 0.3 * (obs.ns_500m + obs.sn_500m)
    ew_we_load = ew_we_100 + 0.3 * (obs.ew_500m + obs.we_500m)

    busy_load = ns_sn_load if busy_corridor == 0 else ew_we_load
    quiet_load = ew_we_load if busy_corridor == 0 else ns_sn_load
    busy_100 = ns_sn_100 if busy_corridor == 0 else ew_we_100
    quiet_100 = ew_we_100 if busy_corridor == 0 else ns_sn_100

    green_dirs = get_green_dirs(current_phase)
    dilemma = estimate_dilemma_risk(obs, green_dirs)

    if current_phase == busy_corridor:
        # On busy corridor — hold for at least 8 steps
        if time_in_phase < 8:
            return current_phase
        # Switch only if quiet axis is building up significantly
        # and busy axis is somewhat drained
        if quiet_100 > 15 and quiet_load > busy_load * 0.6 and dilemma < 6:
            return 1 - busy_corridor
        # Force give quiet axis a window after extended hold
        if time_in_phase >= 12 and quiet_100 > 8 and dilemma < 6:
            return 1 - busy_corridor
        return current_phase
    else:
        # On quiet corridor — return to busy corridor quickly
        if time_in_phase < 5:
            return current_phase
        # Return once quiet axis is drained or busy axis is building
        if quiet_100 <= 4 or busy_100 > quiet_100 * 1.5 or time_in_phase >= 7:
            return busy_corridor
        return current_phase


def _balanced_strategy(obs: Any, current_phase: int, time_in_phase: int,
                        task_name: str) -> int:
    """General adaptive strategy for balanced/random tasks."""
    params = TASK_PARAMS.get(task_name, DEFAULT_PARAMS)
    min_hold = params["min_hold"]
    thresh = params["switch_thresh"]

    ns_sn_100 = obs.ns_100m + obs.sn_100m
    ew_we_100 = obs.ew_100m + obs.we_100m
    ns_sn_load = ns_sn_100 + 0.3 * (obs.ns_500m + obs.sn_500m)
    ew_we_load = ew_we_100 + 0.3 * (obs.ew_500m + obs.we_500m)

    green_dirs = get_green_dirs(current_phase)
    serves_ns = any(d in [0, 1] for d in green_dirs)
    serves_ew = any(d in [2, 3] for d in green_dirs)

    if serves_ns and not serves_ew:
        current_load, opposing_load = ns_sn_load, ew_we_load
    elif serves_ew and not serves_ns:
        current_load, opposing_load = ew_we_load, ns_sn_load
    else:
        current_load, opposing_load = ns_sn_load, ew_we_load

    if time_in_phase < min_hold:
        return current_phase

    # Compute switch ratio
    if current_load > 0:
        ratio = opposing_load / max(current_load, 1.0)
    elif opposing_load > 0:
        ratio = 10.0
    else:
        ratio = 0.0

    dilemma = estimate_dilemma_risk(obs, green_dirs)
    effective_thresh = thresh + (dilemma * 0.08)

    if ratio >= effective_thresh:
        if ns_sn_load < ew_we_load:
            return 1  # EW+WE corridor
        else:
            return 0  # NS+SN corridor

    # Force switch after max hold to prevent starvation
    max_hold = 14 if task_name == "random_spikes" else 12
    if time_in_phase >= max_hold and opposing_load > 5 and dilemma < 6:
        if serves_ns:
            return 1
        else:
            return 0

    return current_phase


# ---------------------------------------------------------------------------
# Smart heuristic (primary decision maker)
# ---------------------------------------------------------------------------

def smart_heuristic(obs: Any, current_phase: int, time_in_phase: int,
                     task_name: str = "balanced",
                     emergency_handled: bool = False) -> int:
    """
    Task-aware heuristic that minimizes switching while maintaining throughput.
    Dispatches to task-specific strategies.
    """
    # During yellow, can't change — hold current
    if obs.yellow_remaining > 0:
        return obs.active_phase if obs.active_phase >= 0 else current_phase

    # Emergency override for ANY task (highest priority)
    if obs.emergency_direction >= 0:
        d = obs.emergency_direction
        target = 0 if d <= 1 else 1
        if current_phase != target:
            return target
        return current_phase

    # Dispatch to task-specific strategy
    if task_name == "alternating_surge":
        return _alternating_surge_strategy(obs, current_phase, time_in_phase)
    elif task_name == "gridlock":
        return _gridlock_strategy(obs, current_phase, time_in_phase)
    elif task_name == "emergency_vehicle":
        return _emergency_strategy(obs, current_phase, time_in_phase, emergency_handled)
    elif task_name in ("rush_hour_ns", "rush_hour_ew"):
        return _rush_hour_strategy(obs, current_phase, time_in_phase, task_name)
    else:
        return _balanced_strategy(obs, current_phase, time_in_phase, task_name)


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
    heuristic_rec = smart_heuristic(obs, obs.active_phase, obs.time_in_phase, obs.task_name)
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

    return smart_heuristic(obs, obs.active_phase, obs.time_in_phase, obs.task_name)


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
    task_name: str = "balanced",
    emergency_handled: bool = False,
) -> int:
    """
    Hybrid approach:
    - Use task-specific heuristic for most steps
    - Consult LLM at strategic intervals for tasks that benefit from it
    - Always use heuristic for emergency overrides and pattern-based tasks
    """
    params = TASK_PARAMS.get(task_name, DEFAULT_PARAMS)
    llm_interval = params["llm_interval"]
    min_hold = params["min_hold"]

    # During yellow, just hold
    if obs.yellow_remaining > 0:
        return current_phase

    # Emergency: always use heuristic (fast, deterministic)
    if obs.emergency_direction >= 0:
        return smart_heuristic(obs, current_phase, time_in_phase, task_name, emergency_handled)

    # Consult LLM at strategic intervals (only for tasks where it helps)
    if llm_interval > 0 and (step % llm_interval == 0) and time_in_phase >= min_hold:
        return get_phase_from_llm(client, obs, history)

    # Default: use task-specific heuristic
    return smart_heuristic(obs, current_phase, time_in_phase, task_name, emergency_handled)


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
        emergency_handled = False

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            phase = decide_phase(
                client, obs, history, step,
                current_phase, time_in_phase,
                task_name=task,
                emergency_handled=emergency_handled,
            )
            # Track if emergency was ever active and then cleared
            if obs.emergency_direction >= 0:
                emergency_handled = True

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
