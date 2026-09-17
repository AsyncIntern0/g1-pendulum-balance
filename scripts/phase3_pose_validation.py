"""
phase3_pose_validation.py
=========================
AXON-R Phase 3 — independent validation of an EXISTING frozen pose library.

IMPORTANT
---------
This script does NOT optimize or modify poses.

It takes an existing pose_library_v3.json, selects the frozen poses, and
tests them against fresh disturbance conditions that were not used by the
Phase-3 optimizer.

For every validation case:
    1. Estimate the natural impact time using an UNPROTECTED baseline.
    2. Run the SAME disturbance again with the frozen protective pose.
    3. Compare fall/no-fall, peak impact forces, CoM height, and joint limits.

The result "recovered=True" therefore means only:
    "this pose prevented a fall in this simulation trial."

It is NOT a hardware-safety guarantee.

Expected local files:
    phase3_pose_validation.py
    pose_library_v3.json
    g1_pendulum.xml
    generate_fall_dataset_final.py

Example:
    python phase3_pose_validation.py ^
        --model g1_pendulum.xml ^
        --library pose_library_v3.json ^
        --tests 20 ^
        --seed 42

Quick smoke test:
    python phase3_pose_validation.py ^
        --model g1_pendulum.xml ^
        --library pose_library_v3.json ^
        --tests 2 ^
        --seed 42

Output:
    phase3_validation_results.json
"""

import argparse
import csv
import importlib
import json
import math
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
import mujoco


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

TRIGGER_LEAD_S = 0.30
POST_FALL_OBSERVATION_S = 1.5

# Fresh validation conditions are intentionally different from the
# optimizer's COND_TIMINGS=[0.0,0.4] and COND_JITTERS=[-15,15].
VALIDATION_TIMING_RANGE_S = (0.10, 0.80)
VALIDATION_DIRECTION_JITTER_DEG = 30.0

# Small tolerance for reporting a joint-limit violation.
JOINT_LIMIT_TOL_RAD = 1e-4

# Used only for the optional "recovery" interpretation.
# The actual primary result is simply fell=True/False.
SETTLE_TIME_FALLBACK_S = 0.5


# ---------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------

@dataclass
class ValidationResult:
    pose_id: int
    scenario_id: int
    scenario_fn: str
    category: str
    nominal_direction: float | None

    magnitude: float
    direction_deg: float | None
    timing_phase_s: float
    natural_time_to_impact_s: float

    baseline_fell: bool
    protected_fell: bool
    recovered: bool
    triggered: bool

    baseline_peak_force_pelvis: float
    baseline_peak_force_head: float
    baseline_peak_force_other: float

    protected_peak_force_pelvis: float
    protected_peak_force_head: float
    protected_peak_force_other: float

    baseline_com_height_at_first_impact_m: float
    protected_com_height_at_first_impact_m: float

    protected_peak_joint_limit_violation_rad: float
    protected_joint_limit_violated: bool

    transition_complete: bool
    transition_time_s: float

    velocity_at_trigger_mps: float


# ---------------------------------------------------------------------
# MODULE / MODEL HELPERS
# ---------------------------------------------------------------------

def load_phase1_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import Phase-1 module '{module_name}'. "
            "Run this script from the project directory containing "
            "generate_fall_dataset_final.py, or pass --p1-module."
        ) from exc


def load_instrumented_model(model_path):
    """
    Load the same MuJoCo model style used by Phase 3 V3.

    The pendulum bob is enabled as a runtime collision proxy for the
    missing head/upper-body geometry.
    """
    model = mujoco.MjModel.from_xml_path(model_path)

    pelvis_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )
    bob_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pendulum_bob"
    )

    if pelvis_id < 0 or bob_id < 0:
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(model.nbody)
        ]
        raise RuntimeError(
            "Could not resolve required bodies 'pelvis' and "
            "'pendulum_bob'.\n"
            f"pelvis_id={pelvis_id}, pendulum_bob_id={bob_id}\n"
            f"Available bodies={names}"
        )

    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == bob_id:
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 1

    return model


MUTABLE_MODEL_FIELDS = [
    "actuator_gainprm",
    "actuator_biasprm",
    "geom_friction",
    "geom_contype",
    "geom_conaffinity",
]


def snapshot_model_state(model):
    snap = {
        name: getattr(model, name).copy()
        for name in MUTABLE_MODEL_FIELDS
    }
    snap["gravity"] = model.opt.gravity.copy()
    return snap


def restore_model_state(model, snapshot):
    for name in MUTABLE_MODEL_FIELDS:
        getattr(model, name)[:] = snapshot[name]
    model.opt.gravity[:] = snapshot["gravity"]


def impact_body_ids(model):
    pelvis_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )
    bob_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pendulum_bob"
    )
    return {"pelvis": pelvis_id, "head": bob_id}


# ---------------------------------------------------------------------
# LIBRARY HELPERS
# ---------------------------------------------------------------------

def load_library(path):
    with open(path, "r", encoding="utf-8") as f:
        library = json.load(f)

    if "poses" not in library:
        raise ValueError("Library has no 'poses' list.")
    if "lookup_keys" not in library:
        raise ValueError("Library has no 'lookup_keys' list.")

    return library


def build_lookup_index(library):
    index = {}
    for row in library["lookup_keys"]:
        index.setdefault(
            (row["scenario_id"], row["direction"]),
            row["pose_id"],
        )
    return index


def pose_for_velocity(pose_entry, velocity_mps):
    """
    Same interpolation idea used by Phase 3 Stage 4:
    linear interpolation between frozen velocity control points,
    clamped at the ends.
    """
    points = sorted(
        pose_entry["velocity_control_points"],
        key=lambda x: x["velocity_mps"],
    )

    if not points:
        raise ValueError(
            f"Pose {pose_entry['pose_id']} has no velocity control points."
        )

    if velocity_mps <= points[0]["velocity_mps"]:
        return np.asarray(points[0]["pose"], dtype=float)

    if velocity_mps >= points[-1]["velocity_mps"]:
        return np.asarray(points[-1]["pose"], dtype=float)

    for lo, hi in zip(points[:-1], points[1:]):
        vlo = float(lo["velocity_mps"])
        vhi = float(hi["velocity_mps"])

        if vlo <= velocity_mps <= vhi:
            if abs(vhi - vlo) < 1e-12:
                return np.asarray(lo["pose"], dtype=float)

            alpha = (velocity_mps - vlo) / (vhi - vlo)
            return (
                np.asarray(lo["pose"], dtype=float)
                + alpha
                * (
                    np.asarray(hi["pose"], dtype=float)
                    - np.asarray(lo["pose"], dtype=float)
                )
            )

    return np.asarray(points[-1]["pose"], dtype=float)


def choose_pose_for_validation(pose_entry, rng):
    """
    Select one of the EXISTING frozen velocity control points.

    We intentionally do not optimize anything here.

    The default validation uses a stored control point rather than
    inventing a new optimized pose.
    """
    points = pose_entry["velocity_control_points"]

    if not points:
        raise ValueError(
            f"Pose {pose_entry['pose_id']} contains no control points."
        )

    point = points[int(rng.integers(0, len(points)))]
    pose = np.asarray(point["pose"], dtype=float)

    return pose, float(point["velocity_mps"])


# ---------------------------------------------------------------------
# CONTACT / FORCE / COM / LIMIT HELPERS
# ---------------------------------------------------------------------

def contact_peak_forces(
    model,
    data,
    ground_id,
    body_ids,
    foot_geom_ids,
):
    peak = {
        "pelvis": 0.0,
        "head": 0.0,
        "other": 0.0,
    }

    for i in range(data.ncon):
        contact = data.contact[i]

        if contact.geom1 != ground_id and contact.geom2 != ground_id:
            continue

        other_geom = (
            contact.geom2 if contact.geom1 == ground_id
            else contact.geom1
        )

        if other_geom in foot_geom_ids:
            continue

        force6 = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, i, force6)
        force = float(np.linalg.norm(force6[:3]))

        body_id = model.geom_bodyid[other_geom]

        if body_id == body_ids["pelvis"]:
            peak["pelvis"] = max(peak["pelvis"], force)
        elif body_id == body_ids["head"]:
            peak["head"] = max(peak["head"], force)
        else:
            peak["other"] = max(peak["other"], force)

    return peak


def center_of_mass_height(model, data):
    """
    Root-subtree COM z coordinate.

    data.subtree_com[1] is the COM of the root body's complete subtree.
    """
    mujoco.mj_comPos(model, data)
    return float(data.subtree_com[1][2])


def joint_limit_violation(model, data):
    """
    Return:
        (maximum violation in radians, any_violation)

    The check uses the actual joint ranges from the MJCF, not actuator
    ctrlrange. Only hinge/slide joints with finite ranges are considered.
    """
    max_violation = 0.0

    for jid in range(model.njnt):
        if model.jnt_type[jid] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue

        if model.jnt_limited[jid] == 0:
            continue

        qadr = model.jnt_qposadr[jid]
        q = float(data.qpos[qadr])
        lo, hi = model.jnt_range[jid]

        violation = max(
            0.0,
            float(lo) - q,
            q - float(hi),
        )

        max_violation = max(max_violation, violation)

    return max_violation, max_violation > JOINT_LIMIT_TOL_RAD


def new_bad_ground_contact(
    model,
    data,
    ground_id,
    baseline_contacts,
    foot_geom_ids,
    p1,
):
    current = p1.snapshot_contacts(model, data, ground_id)
    return (current - baseline_contacts) - foot_geom_ids


# ---------------------------------------------------------------------
# UNPROTECTED BASELINE
# ---------------------------------------------------------------------

def run_unprotected_trial(
    model,
    p1,
    scenario,
    magnitude,
    direction_deg,
    timing_phase_s,
    snapshot,
):
    """
    Run the disturbance with the normal standing controller.

    Returns:
        dict containing natural impact time and baseline measurements.
    """
    restore_model_state(model, snapshot)

    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)

    stand_ctrl = model.key_ctrl[0].copy()
    d.ctrl[:] = stand_ctrl

    ground_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
    )
    foot_ids = p1.get_foot_geom_ids(model)
    body_ids = impact_body_ids(model)

    dt = model.opt.timestep
    settle_steps = int(p1.SETTLE_TIME / dt)

    for _ in range(settle_steps):
        d.ctrl[:] = stand_ctrl
        mujoco.mj_step(model, d)

    baseline_contacts = p1.snapshot_contacts(
        model, d, ground_id
    )

    start_com = center_of_mass_height(model, d)

    builder = p1.SCENARIO_BUILDERS[scenario[2]]
    disturb_fn = builder(
        model,
        magnitude,
        direction=direction_deg,
    )

    timing_steps = int(timing_phase_s / dt)
    max_steps = int(p1.MAX_EPISODE_TIME / dt)

    peak = {
        "pelvis": 0.0,
        "head": 0.0,
        "other": 0.0,
    }

    first_impact_time = None
    first_impact_com = float("nan")
    fell = False
    fell_step = None

    for step in range(max_steps):
        t_rel = (step - timing_steps) * dt

        d.ctrl[:] = stand_ctrl

        if step >= timing_steps:
            disturb_fn(model, d, max(t_rel, 0.0))

        mujoco.mj_step(model, d)

        forces = contact_peak_forces(
            model,
            d,
            ground_id,
            body_ids,
            foot_ids,
        )

        for key in peak:
            peak[key] = max(peak[key], forces[key])

        bad = new_bad_ground_contact(
            model,
            d,
            ground_id,
            baseline_contacts,
            foot_ids,
            p1,
        )

        if bad and not fell:
            fell = True
            fell_step = step
            first_impact_time = (
                step * dt - timing_phase_s
            )
            first_impact_com = center_of_mass_height(model, d)

        if fell and (
            (step - fell_step) * dt
            >= POST_FALL_OBSERVATION_S
        ):
            break

    return {
        "natural_time_to_impact_s": (
            float(first_impact_time)
            if first_impact_time is not None
            else float("nan")
        ),
        "fell": bool(fell),
        "peak": peak,
        "com_height_at_first_impact_m": first_impact_com,
        "start_com_height_m": start_com,
    }


# ---------------------------------------------------------------------
# PROTECTED VALIDATION TRIAL
# ---------------------------------------------------------------------

def run_protected_trial(
    model,
    p1,
    scenario,
    magnitude,
    direction_deg,
    timing_phase_s,
    pose_ctrl,
    natural_time_to_impact_s,
    snapshot,
):
    """
    Test one frozen pose against one fresh disturbance.

    The trigger time is derived from the UNPROTECTED trial of the SAME
    disturbance condition.

    This deliberately does not call Phase-3's optimizer.
    """
    restore_model_state(model, snapshot)

    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)

    stand_ctrl = model.key_ctrl[0].copy()
    d.ctrl[:] = stand_ctrl

    ground_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
    )
    foot_ids = p1.get_foot_geom_ids(model)
    body_ids = impact_body_ids(model)

    dt = model.opt.timestep
    settle_steps = int(p1.SETTLE_TIME / dt)

    for _ in range(settle_steps):
        d.ctrl[:] = stand_ctrl
        mujoco.mj_step(model, d)

    baseline_contacts = p1.snapshot_contacts(
        model, d, ground_id
    )

    if not np.isfinite(natural_time_to_impact_s):
        return {
            "triggered": False,
            "fell": False,
            "recovered": False,
            "peak": {
                "pelvis": 0.0,
                "head": 0.0,
                "other": 0.0,
            },
            "com_height_at_first_impact_m": float("nan"),
            "peak_joint_limit_violation_rad": 0.0,
            "joint_limit_violated": False,
            "transition_complete": False,
            "transition_time_s": -1.0,
            "velocity_at_trigger_mps": -1.0,
        }

    trigger_t_rel = (
        natural_time_to_impact_s - TRIGGER_LEAD_S
    )

    builder = p1.SCENARIO_BUILDERS[scenario[2]]
    disturb_fn = builder(
        model,
        magnitude,
        direction=direction_deg,
    )

    timing_steps = int(timing_phase_s / dt)
    max_steps = int(p1.MAX_EPISODE_TIME / dt)

    n_act = len(pose_ctrl)

    peak = {
        "pelvis": 0.0,
        "head": 0.0,
        "other": 0.0,
    }

    triggered = False
    trigger_step = None

    fell = False
    fell_step = None

    first_impact_com = float("nan")

    transition_complete = False
    transition_time_s = -1.0

    peak_limit_violation = 0.0
    any_limit_violation = False

    velocity_at_trigger = -1.0

    for step in range(max_steps):
        t_rel = (step - timing_steps) * dt

        if (
            step >= timing_steps
            and max(t_rel, 0.0) >= trigger_t_rel
        ):
            d.ctrl[:n_act] = pose_ctrl

            if not triggered:
                triggered = True
                trigger_step = step
                velocity_at_trigger = float(
                    np.linalg.norm(d.qvel[0:3])
                )
        else:
            d.ctrl[:] = stand_ctrl

        if step >= timing_steps:
            disturb_fn(
                model,
                d,
                max(t_rel, 0.0),
            )

        mujoco.mj_step(model, d)

        forces = contact_peak_forces(
            model,
            d,
            ground_id,
            body_ids,
            foot_ids,
        )

        for key in peak:
            peak[key] = max(peak[key], forces[key])

        limit_violation, violated = joint_limit_violation(
            model, d
        )

        peak_limit_violation = max(
            peak_limit_violation,
            limit_violation,
        )
        any_limit_violation = (
            any_limit_violation or violated
        )

        if triggered and not transition_complete:
            q_actual = d.qpos[7:7 + n_act]
            err = np.abs(q_actual - pose_ctrl)

            if np.all(err < 0.05):
                transition_complete = True
                transition_time_s = (
                    (step - trigger_step) * dt
                )

        bad = new_bad_ground_contact(
            model,
            d,
            ground_id,
            baseline_contacts,
            foot_ids,
            p1,
        )

        if bad and not fell:
            fell = True
            fell_step = step
            first_impact_com = center_of_mass_height(
                model, d
            )

        if fell and (
            (step - fell_step) * dt
            >= POST_FALL_OBSERVATION_S
        ):
            break

    recovered = (
        triggered
        and not fell
    )

    return {
        "triggered": bool(triggered),
        "fell": bool(fell),
        "recovered": bool(recovered),
        "peak": peak,
        "com_height_at_first_impact_m": first_impact_com,
        "peak_joint_limit_violation_rad": (
            float(peak_limit_violation)
        ),
        "joint_limit_violated": bool(any_limit_violation),
        "transition_complete": bool(transition_complete),
        "transition_time_s": float(transition_time_s),
        "velocity_at_trigger_mps": float(
            velocity_at_trigger
        ),
    }


# ---------------------------------------------------------------------
# VALIDATION CONDITION GENERATION
# ---------------------------------------------------------------------

def scenario_by_id(p1):
    return {
        int(s[0]): s
        for s in p1.SCENARIOS
    }


def random_condition(p1, scenario, rng):
    """
    Generate a fresh condition.

    Magnitude:
        uniformly sampled from the Phase-1 calibrated range.

    Timing:
        0.10–0.80 s, deliberately outside the optimizer's
        two-point [0.0, 0.4] timing set.

    Direction:
        nominal direction +/-30 degrees for directional
        scenarios.

    Non-directional mechanisms retain direction=None.
    """
    _, _, fn_name, nominal_dir = scenario

    lo, hi = p1.MAGNITUDE_RANGES[fn_name]
    magnitude = float(rng.uniform(lo, hi))

    timing = float(
        rng.uniform(
            VALIDATION_TIMING_RANGE_S[0],
            VALIDATION_TIMING_RANGE_S[1],
        )
    )

    if nominal_dir is None:
        direction = None
    else:
        direction = (
            float(nominal_dir)
            + float(
                rng.uniform(
                    -VALIDATION_DIRECTION_JITTER_DEG,
                    VALIDATION_DIRECTION_JITTER_DEG,
                )
            )
        ) % 360.0

    return magnitude, direction, timing


# ---------------------------------------------------------------------
# MAIN VALIDATION LOOP
# ---------------------------------------------------------------------

def validate_pose(
    model,
    p1,
    pose_entry,
    scenario,
    tests,
    rng,
    snapshot,
):
    results = []

    for trial_index in range(tests):
        magnitude, direction, timing = random_condition(
            p1,
            scenario,
            rng,
        )

        baseline = run_unprotected_trial(
            model,
            p1,
            scenario,
            magnitude,
            direction,
            timing,
            snapshot,
        )

        if not baseline["fell"]:
            # This condition does not provide a meaningful
            # protected-vs-baseline fall comparison.
            print(
                f"    test {trial_index + 1:>3}: "
                "baseline did not fall -> excluded"
            )
            continue

        # Use the frozen velocity control point whose stored
        # velocity is closest to the velocity observed in the
        # baseline condition. This is Stage-4-style lookup without
        # optimization.
        baseline_velocity = estimate_baseline_trigger_velocity(
            model,
            p1,
            scenario,
            magnitude,
            direction,
            timing,
            baseline["natural_time_to_impact_s"],
            snapshot,
        )

        pose = pose_for_velocity(
            pose_entry,
            baseline_velocity,
        )

        protected = run_protected_trial(
            model,
            p1,
            scenario,
            magnitude,
            direction,
            timing,
            pose,
            baseline["natural_time_to_impact_s"],
            snapshot,
        )

        result = ValidationResult(
            pose_id=int(pose_entry["pose_id"]),
            scenario_id=int(scenario[0]),
            scenario_fn=str(scenario[2]),
            category=str(scenario[1]),
            nominal_direction=(
                None
                if scenario[3] is None
                else float(scenario[3])
            ),

            magnitude=float(magnitude),
            direction_deg=(
                None
                if direction is None
                else float(direction)
            ),
            timing_phase_s=float(timing),
            natural_time_to_impact_s=float(
                baseline["natural_time_to_impact_s"]
            ),

            baseline_fell=bool(baseline["fell"]),
            protected_fell=bool(protected["fell"]),
            recovered=bool(protected["recovered"]),
            triggered=bool(protected["triggered"]),

            baseline_peak_force_pelvis=float(
                baseline["peak"]["pelvis"]
            ),
            baseline_peak_force_head=float(
                baseline["peak"]["head"]
            ),
            baseline_peak_force_other=float(
                baseline["peak"]["other"]
            ),

            protected_peak_force_pelvis=float(
                protected["peak"]["pelvis"]
            ),
            protected_peak_force_head=float(
                protected["peak"]["head"]
            ),
            protected_peak_force_other=float(
                protected["peak"]["other"]
            ),

            baseline_com_height_at_first_impact_m=float(
                baseline["com_height_at_first_impact_m"]
            ),
            protected_com_height_at_first_impact_m=float(
                protected["com_height_at_first_impact_m"]
            ),

            protected_peak_joint_limit_violation_rad=float(
                protected["peak_joint_limit_violation_rad"]
            ),
            protected_joint_limit_violated=bool(
                protected["joint_limit_violated"]
            ),

            transition_complete=bool(
                protected["transition_complete"]
            ),
            transition_time_s=float(
                protected["transition_time_s"]
            ),

            velocity_at_trigger_mps=float(
                protected["velocity_at_trigger_mps"]
            ),
        )

        results.append(result)

        status = "RECOVERED" if result.recovered else "FELL"
        print(
            f"    test {trial_index + 1:>3}: "
            f"{status:<9} "
            f"mag={magnitude:.3f} "
            f"dir={direction if direction is not None else 'NA'} "
            f"t={timing:.3f}s "
            f"F(head)={result.protected_peak_force_head:.1f}N"
        )

    return results


def estimate_baseline_trigger_velocity(
    model,
    p1,
    scenario,
    magnitude,
    direction_deg,
    timing_phase_s,
    natural_time_to_impact_s,
    snapshot,
):
    """
    Measure the same proxy used by Phase 3:
        ||qvel[0:3]||

    at the trigger instant.

    This is deliberately a separate baseline replay so the validation
    uses the velocity estimate corresponding to the SAME fresh
    disturbance condition.
    """
    restore_model_state(model, snapshot)

    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)

    stand_ctrl = model.key_ctrl[0].copy()
    d.ctrl[:] = stand_ctrl

    dt = model.opt.timestep
    settle_steps = int(p1.SETTLE_TIME / dt)

    for _ in range(settle_steps):
        d.ctrl[:] = stand_ctrl
        mujoco.mj_step(model, d)

    builder = p1.SCENARIO_BUILDERS[scenario[2]]
    disturb_fn = builder(
        model,
        magnitude,
        direction=direction_deg,
    )

    trigger_t_rel = (
        natural_time_to_impact_s
        - TRIGGER_LEAD_S
    )

    timing_steps = int(timing_phase_s / dt)
    max_steps = int(p1.MAX_EPISODE_TIME / dt)

    for step in range(max_steps):
        t_rel = (step - timing_steps) * dt

        d.ctrl[:] = stand_ctrl

        if step >= timing_steps:
            disturb_fn(
                model,
                d,
                max(t_rel, 0.0),
            )

        if (
            step >= timing_steps
            and max(t_rel, 0.0) >= trigger_t_rel
        ):
            return float(np.linalg.norm(d.qvel[0:3]))

        mujoco.mj_step(model, d)

    return 0.0


# ---------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------

def summarize(results):
    if not results:
        return {
            "tests_with_natural_baseline_fall": 0,
            "recovery_rate": None,
            "fall_rate": None,
            "mean_head_force_protected_N": None,
            "mean_head_force_baseline_N": None,
            "mean_pelvis_force_protected_N": None,
            "mean_pelvis_force_baseline_N": None,
            "mean_other_force_protected_N": None,
            "mean_com_height_at_impact_protected_m": None,
            "mean_joint_limit_violation_rad": None,
            "joint_limit_violation_rate": None,
            "transition_completion_rate": None,
            "mean_transition_time_s": None,
        }

    n = len(results)
    recovered = sum(r.recovered for r in results)
    fell = sum(r.protected_fell for r in results)
    limit_violations = sum(
        r.protected_joint_limit_violated
        for r in results
    )
    transition_complete = sum(
        r.transition_complete
        for r in results
    )

    def mean(field):
        values = [
            getattr(r, field)
            for r in results
            if np.isfinite(getattr(r, field))
        ]
        return (
            float(np.mean(values))
            if values else None
        )

    return {
        "tests_with_natural_baseline_fall": n,
        "recovery_rate": float(recovered / n),
        "fall_rate": float(fell / n),

        "mean_head_force_protected_N": mean(
            "protected_peak_force_head"
        ),
        "mean_head_force_baseline_N": mean(
            "baseline_peak_force_head"
        ),

        "mean_pelvis_force_protected_N": mean(
            "protected_peak_force_pelvis"
        ),
        "mean_pelvis_force_baseline_N": mean(
            "baseline_peak_force_pelvis"
        ),

        "mean_other_force_protected_N": mean(
            "protected_peak_force_other"
        ),

        "mean_com_height_at_impact_protected_m": mean(
            "protected_com_height_at_first_impact_m"
        ),

        "mean_joint_limit_violation_rad": mean(
            "protected_peak_joint_limit_violation_rad"
        ),
        "joint_limit_violation_rate": float(
            limit_violations / n
        ),

        "transition_completion_rate": float(
            transition_complete / n
        ),
        "mean_transition_time_s": mean(
            "transition_time_s"
        ),
    }


def save_json(path, all_results, summary, metadata):
    payload = {
        "metadata": metadata,
        "summary": summary,
        "trials": [
            asdict(r)
            for r in all_results
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            allow_nan=False,
        )


def save_csv(path, results):
    if not results:
        return

    rows = [asdict(r) for r in results]
    fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Independently validate a frozen Phase-3 protective "
            "pose library against fresh MuJoCo disturbances."
        )
    )

    ap.add_argument(
        "--model",
        required=True,
        help="Path to g1_pendulum.xml",
    )
    ap.add_argument(
        "--library",
        required=True,
        help="Existing pose_library_v3.json",
    )
    ap.add_argument(
        "--p1-module",
        default="generate_fall_dataset_final",
        help="Phase-1 Python module name",
    )
    ap.add_argument(
        "--tests",
        type=int,
        default=20,
        help="Fresh disturbance trials per pose/scenario pair",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fresh validation conditions",
    )
    ap.add_argument(
        "--pose-id",
        type=int,
        default=None,
        help="Validate only this pose_id",
    )
    ap.add_argument(
        "--scenario-id",
        type=int,
        default=None,
        help="Validate only this scenario_id",
    )
    ap.add_argument(
        "--out",
        default="phase3_validation_results.json",
        help="JSON output path",
    )
    ap.add_argument(
        "--csv-out",
        default="phase3_validation_trials.csv",
        help="CSV trial-level output path",
    )

    args = ap.parse_args()

    if args.tests <= 0:
        raise ValueError("--tests must be > 0")

    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("AXON-R PHASE 3 — INDEPENDENT POSE VALIDATION")
    print("=" * 72)
    print("IMPORTANT: poses are frozen. No optimization is performed.")
    print()

    p1 = load_phase1_module(args.p1_module)
    library = load_library(args.library)

    model = load_instrumented_model(args.model)
    snapshot = snapshot_model_state(model)

    scenarios_by_id = scenario_by_id(p1)

    poses = library["poses"]

    if args.pose_id is not None:
        poses = [
            p for p in poses
            if int(p["pose_id"]) == args.pose_id
        ]

        if not poses:
            raise ValueError(
                f"pose_id {args.pose_id} not found in library."
            )

    all_results = []
    summary_by_pose = {}

    for pose_entry in poses:
        pose_id = int(pose_entry["pose_id"])

        scenario_ids = [
            int(x)
            for x in pose_entry.get(
                "covers_scenario_ids", []
            )
        ]

        if args.scenario_id is not None:
            scenario_ids = [
                x for x in scenario_ids
                if x == args.scenario_id
            ]

        if not scenario_ids:
            print(
                f"\nPose {pose_id}: no matching scenarios -> skipped"
            )
            continue

        print()
        print("-" * 72)
        print(
            f"POSE {pose_id} | "
            f"covered scenarios={scenario_ids}"
        )
        print("-" * 72)

        pose_results = []

        for sid in scenario_ids:
            if sid not in scenarios_by_id:
                print(
                    f"  Scenario {sid}: not found in Phase-1 "
                    "SCENARIOS -> skipped"
                )
                continue

            scenario = scenarios_by_id[sid]

            print(
                f"\n  Scenario {sid}: "
                f"{scenario[2]} "
                f"(category={scenario[1]})"
            )

            results = validate_pose(
                model=model,
                p1=p1,
                pose_entry=pose_entry,
                scenario=scenario,
                tests=args.tests,
                rng=rng,
                snapshot=snapshot,
            )

            pose_results.extend(results)
            all_results.extend(results)

        summary_by_pose[str(pose_id)] = summarize(
            pose_results
        )

        s = summary_by_pose[str(pose_id)]

        print()
        print(
            f"  Pose {pose_id} summary:"
        )
        print(
            f"    valid baseline-fall tests : "
            f"{s['tests_with_natural_baseline_fall']}"
        )
        print(
            f"    recovery rate             : "
            f"{s['recovery_rate']}"
        )
        print(
            f"    protected fall rate       : "
            f"{s['fall_rate']}"
        )
        print(
            f"    mean protected head force : "
            f"{s['mean_head_force_protected_N']} N"
        )
        print(
            f"    joint-limit violation    : "
            f"{s['joint_limit_violation_rate']}"
        )
        print(
            f"    transition completion     : "
            f"{s['transition_completion_rate']}"
        )

    overall = summarize(all_results)

    metadata = {
        "validator": "phase3_pose_validation.py",
        "purpose": (
            "Independent validation of frozen Phase-3 poses "
            "using fresh disturbance conditions."
        ),
        "library": os.path.abspath(args.library),
        "model": os.path.abspath(args.model),
        "tests_requested_per_pose_scenario": args.tests,
        "random_seed": args.seed,
        "trigger_lead_s": TRIGGER_LEAD_S,
        "post_fall_observation_s": POST_FALL_OBSERVATION_S,
        "validation_timing_range_s": list(
            VALIDATION_TIMING_RANGE_S
        ),
        "validation_direction_jitter_deg": (
            VALIDATION_DIRECTION_JITTER_DEG
        ),
        "important_interpretation": (
            "recovered=True means the frozen pose prevented "
            "a detected non-foot ground contact in this MuJoCo "
            "validation trial. It is not evidence of hardware "
            "safety."
        ),
    }

    save_json(
        args.out,
        all_results,
        {
            "overall": overall,
            "by_pose": summary_by_pose,
        },
        metadata,
    )

    save_csv(
        args.csv_out,
        all_results,
    )

    print()
    print("=" * 72)
    print("VALIDATION COMPLETE")
    print("=" * 72)
    print(
        f"Valid baseline-fall trials : "
        f"{overall['tests_with_natural_baseline_fall']}"
    )
    print(
        f"Overall recovery rate      : "
        f"{overall['recovery_rate']}"
    )
    print(
        f"Overall protected fall rate: "
        f"{overall['fall_rate']}"
    )
    print(
        f"Results JSON               : {args.out}"
    )
    print(
        f"Trial CSV                  : {args.csv_out}"
    )


if __name__ == "__main__":
    main()
