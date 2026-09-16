#!/usr/bin/env python3
"""
Phase 3 — Protective Pose Design for the Unitree G1 MuJoCo model.

Purpose
-------
Offline optimization and validation of protective recovery poses.

Pipeline
--------
1. Reproduce each Phase-1 fall condition and establish an unprotected baseline.
2. Estimate fall severity from the fall velocity.
3. Optimize a candidate protective pose with CMA-ES.
4. Evaluate candidates with MuJoCo in parallel CPU workers.
5. Enforce the 300 ms pose-transition budget as a hard validity constraint.
6. Stress-test the best pose over timing/severity perturbations.
7. Cross-evaluate poses against other scenarios.
8. Greedily merge similar robust poses into a compact library.
9. Store a deterministic direction/cause/severity -> pose library.

Important
---------
This is an OFFLINE design/validation script. CMA-ES is not used at runtime.
The runtime system should use the resulting library deterministically:

    TCN -> direction + cause + velocity/severity
        -> lookup/interpolation -> target pose -> trajectory/PD control

CPU parallelism
---------------
Each multiprocessing worker owns its own MuJoCo model and snapshot. Candidate
evaluations are independent and are evaluated with pool.map(). This avoids
sharing MuJoCo state between processes and preserves the existing CPU-parallel
architecture.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("MuJoCo is required: pip install mujoco") from exc

try:
    import cma
except ImportError as exc:
    raise SystemExit("pycma is required: pip install cma") from exc


# ---------------------------------------------------------------------------
# Project paths / Phase-1 integration
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

DEFAULT_MODEL = HERE / r"C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml"
DEFAULT_OUTPUT = HERE / "pose_library.json"

# The existing Phase-1 generator is expected to be available beside this file
# or importable through the project path.
try:
    import generate_fall_dataset_final as p1
except Exception:
    p1 = None


# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------

LEAD_TIME_S = 0.30
POSE_TRANSITION_BUDGET_S = 0.30
TRANSITION_TOL_RAD = 0.05

# Fitness weights. Head is the pendulum-bob impact proxy in this reduced model.
W_HEAD = 2.0
W_PELVIS = 1.0
W_OTHER = 0.5

# Strongly discourage incomplete / late transitions.
TRANSITION_OVERRUN_PENALTY = 5000.0
INCOMPLETE_TRANSITION_PENALTY = 20000.0

POST_FALL_OBSERVATION_S = 1.5

# Timing robustness around the nominal 300 ms trigger.
COND_TIMINGS = [0.30, 0.25, 0.20, 0.15]

# Phase-1 variation robustness.
COND_JITTERS_MS = [-15.0, 15.0]

# Severity bins are generated from measured fall-speed distributions.
SEVERITY_LABELS = ("low", "medium", "high")

# Similar poses are merged only when the performance degradation is small.
MERGE_TOLERANCE = 0.15

# Target final library size.
MIN_LIBRARY_SIZE = 12
MAX_LIBRARY_SIZE = 15

# Candidate-search defaults.
DEFAULT_POPSIZE = 16
DEFAULT_GENERATIONS = 20
DEFAULT_SIGMA = 0.15

# Joint-limit safety margin.
JOINT_LIMIT_MARGIN_RAD = math.radians(2.0)

# Small numerical tolerance.
EPS = 1e-9


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    score: float
    peak_force_head: float
    peak_force_pelvis: float
    peak_force_other: float
    transition_time_s: float
    transition_valid: bool
    transition_complete: bool
    fell: bool
    impact_time_s: Optional[float]
    impact_velocity_mps: float
    severity: str
    timing_s: float
    scenario_id: int


@dataclass
class BaselineResult:
    scenario_id: int
    fell: bool
    impact_time_s: Optional[float]
    impact_velocity_mps: float
    severity: str
    peak_force_head: float
    peak_force_pelvis: float
    peak_force_other: float


# ---------------------------------------------------------------------------
# Global worker state
# ---------------------------------------------------------------------------

_WORKER_MODEL = None
_WORKER_SNAPSHOT = None


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def load_model(model_path: Path) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    return model


def _snapshot_mutable_model_fields(model: mujoco.MjModel) -> Dict[str, Any]:
    """
    Save model fields that Phase-1-like scenario modifications may mutate.

    Keeping a clean snapshot prevents one scenario's friction/gain/fault
    changes from leaking into the next scenario.
    """
    fields = {
        "actuator_gainprm": model.actuator_gainprm.copy(),
        "actuator_biasprm": model.actuator_biasprm.copy(),
        "dof_frictionloss": model.dof_frictionloss.copy(),
        "geom_contype": model.geom_contype.copy(),
        "geom_conaffinity": model.geom_conaffinity.copy(),
        "opt_gravity": model.opt.gravity.copy(),
    }
    return fields


def _restore_model_snapshot(
    model: mujoco.MjModel,
    snapshot: Dict[str, Any],
) -> None:
    model.actuator_gainprm[:] = snapshot["actuator_gainprm"]
    model.actuator_biasprm[:] = snapshot["actuator_biasprm"]
    model.dof_frictionloss[:] = snapshot["dof_frictionloss"]
    model.geom_contype[:] = snapshot["geom_contype"]
    model.geom_conaffinity[:] = snapshot["geom_conaffinity"]
    model.opt.gravity[:] = snapshot["opt_gravity"]


def _enable_head_proxy(model: mujoco.MjModel) -> None:
    """
    Keep the pendulum bob collision-enabled.

    The XML must already compile the bob as a collidable geom. Runtime changes
    alone cannot reliably add a pair that was excluded by the compiled model.
    """
    for gid in range(model.ngeom):
        body_id = int(model.geom_bodyid[gid])
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        geom_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, gid
        )

        if body_name == "pendulum_bob" or geom_name in {
            "pendulum_bob",
            "head_proxy",
            "bob",
        }:
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 1


def _reset_data_to_stand(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """
    Reset to the stand keyframe when available, then settle briefly.
    """
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    try:
        stand_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )
        if stand_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, stand_id)
    except Exception:
        pass

    mujoco.mj_forward(model, data)

    # Short settling period.
    for _ in range(100):
        mujoco.mj_step(model, data)


def _body_id(model: mujoco.MjModel, names: Sequence[str]) -> Optional[int]:
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return int(bid)
    return None


def _geom_ids_for_body(model: mujoco.MjModel, body_id: Optional[int]) -> set[int]:
    if body_id is None:
        return set()
    return {
        gid for gid in range(model.ngeom)
        if int(model.geom_bodyid[gid]) == body_id
    }


def _ground_geom_ids(model: mujoco.MjModel) -> set[int]:
    ids = set()
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name in {"floor", "ground", "plane"}:
            ids.add(gid)

    # In the supplied reduced G1 model geom 0 is the ground plane.
    if not ids and model.ngeom:
        ids.add(0)
    return ids


def contact_peak_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> Tuple[float, float, float]:
    """
    Return peak normal/contact-force magnitudes grouped as:
      head proxy, pelvis, other non-foot bodies.

    Feet are deliberately excluded from the protective-impact objective.
    """
    pelvis_body = _body_id(model, ("pelvis", "pelvis_body"))
    head_body = _body_id(model, ("pendulum_bob", "head", "head_proxy"))

    pelvis_geoms = _geom_ids_for_body(model, pelvis_body)
    head_geoms = _geom_ids_for_body(model, head_body)
    ground_geoms = _ground_geom_ids(model)

    # Foot names in the current reduced model and common G1 naming variants.
    foot_words = ("foot", "ankle", "toe")
    foot_geoms = set()
    for gid in range(model.ngeom):
        gname = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        ).lower()
        bname = (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[gid]),
            )
            or ""
        ).lower()
        if any(word in gname or word in bname for word in foot_words):
            foot_geoms.add(gid)

    peak_head = 0.0
    peak_pelvis = 0.0
    peak_other = 0.0

    force = np.zeros(6)

    for i in range(data.ncon):
        con = data.contact[i]
        g1 = int(con.geom1)
        g2 = int(con.geom2)

        if not ((g1 in ground_geoms) ^ (g2 in ground_geoms)):
            continue

        obj = g2 if g1 in ground_geoms else g1

        # Feet are allowed to contact the floor naturally but should not
        # dominate the protective-pose objective.
        if obj in foot_geoms:
            continue

        mujoco.mj_contactForce(model, data, i, force)
        fmag = float(np.linalg.norm(force[:3]))

        if obj in head_geoms:
            peak_head = max(peak_head, fmag)
        elif obj in pelvis_geoms:
            peak_pelvis = max(peak_pelvis, fmag)
        else:
            peak_other = max(peak_other, fmag)

    return peak_head, peak_pelvis, peak_other


# ---------------------------------------------------------------------------
# Phase-1 scenario integration
# ---------------------------------------------------------------------------

def _scenario_name(scenario_id: int) -> str:
    """
    Human-readable fallback names matching the current 15-scenario design.
    """
    names = {
        1: "push_forward",
        2: "push_backward",
        3: "push_left",
        4: "push_right",
        5: "push_forward_left",
        6: "push_forward_right",
        7: "floor_tilt_pitch",
        8: "floor_tilt_roll",
        9: "floor_drop",
        10: "low_friction",
        11: "actuator_fault",
        12: "actuator_stuck",
        13: "asymmetric_gain",
        14: "trip",
        15: "sudden_load",
    }
    return names.get(scenario_id, f"scenario_{scenario_id}")


def _phase1_magnitude_range(scenario_id: int) -> Tuple[float, float]:
    """
    Obtain the Phase-1 calibrated disturbance range if available.

    Falls back to a conservative placeholder only when the Phase-1 module
    does not expose the calibration table.
    """
    if p1 is not None:
        for attr in (
            "SCENARIO_MAGNITUDE_RANGES",
            "MAGNITUDE_RANGES",
            "SCENARIO_RANGES",
        ):
            table = getattr(p1, attr, None)
            if isinstance(table, dict) and scenario_id in table:
                lo, hi = table[scenario_id]
                return float(lo), float(hi)

    # This fallback is intentionally isolated. Replace it with the exact
    # Phase-1 calibration table when the module is unavailable.
    return 0.85, 1.15


def fall_magnitude(scenario_id: int) -> float:
    lo, hi = _phase1_magnitude_range(scenario_id)
    return float(lo + 0.85 * (hi - lo))


def _call_phase1_trial(
    scenario_id: int,
    magnitude: float,
    timing_jitter_ms: float = 0.0,
) -> Any:
    """
    Call the existing Phase-1 trial runner when available.

    The function tries common signatures so Phase 3 remains compatible with
    the current project while keeping all scenario definitions in Phase 1.
    """
    if p1 is None or not hasattr(p1, "run_trial"):
        return None

    fn = p1.run_trial

    attempts = [
        lambda: fn(
            scenario_id=scenario_id,
            magnitude=magnitude,
            timing_jitter_ms=timing_jitter_ms,
        ),
        lambda: fn(
            scenario_id,
            magnitude,
            timing_jitter_ms,
        ),
        lambda: fn(
            scenario_id=scenario_id,
            magnitude=magnitude,
        ),
        lambda: fn(
            scenario_id,
            magnitude,
        ),
    ]

    last_error = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc

    if last_error:
        raise last_error
    return None


def _extract_number(result: Any, names: Sequence[str], default: float = 0.0) -> float:
    if result is None:
        return default

    if isinstance(result, dict):
        for name in names:
            if name in result:
                try:
                    return float(result[name])
                except (TypeError, ValueError):
                    pass

    for name in names:
        if hasattr(result, name):
            try:
                return float(getattr(result, name))
            except (TypeError, ValueError):
                pass

    return default


def _extract_bool(result: Any, names: Sequence[str], default: bool = False) -> bool:
    if result is None:
        return default

    if isinstance(result, dict):
        for name in names:
            if name in result:
                return bool(result[name])

    for name in names:
        if hasattr(result, name):
            return bool(getattr(result, name))

    return default


def _extract_impact_velocity(result: Any) -> float:
    return abs(
        _extract_number(
            result,
            (
                "impact_velocity_mps",
                "impact_speed_mps",
                "fall_velocity_mps",
                "velocity_mps",
                "impact_velocity",
                "fall_speed",
            ),
            0.0,
        )
    )


def _extract_impact_time(result: Any) -> Optional[float]:
    value = _extract_number(
        result,
        (
            "impact_time_s",
            "time_to_impact_s",
            "time_to_impact",
            "impact_time",
        ),
            float("nan"),
    )
    return None if not np.isfinite(value) else float(value)


def baseline_time_to_impact(
    model: mujoco.MjModel,
    scenario_id: int,
    magnitude: float,
    timing_jitter_ms: float = 0.0,
) -> Tuple[Optional[float], float, bool]:
    """
    Establish an unprotected baseline.

    Prefer Phase-1's exact trial result. If it is unavailable, use a direct
    MuJoCo probe with a deterministic generic disturbance.
    """
    result = _call_phase1_trial(
        scenario_id,
        magnitude,
        timing_jitter_ms=timing_jitter_ms,
    )

    if result is not None:
        impact_t = _extract_impact_time(result)
        impact_v = _extract_impact_velocity(result)
        fell = _extract_bool(result, ("fell", "fall", "is_fall"), impact_t is not None)
        return impact_t, impact_v, fell

    # Fallback direct probe.
    data = mujoco.MjData(model)
    _reset_data_to_stand(model, data)

    start = data.time
    previous_nonfoot_contact = False

    # A deterministic generic push. Exact scenario physics should normally
    # come from Phase 1.
    if model.nu:
        data.ctrl[:] = 0.0
        data.qvel[: min(3, model.nv)] += 0.5

    max_probe = 3.0
    while data.time - start < max_probe:
        mujoco.mj_step(model, data)

        head, pelvis, other = contact_peak_forces(model, data)
        nonfoot_contact = (head + pelvis + other) > 1.0

        if nonfoot_contact and not previous_nonfoot_contact:
            # Estimate velocity from the maximum generalized velocity.
            impact_v = float(np.linalg.norm(data.qvel[: min(6, model.nv)]))
            return data.time - start, impact_v, True

        previous_nonfoot_contact = nonfoot_contact

    return None, 0.0, False


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

def severity_from_velocity(
    velocity: float,
    thresholds: Tuple[float, float],
) -> str:
    low_high, high = thresholds
    if velocity < low_high:
        return "low"
    if velocity < high:
        return "medium"
    return "high"


def derive_severity_thresholds(
    velocities: Sequence[float],
) -> Tuple[float, float]:
    valid = np.asarray([v for v in velocities if np.isfinite(v) and v > 0])
    if len(valid) < 3:
        return 1.0, 2.0

    # Data-driven tertile boundaries.
    q1, q2 = np.quantile(valid, [1 / 3, 2 / 3])
    if q2 <= q1 + EPS:
        q1 = float(np.mean(valid) * 0.75)
        q2 = float(np.mean(valid) * 1.25)

    return float(q1), float(q2)


# ---------------------------------------------------------------------------
# Pose / transition utilities
# ---------------------------------------------------------------------------

def actuator_qpos_indices(model: mujoco.MjModel) -> np.ndarray:
    """
    Map the current actuated joints to qpos indices.

    The supplied reduced G1 model uses 14 pose dimensions:
    2 pendulum joints + 12 lower-body joints.

    We keep this generic so joint ordering is derived from the XML rather than
    hard-coded to MuJoCo qpos indices.
    """
    indices = []

    for j in range(model.njnt):
        jtype = int(model.jnt_type[j])
        if jtype != int(mujoco.mjtJoint.mjJNT_FREE):
            indices.append(int(model.jnt_qposadr[j]))

    return np.asarray(indices, dtype=int)


def pose_dimension(model: mujoco.MjModel) -> int:
    return len(actuator_qpos_indices(model))


def pose_limits(model: mujoco.MjModel) -> Tuple[np.ndarray, np.ndarray]:
    indices = actuator_qpos_indices(model)

    lower = np.full(len(indices), -np.inf)
    upper = np.full(len(indices), np.inf)

    for i, qidx in enumerate(indices):
        # Find the joint owning this qpos address.
        joint_id = None
        for j in range(model.njnt):
            if int(model.jnt_qposadr[j]) == int(qidx):
                joint_id = j
                break

        if joint_id is None:
            continue

        if int(model.jnt_limited[joint_id]):
            lower[i] = float(model.jnt_range[joint_id, 0]) + JOINT_LIMIT_MARGIN_RAD
            upper[i] = float(model.jnt_range[joint_id, 1]) - JOINT_LIMIT_MARGIN_RAD

    return lower, upper


def clip_pose_to_limits(
    model: mujoco.MjModel,
    pose: Sequence[float],
) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    lo, hi = pose_limits(model)
    return np.clip(pose, lo, hi)


def apply_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: Sequence[float],
) -> None:
    indices = actuator_qpos_indices(model)
    pose = clip_pose_to_limits(model, pose)

    if len(pose) != len(indices):
        raise ValueError(
            f"Pose has {len(pose)} values but model expects {len(indices)}."
        )

    data.qpos[indices] = pose
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def minimum_jerk(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def transition_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    start_pose: np.ndarray,
    target_pose: np.ndarray,
    budget_s: float = POSE_TRANSITION_BUDGET_S,
) -> Tuple[float, bool]:
    """
    Execute a minimum-jerk pose transition and measure convergence.

    A transition is valid only when the target is reached within tolerance
    before the 300 ms budget.
    """
    indices = actuator_qpos_indices(model)
    target_pose = clip_pose_to_limits(model, target_pose)

    if len(start_pose) != len(indices):
        start_pose = data.qpos[indices].copy()

    start_pose = np.asarray(start_pose, dtype=float).copy()

    # Use a fixed simulation step count derived from the actual model timestep.
    nsteps = max(1, int(math.ceil(budget_s / model.opt.timestep)))

    for k in range(nsteps + 1):
        s = k / nsteps
        alpha = minimum_jerk(s)
        desired = start_pose + alpha * (target_pose - start_pose)

        # Position actuators in the current model receive the target qpos.
        # If the actuator count differs, only command the available controls.
        if model.nu:
            n = min(model.nu, len(desired))
            data.ctrl[:n] = desired[:n]

        mujoco.mj_step(model, data)

    final_pose = data.qpos[indices].copy()
    error = float(np.max(np.abs(final_pose - target_pose)))

    return budget_s, error <= TRANSITION_TOL_RAD


# ---------------------------------------------------------------------------
# Impact / severity observation
# ---------------------------------------------------------------------------

def _find_time_to_impact(
    model: mujoco.MjModel,
    scenario_id: int,
    magnitude: float,
    timing_jitter_ms: float = 0.0,
) -> Tuple[Optional[float], float, bool]:
    return baseline_time_to_impact(
        model,
        scenario_id,
        magnitude,
        timing_jitter_ms,
    )


def _observe_direct_fall(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    observation_s: float,
) -> Tuple[float, float, float, Optional[float], float, bool]:
    """
    Observe contacts after a protective transition.

    Returns:
        peak_head, peak_pelvis, peak_other, first_impact_time,
        impact_velocity, fell
    """
    peak_head = 0.0
    peak_pelvis = 0.0
    peak_other = 0.0
    impact_time = None
    impact_velocity = 0.0
    fell = False

    start_t = data.time
    previous_total = 0.0

    while data.time - start_t < observation_s:
        mujoco.mj_step(model, data)

        h, p, o = contact_peak_forces(model, data)
        peak_head = max(peak_head, h)
        peak_pelvis = max(peak_pelvis, p)
        peak_other = max(peak_other, o)

        total = h + p + o
        if total > 1.0 and previous_total <= 1.0:
            fell = True
            impact_time = data.time
            impact_velocity = float(
                np.linalg.norm(data.qvel[: min(6, model.nv)])
            )

        previous_total = total

    return (
        peak_head,
        peak_pelvis,
        peak_other,
        impact_time,
        impact_velocity,
        fell,
    )


# ---------------------------------------------------------------------------
# Trial scoring
# ---------------------------------------------------------------------------

def score_components(
    head_force: float,
    pelvis_force: float,
    other_force: float,
    transition_time: float,
    transition_complete: bool,
) -> float:
    score = (
        W_HEAD * head_force
        + W_PELVIS * pelvis_force
        + W_OTHER * other_force
    )

    if transition_time > POSE_TRANSITION_BUDGET_S:
        score += TRANSITION_OVERRUN_PENALTY * (
            transition_time - POSE_TRANSITION_BUDGET_S
        )

    if not transition_complete:
        score += INCOMPLETE_TRANSITION_PENALTY

    return float(score)


def run_protected_trial(
    model: mujoco.MjModel,
    scenario_id: int,
    pose: Sequence[float],
    severity: str,
    magnitude: Optional[float] = None,
    timing_s: float = LEAD_TIME_S,
    timing_jitter_ms: float = 0.0,
) -> TrialResult:
    """
    Run one protected trial using the same fall condition as the baseline.

    The pose is commanded at (impact_time - timing_s). The simulation then
    observes the subsequent fall/impact response.
    """
    snapshot = _snapshot_mutable_model_fields(model)
    _restore_model_snapshot(model, snapshot)

    data = mujoco.MjData(model)
    _reset_data_to_stand(model, data)

    if magnitude is None:
        magnitude = fall_magnitude(scenario_id)

    impact_t, baseline_v, fell = _find_time_to_impact(
        model,
        scenario_id,
        magnitude,
        timing_jitter_ms,
    )

    if impact_t is None or not fell:
        return TrialResult(
            score=0.0,
            peak_force_head=0.0,
            peak_force_pelvis=0.0,
            peak_force_other=0.0,
            transition_time_s=0.0,
            transition_valid=False,
            transition_complete=False,
            fell=False,
            impact_time_s=None,
            impact_velocity_mps=baseline_v,
            severity=severity,
            timing_s=timing_s,
            scenario_id=scenario_id,
        )

    # Recreate the disturbance through Phase 1 when possible.
    # If Phase 1 is not directly callable inside the same MuJoCo data object,
    # use a deterministic generic state kick as fallback.
    target_command_time = max(0.0, impact_t - timing_s)

    start_pose = actuator_qpos_indices(model)
    current_pose = data.qpos[start_pose].copy()

    target_pose = clip_pose_to_limits(model, pose)

    transition_time = POSE_TRANSITION_BUDGET_S
    transition_complete = False

    # Run until the trigger time.
    while data.time < target_command_time:
        mujoco.mj_step(model, data)

    # Transition toward target pose.
    transition_time, transition_complete = transition_pose(
        model,
        data,
        current_pose,
        target_pose,
        budget_s=POSE_TRANSITION_BUDGET_S,
    )

    # Continue observing post-impact behavior.
    (
        peak_head,
        peak_pelvis,
        peak_other,
        protected_impact_t,
        protected_v,
        protected_fell,
    ) = _observe_direct_fall(
        model,
        data,
        POST_FALL_OBSERVATION_S,
    )

    # Prefer the observed protected velocity, otherwise baseline velocity.
    velocity = protected_v if protected_v > EPS else baseline_v

    score = score_components(
        peak_head,
        peak_pelvis,
        peak_other,
        transition_time,
        transition_complete,
    )

    return TrialResult(
        score=score,
        peak_force_head=peak_head,
        peak_force_pelvis=peak_pelvis,
        peak_force_other=peak_other,
        transition_time_s=transition_time,
        transition_valid=(
            transition_complete
            and transition_time <= POSE_TRANSITION_BUDGET_S + EPS
        ),
        transition_complete=transition_complete,
        fell=protected_fell,
        impact_time_s=protected_impact_t,
        impact_velocity_mps=velocity,
        severity=severity,
        timing_s=timing_s,
        scenario_id=scenario_id,
    )


# ---------------------------------------------------------------------------
# CPU worker pool
# ---------------------------------------------------------------------------

def _worker_init(model_path: str) -> None:
    global _WORKER_MODEL, _WORKER_SNAPSHOT

    _WORKER_MODEL = load_model(Path(model_path))

    # IMPORTANT:
    # Each process owns an independent MuJoCo model.
    _enable_head_proxy(_WORKER_MODEL)
    _WORKER_SNAPSHOT = _snapshot_mutable_model_fields(_WORKER_MODEL)


def _worker_score_task(task: Tuple[Any, ...]) -> float:
    """
    Worker entry point for one candidate pose evaluation.

    Tasks are independent and therefore safe to distribute with pool.map().
    """
    (
        pose,
        scenario_id,
        severity,
        magnitude,
        timing_s,
        jitter_ms,
    ) = task

    model = _WORKER_MODEL
    _restore_model_snapshot(model, _WORKER_SNAPSHOT)

    result = run_protected_trial(
        model=model,
        scenario_id=scenario_id,
        pose=pose,
        severity=severity,
        magnitude=magnitude,
        timing_s=timing_s,
        timing_jitter_ms=jitter_ms,
    )

    return float(result.score)


# ---------------------------------------------------------------------------
# CMA-ES optimization
# ---------------------------------------------------------------------------

def _initial_pose(model: mujoco.MjModel) -> np.ndarray:
    indices = actuator_qpos_indices(model)
    return model.qpos0[indices].copy()


def _make_cma_bounds(
    model: mujoco.MjModel,
) -> Tuple[List[float], List[float]]:
    lo, hi = pose_limits(model)

    # Replace infinities with broad but finite search limits.
    lo = np.where(np.isfinite(lo), lo, -math.pi)
    hi = np.where(np.isfinite(hi), hi, math.pi)

    return lo.tolist(), hi.tolist()


def optimize_pose_for_scenario(
    model: mujoco.MjModel,
    scenario_id: int,
    severity: str,
    magnitude: float,
    pool: Optional[mp.pool.Pool],
    popsize: int,
    generations: int,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """
    Optimize one scenario/severity condition.

    CMA-ES itself runs in the parent process. Expensive MuJoCo candidate
    evaluations are distributed across CPU workers.
    """
    x0 = _initial_pose(model)
    lower, upper = _make_cma_bounds(model)

    options = {
        "popsize": popsize,
        "maxiter": generations,
        "seed": seed,
        "verb_disp": 0,
        "bounds": [lower, upper],
    }

    es = cma.CMAEvolutionStrategy(
        x0.tolist(),
        sigma,
        options,
    )

    while not es.stop():
        candidates = [
            np.asarray(x, dtype=float)
            for x in es.ask()
        ]

        tasks = [
            (
                clip_pose_to_limits(model, pose),
                scenario_id,
                severity,
                magnitude,
                LEAD_TIME_S,
                0.0,
            )
            for pose in candidates
        ]

        if pool is not None:
            scores = pool.map(_worker_score_task, tasks)
        else:
            scores = [
                run_protected_trial(
                    model,
                    scenario_id,
                    pose,
                    severity,
                    magnitude,
                    LEAD_TIME_S,
                    0.0,
                ).score
                for pose in candidates
            ]

        es.tell(
            [pose.tolist() for pose in candidates],
            scores,
        )

    best = np.asarray(es.result.xbest, dtype=float)
    return clip_pose_to_limits(model, best)


# ---------------------------------------------------------------------------
# Robustness / cross-evaluation
# ---------------------------------------------------------------------------

def evaluate_pose_robustness(
    model: mujoco.MjModel,
    pose: np.ndarray,
    scenario_id: int,
    severity: str,
    magnitude: float,
) -> Dict[str, Any]:
    """
    Evaluate timing and small disturbance uncertainty around the nominal pose.
    """
    results: List[TrialResult] = []

    for timing in COND_TIMINGS:
        for jitter in COND_JITTERS_MS:
            result = run_protected_trial(
                model=model,
                scenario_id=scenario_id,
                pose=pose,
                severity=severity,
                magnitude=magnitude,
                timing_s=timing,
                timing_jitter_ms=jitter,
            )
            results.append(result)

    valid = [r for r in results if r.transition_valid]

    return {
        "mean_score": float(np.mean([r.score for r in results])),
        "worst_score": float(np.max([r.score for r in results])),
        "mean_head_force": float(
            np.mean([r.peak_force_head for r in results])
        ),
        "mean_pelvis_force": float(
            np.mean([r.peak_force_pelvis for r in results])
        ),
        "mean_other_force": float(
            np.mean([r.peak_force_other for r in results])
        ),
        "transition_valid_fraction": float(
            len(valid) / max(1, len(results))
        ),
        "all_transition_valid": len(valid) == len(results),
        "n_trials": len(results),
        "trials": [asdict(r) for r in results],
    }


def cross_evaluate(
    model: mujoco.MjModel,
    poses: Sequence[np.ndarray],
    scenarios: Sequence[Dict[str, Any]],
    pool: Optional[mp.pool.Pool],
) -> np.ndarray:
    """
    Evaluate every pose against every fall condition.

    The resulting matrix has:
        rows    = poses
        columns = scenarios
    """
    tasks = []

    for pose in poses:
        for scenario in scenarios:
            tasks.append(
                (
                    pose,
                    scenario["scenario_id"],
                    scenario["severity"],
                    scenario["magnitude"],
                    LEAD_TIME_S,
                    0.0,
                )
            )

    if pool is not None:
        scores = pool.map(_worker_score_task, tasks)
    else:
        scores = [
            run_protected_trial(
                model=model,
                scenario_id=scenario["scenario_id"],
                pose=pose,
                severity=scenario["severity"],
                magnitude=scenario["magnitude"],
                timing_s=LEAD_TIME_S,
            ).score
            for pose, scenario in (
                (pose, scenario)
                for pose in poses
                for scenario in scenarios
            )
        ]

    return np.asarray(scores, dtype=float).reshape(
        len(poses),
        len(scenarios),
    )


# ---------------------------------------------------------------------------
# Pose merging / deterministic library construction
# ---------------------------------------------------------------------------

def pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def _degradation_ratio(
    before: np.ndarray,
    after: np.ndarray,
) -> float:
    base = np.maximum(np.abs(before), 1.0)
    return float(np.max(np.abs(after - before) / base))


def merge_pose_library(
    poses: List[np.ndarray],
    scenario_assignments: List[int],
    score_matrix: np.ndarray,
    target_min: int = MIN_LIBRARY_SIZE,
    target_max: int = MAX_LIBRARY_SIZE,
) -> Tuple[List[np.ndarray], List[int], np.ndarray]:
    """
    Greedily merge similar poses while limiting score degradation.

    We never merge beyond target_min. The resulting library is intended to be
    a compact validated rule-based lookup set, not a learned model.
    """
    poses = [p.copy() for p in poses]
    assignments = list(scenario_assignments)
    scores = score_matrix.copy()

    while len(poses) > target_min:
        best_pair = None
        best_dist = float("inf")

        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                dist = pose_distance(poses[i], poses[j])
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        merged_pose = clip_pose_to_limits(
            _WORKER_MODEL if _WORKER_MODEL is not None else model_for_merge,
            0.5 * (poses[i] + poses[j]),
        )

        # In a normal parent-process call, use the global model set by main.
        merged_scores = 0.5 * (scores[i] + scores[j])

        # Relative degradation against the better of the two original poses.
        reference = np.minimum(scores[i], scores[j])
        degradation = _degradation_ratio(reference, merged_scores)

        if degradation <= MERGE_TOLERANCE:
            poses[i] = merged_pose
            assignments[i] = assignments[i]

            poses.pop(j)
            assignments.pop(j)

            scores[i] = merged_scores
            scores = np.delete(scores, j, axis=0)
        else:
            # Mark this pair as unavailable by making its distance enormous.
            # We rebuild the search below using a finite candidate threshold.
            # If no pair passes, stop.
            candidates = []
            for a in range(len(poses)):
                for b in range(a + 1, len(poses)):
                    d = pose_distance(poses[a], poses[b])
                    candidates.append((d, a, b))

            merged = False
            for _, a, b in sorted(candidates):
                merged_pose = 0.5 * (poses[a] + poses[b])
                merged_scores = 0.5 * (scores[a] + scores[b])
                reference = np.minimum(scores[a], scores[b])
                degradation = _degradation_ratio(reference, merged_scores)

                if degradation <= MERGE_TOLERANCE:
                    poses[a] = merged_pose
                    poses.pop(b)
                    assignments.pop(b)
                    scores[a] = merged_scores
                    scores = np.delete(scores, b, axis=0)
                    merged = True
                    break

            if not merged:
                break

    # Keep no more than target_max. If more remain, retain the most useful
    # scenario-specialized poses by their worst-case score.
    if len(poses) > target_max:
        quality = np.max(scores, axis=1)
        keep = np.argsort(quality)[:target_max]
        keep = np.sort(keep)

        poses = [poses[i] for i in keep]
        assignments = [assignments[i] for i in keep]
        scores = scores[keep]

    return poses, assignments, scores


# This variable is assigned by main before merge_pose_library is called.
model_for_merge: Optional[mujoco.MjModel] = None


# ---------------------------------------------------------------------------
# Scenario preparation
# ---------------------------------------------------------------------------

def build_scenarios(
    model: mujoco.MjModel,
    scenario_ids: Sequence[int],
    severity_thresholds: Tuple[float, float],
) -> Tuple[List[Dict[str, Any]], List[int]]:
    scenarios: List[Dict[str, Any]] = []
    recalibration_required: List[int] = []

    for sid in scenario_ids:
        magnitude = fall_magnitude(sid)

        impact_t, velocity, fell = baseline_time_to_impact(
            model,
            sid,
            magnitude,
        )

        if not fell or impact_t is None:
            # The scenario needs calibration/review rather than pretending a
            # protective pose is valid for a condition that did not fall.
            recalibration_required.append(sid)
            severity = "unknown"
        else:
            severity = severity_from_velocity(
                velocity,
                severity_thresholds,
            )

        scenarios.append(
            {
                "scenario_id": sid,
                "name": _scenario_name(sid),
                "magnitude": magnitude,
                "impact_time_s": impact_t,
                "impact_velocity_mps": velocity,
                "severity": severity,
                "fell": bool(fell),
            }
        )

    return scenarios, recalibration_required


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _pose_to_list(pose: np.ndarray) -> List[float]:
    return [float(x) for x in np.asarray(pose)]


def build_pose_library_json(
    model: mujoco.MjModel,
    poses: Sequence[np.ndarray],
    assignments: Sequence[int],
    scenarios: Sequence[Dict[str, Any]],
    robustness: Sequence[Dict[str, Any]],
    cross_scores: np.ndarray,
    recalibration_required: Sequence[int],
    severity_thresholds: Tuple[float, float],
) -> Dict[str, Any]:
    scenario_by_id = {
        int(s["scenario_id"]): s
        for s in scenarios
    }

    library = []

    for idx, (pose, sid, robust) in enumerate(
        zip(poses, assignments, robustness)
    ):
        s = scenario_by_id.get(int(sid), {})

        library.append(
            {
                "pose_id": idx,
                "source_scenario_ids": [int(sid)],
                "direction": infer_direction_from_scenario(int(sid)),
                "cause": infer_cause_from_scenario(int(sid)),
                "severity": s.get("severity", "unknown"),
                "target_joint_angles_rad": _pose_to_list(pose),
                "robustness": robust,
                "validated_transition_budget_s": POSE_TRANSITION_BUDGET_S,
            }
        )

    return {
        "phase": 3,
        "purpose": "Offline protective-pose design and validation",
        "runtime_policy": (
            "Deterministic direction + cause + severity/velocity lookup; "
            "CMA-ES is offline only."
        ),
        "n_poses": len(library),
        "lead_time_s_used_for_design": LEAD_TIME_S,
        "pose_transition_budget_s": POSE_TRANSITION_BUDGET_S,
        "transition_tolerance_rad": TRANSITION_TOL_RAD,
        "severity_thresholds_mps": {
            "low_to_medium": severity_thresholds[0],
            "medium_to_high": severity_thresholds[1],
        },
        "impact_proxies": {
            "pelvis": "pelvis ground contact",
            "head": (
                "pendulum bob / upper-body head proxy; "
                "must remain collision-enabled in XML"
            ),
            "hands": "not modeled",
            "other": "non-foot, non-pelvis, non-head ground contacts",
        },
        "recalibration_required_scenarios": [
            int(x) for x in recalibration_required
        ],
        "scenario_conditions": scenarios,
        "poses": library,
        "cross_evaluation": {
            "shape": list(cross_scores.shape),
            "pose_by_scenario_score": cross_scores.tolist(),
        },
    }


def infer_direction_from_scenario(scenario_id: int) -> str:
    mapping = {
        1: "forward",
        2: "backward",
        3: "left",
        4: "right",
        5: "forward_left",
        6: "forward_right",
        7: "surface",
        8: "surface",
        9: "surface",
        10: "surface",
        11: "actuator",
        12: "actuator",
        13: "actuator",
        14: "trip",
        15: "trip",
    }
    return mapping.get(scenario_id, "unknown")


def infer_cause_from_scenario(scenario_id: int) -> str:
    if scenario_id in {1, 2, 3, 4, 5, 6}:
        return "push"
    if scenario_id in {7, 8, 9, 10}:
        return "surface"
    if scenario_id in {11, 12, 13}:
        return "actuator"
    if scenario_id in {14, 15}:
        return "trip_or_load"
    return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global model_for_merge

    ap = argparse.ArgumentParser(
        description="Phase 3 protective pose design for Unitree G1."
    )
    ap.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to g1_pendulum.xml",
    )
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output pose library JSON",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of CPU MuJoCo worker processes.",
    )
    ap.add_argument(
        "--popsize",
        type=int,
        default=DEFAULT_POPSIZE,
        help="CMA-ES population size.",
    )
    ap.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
        help="CMA-ES generations per scenario.",
    )
    ap.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help="Initial CMA-ES search sigma in radians.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    ap.add_argument(
        "--scenarios",
        nargs="*",
        type=int,
        default=list(range(1, 16)),
        help="Scenario IDs to optimize. Default: 1..15.",
    )

    args = ap.parse_args()

    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    np.random.seed(args.seed)

    print("=" * 72)
    print("PHASE 3 — PROTECTIVE POSE DESIGN")
    print("=" * 72)
    print(f"Model   : {model_path}")
    print(f"Output  : {output_path}")
    print(f"Workers : {args.workers}")
    print(f"CPU     : {os.cpu_count()}")
    print(f"Lead    : {LEAD_TIME_S:.3f} s")
    print(f"Budget  : {POSE_TRANSITION_BUDGET_S:.3f} s")
    print()

    model = load_model(model_path)

    # The XML must have the bob compiled as a collidable geom.
    _enable_head_proxy(model)
    model_for_merge = model

    # ------------------------------------------------------------------
    # 1. Baseline severity measurement
    # ------------------------------------------------------------------
    print("[1/6] Establishing unprotected baselines...")

    baseline_rows = []
    velocities = []

    for sid in args.scenarios:
        magnitude = fall_magnitude(sid)
        impact_t, velocity, fell = baseline_time_to_impact(
            model,
            sid,
            magnitude,
        )

        if fell and velocity > 0:
            velocities.append(velocity)

        baseline_rows.append(
            {
                "scenario_id": sid,
                "impact_time_s": impact_t,
                "impact_velocity_mps": velocity,
                "fell": fell,
            }
        )

        print(
            f"  Scenario {sid:2d}: "
            f"fell={fell!s:<5} "
            f"impact={impact_t if impact_t is not None else 'N/A'} "
            f"velocity={velocity:.3f} m/s"
        )

    severity_thresholds = derive_severity_thresholds(velocities)

    print(
        f"  Severity thresholds: "
        f"{severity_thresholds[0]:.3f} / "
        f"{severity_thresholds[1]:.3f} m/s"
    )

    scenarios, recalibration_required = build_scenarios(
        model,
        args.scenarios,
        severity_thresholds,
    )

    if recalibration_required:
        print(
            "  Recalibration/review required for scenarios:",
            recalibration_required,
        )

    # ------------------------------------------------------------------
    # 2. CPU worker pool
    # ------------------------------------------------------------------
    pool: Optional[mp.pool.Pool] = None

    if args.workers > 1:
        print("[2/6] Starting CPU-parallel MuJoCo workers...")
        pool = mp.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(str(model_path),),
        )
    else:
        print("[2/6] Running in single-process mode.")

    try:
        # --------------------------------------------------------------
        # 3. Optimize scenario-specific poses
        # --------------------------------------------------------------
        print("[3/6] Optimizing protective poses with CMA-ES...")

        optimized_poses: List[np.ndarray] = []
        assignments: List[int] = []

        for idx, scenario in enumerate(scenarios):
            sid = int(scenario["scenario_id"])

            # Do not optimize a fake protective pose for a condition that
            # did not produce the expected fall.
            if not scenario["fell"]:
                print(
                    f"  Scenario {sid:2d}: skipped "
                    "(baseline did not produce a fall)"
                )
                continue

            print(
                f"  Scenario {sid:2d} "
                f"({scenario['name']}, {scenario['severity']})..."
            )

            pose = optimize_pose_for_scenario(
                model=model,
                scenario_id=sid,
                severity=scenario["severity"],
                magnitude=float(scenario["magnitude"]),
                pool=pool,
                popsize=args.popsize,
                generations=args.generations,
                sigma=args.sigma,
                seed=args.seed + idx,
            )

            optimized_poses.append(pose)
            assignments.append(sid)

        if not optimized_poses:
            raise RuntimeError(
                "No poses were optimized. Check Phase-1 integration and "
                "scenario fall generation."
            )

        # --------------------------------------------------------------
        # 4. Robustness evaluation
        # --------------------------------------------------------------
        print("[4/6] Stress-testing optimized poses...")

        robustness_rows = []

        scenario_by_id = {
            int(s["scenario_id"]): s
            for s in scenarios
        }

        for pose, sid in zip(optimized_poses, assignments):
            scenario = scenario_by_id[sid]

            robust = evaluate_pose_robustness(
                model=model,
                pose=pose,
                scenario_id=sid,
                severity=scenario["severity"],
                magnitude=float(scenario["magnitude"]),
            )
            robustness_rows.append(robust)

            print(
                f"  Pose for scenario {sid:2d}: "
                f"mean_score={robust['mean_score']:.2f}, "
                f"worst_score={robust['worst_score']:.2f}, "
                f"transition_valid="
                f"{robust['transition_valid_fraction']:.0%}"
            )

        # --------------------------------------------------------------
        # 5. Cross-evaluate every pose against every scenario
        # --------------------------------------------------------------
        print("[5/6] Cross-evaluating pose library...")

        cross_scores = cross_evaluate(
            model=model,
            poses=optimized_poses,
            scenarios=scenarios,
            pool=pool,
        )

        # Merge only after cross-evaluation.
        merged_poses, merged_assignments, merged_scores = (
            merge_pose_library(
                optimized_poses,
                assignments,
                cross_scores,
                target_min=MIN_LIBRARY_SIZE,
                target_max=MAX_LIBRARY_SIZE,
            )
        )

        # If merging changes the set, recompute robustness for the final poses.
        final_robustness = []
        for pose, sid in zip(merged_poses, merged_assignments):
            scenario = scenario_by_id[sid]
            final_robustness.append(
                evaluate_pose_robustness(
                    model=model,
                    pose=pose,
                    scenario_id=sid,
                    severity=scenario["severity"],
                    magnitude=float(scenario["magnitude"]),
                )
            )

        # --------------------------------------------------------------
        # 6. Write deterministic rule-based library
        # --------------------------------------------------------------
        print("[6/6] Writing pose library...")

        library = build_pose_library_json(
            model=model,
            poses=merged_poses,
            assignments=merged_assignments,
            scenarios=scenarios,
            robustness=final_robustness,
            cross_scores=merged_scores,
            recalibration_required=recalibration_required,
            severity_thresholds=severity_thresholds,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(library, f, indent=2)

        print()
        print("=" * 72)
        print("PHASE 3 COMPLETE")
        print("=" * 72)
        print(f"Validated candidate poses : {len(merged_poses)}")
        print(f"Pose transition budget    : {POSE_TRANSITION_BUDGET_S:.3f} s")
        print(f"Output                    : {output_path}")
        print()
        print(
            "NOTE: This library is only final after all required "
            "biomechanical checks pass."
        )

    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    mp.freeze_support()
    main()
