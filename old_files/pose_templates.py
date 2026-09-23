#!/usr/bin/env python3
"""
pose_templates.py -- 3 hand-structured, knee-dominant pose templates for the
AXON-R protective pose library.

WHY TEMPLATES INSTEAD OF FREE 12-DIM CMA-ES
    Free search over all 12 leg actuators can "discover" biomechanically
    invalid poses (leg-crossing, knee-slam) because nothing in a pure
    peak-force objective forbids them.  Each template below exposes only
    2-4 scalar parameters (e.g. "how deep", "how wide"); every parameter is
    hard-clamped, via validity_spec's own envelope, before it ever becomes a
    pose_ctrl vector.  CMA-ES tunes the FEW parameters; it can no longer
    reach an invalid pose because the invalid region isn't in the search
    space at all.

    Design is knee-dominant because this rig's hip_pitch flexion is
    hardware-limited to ~30 deg (confirmed against g1_pendulum.xml on
    2026-09) -- squatting/lowering must come mostly from the knees + ankle
    dorsiflexion, with hip flexion contributing only a small amount.

TEMPLATES
    lower_squat    -- symmetric CoM lowering. Forward/backward falls, or as
                      the generic fallback when direction is uncertain.
    widen_crouch   -- lower_squat + hip abduction for a wider base. Sudden
                      load / floor-tilt causes where stability matters more
                      than CoM height alone.
    step_recover   -- asymmetric: one leg steps toward the fall direction
                      (hip yaw + extra knee bend), the other mirrors
                      lower_squat. Lateral (left/right) falls and pushes --
                      the only family that can plausibly prevent, not just
                      soften, some falls.

Mirroring: for a "left" direction pose, build with side="left" (that leg
steps); the model is assumed left/right symmetric in the hip layout probed
by validity_spec, so side="right" mirrors automatically via the resolved
per-side actuator indices -- no separate geometry work needed.

USAGE
    from validity_spec import build_spec
    from pose_templates import TEMPLATES, params_to_ctrl, cma_bounds, PLAN

    spec = build_spec(model)
    tmpl = TEMPLATES["lower_squat"]
    x0, lo, hi = cma_bounds(spec, tmpl)          # feed straight to cma.CMAEvolutionStrategy
    ctrl = params_to_ctrl(spec, tmpl, x)          # x -> full nu-length pose_ctrl
    violations = check_pose_ctrl(spec, ctrl)      # still worth a final check (see note below)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from validity_spec import JointSpec, ValiditySpec


# =============================================================================
# Template definition
# =============================================================================

@dataclass
class ParamSpec:
    name: str
    lo: float   # 0..1 normalized
    hi: float
    default: float


@dataclass
class Template:
    name: str
    params: List[ParamSpec]
    build: Callable[[ValiditySpec, Dict[str, float], str], np.ndarray]
    directions: Tuple[str, ...]   # which fall directions this template is meant for
    description: str


def _joint(spec: ValiditySpec, side: str, jtype: str) -> JointSpec:
    for j in spec.leg_joints():
        if j.side == side and j.jtype == jtype:
            return j
    raise KeyError(f"no {jtype} joint on {side} side")


def _clamp(j: JointSpec, val: float) -> float:
    return float(np.clip(val, j.lo, j.hi))


def _set(ctrl: np.ndarray, j: JointSpec, val: float) -> None:
    ctrl[j.act_idx] = _clamp(j, val)


def _lerp(j: JointSpec, sign: int, frac: float, extreme_frac: float = 1.0) -> float:
    """stand -> stand + sign*frac*extreme_frac*(available range on that side)."""
    frac = float(np.clip(frac, 0.0, 1.0)) * extreme_frac
    target = j.hi if sign > 0 else j.lo
    return j.stand + frac * (target - j.stand)


# --- lower_squat -------------------------------------------------------------

def _build_lower_squat(spec: ValiditySpec, p: Dict[str, float], side: str = "both") -> np.ndarray:
    """p: depth (0=stand, 1=deepest safe knee+ankle bend, small hip contribution)."""
    ctrl = spec.stand_ctrl.copy()
    depth = float(np.clip(p["depth"], 0.0, 1.0))
    for s in ("left", "right"):
        knee, hip, ankle = _joint(spec, s, "knee"), _joint(spec, s, "hip_pitch"), _joint(spec, s, "ankle_pitch")
        _set(ctrl, knee, _lerp(knee, +1, depth))                 # main lowering
        _set(ctrl, hip, _lerp(hip, -hip.sign, depth, 0.5))       # small flexion contribution (hip-limited)
        _set(ctrl, ankle, _lerp(ankle, +1, depth, 0.6))          # dorsiflex to keep foot flat-ish
    return ctrl


# --- widen_crouch --------------------------------------------------------------

def _build_widen_crouch(spec: ValiditySpec, p: Dict[str, float], side: str = "both") -> np.ndarray:
    """p: depth (as above), width (0=stand stance, 1=max safe abduction each side)."""
    ctrl = _build_lower_squat(spec, {"depth": p["depth"]})
    width = float(np.clip(p.get("width", 0.0), 0.0, 1.0))
    for s in ("left", "right"):
        roll = _joint(spec, s, "hip_roll")
        # abduction direction is always "away from midline"; ENVELOPE_DEG already
        # resolves which raw sign that is per-side via build_spec's probe.
        away = +1 if roll.hi >= roll.stand and abs(roll.hi - roll.stand) >= abs(roll.lo - roll.stand) else -1
        _set(ctrl, roll, _lerp(roll, away, width))
    return ctrl


# --- step_recover --------------------------------------------------------------

def _build_step_recover(spec: ValiditySpec, p: Dict[str, float], side: str) -> np.ndarray:
    """p: depth (mirror leg lowering), step (0=no step, 1=max safe yaw+extra knee
    bend on the stepping leg). `side` picks which leg steps -- pass the fall
    direction's near-side leg ("left" for a leftward fall, etc.)."""
    if side not in ("left", "right"):
        raise ValueError("step_recover requires side='left' or 'right'")
    mirror = "right" if side == "left" else "left"
    ctrl = spec.stand_ctrl.copy()
    depth = float(np.clip(p["depth"], 0.0, 1.0))
    step = float(np.clip(p.get("step", 0.0), 0.0, 1.0))

    # mirror leg: plain lower_squat behavior, takes most of the standing load
    for jtype, sign, extreme in (("knee", +1, 1.0), ("hip_pitch", None, 0.5), ("ankle_pitch", +1, 0.6)):
        j = _joint(spec, mirror, jtype)
        s = sign if sign is not None else -j.sign
        _set(ctrl, j, _lerp(j, s, depth, extreme))

    # stepping leg: yaw toward the fall direction + extra knee bend to plant wider/lower
    yaw = _joint(spec, side, "hip_yaw")
    away = +1 if side == "left" else -1  # left leg stepping left = yaw away from midline; mirrors for right
    _set(ctrl, yaw, _lerp(yaw, away, step))
    knee = _joint(spec, side, "knee")
    _set(ctrl, knee, _lerp(knee, +1, min(1.0, depth * 0.6 + step * 0.4)))
    ankle = _joint(spec, side, "ankle_pitch")
    _set(ctrl, ankle, _lerp(ankle, +1, depth, 0.6))
    return ctrl


TEMPLATES: Dict[str, Template] = {
    "lower_squat": Template(
        "lower_squat",
        [ParamSpec("depth", 0.0, 1.0, 0.5)],
        _build_lower_squat,
        directions=("forward", "backward", "unknown"),
        description="Symmetric CoM lowering via knee flexion + ankle dorsiflexion, small hip contribution.",
    ),
    "widen_crouch": Template(
        "widen_crouch",
        [ParamSpec("depth", 0.0, 1.0, 0.5), ParamSpec("width", 0.0, 1.0, 0.4)],
        _build_widen_crouch,
        directions=("sudden_load", "floor_tilt", "unknown"),
        description="lower_squat plus hip abduction for a wider, more stable base.",
    ),
    "step_recover": Template(
        "step_recover",
        [ParamSpec("depth", 0.0, 1.0, 0.5), ParamSpec("step", 0.0, 1.0, 0.4)],
        _build_step_recover,
        directions=("left", "right"),
        description="One leg steps toward the fall direction (hip yaw + knee), the other lowers like lower_squat.",
    ),
}

# Suggested direction/cause -> template mapping to reach ~12-15 poses total
# (5 direction/cause groups x ~2-3 velocity-bin variants each, tuned separately
# per bin by CMA-ES with the SAME template/side, different params).
PLAN: List[Tuple[str, str, str]] = [
    # (scenario/direction label, template name, side arg for the build fn)
    ("forward", "lower_squat", "both"),
    ("backward", "lower_squat", "both"),
    ("sudden_load", "widen_crouch", "both"),
    ("floor_tilt", "widen_crouch", "both"),
    ("left", "step_recover", "left"),
    ("right", "step_recover", "right"),
]


# =============================================================================
# CMA-ES glue: normalized [0,1]^k params -> full pose_ctrl, with bounds
# =============================================================================

def params_to_ctrl(spec: ValiditySpec, tmpl: Template, x: np.ndarray, side: str = "both") -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    p = {ps.name: float(x[i]) for i, ps in enumerate(tmpl.params)}
    return tmpl.build(spec, p, side)


def cma_bounds(tmpl: Template) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x0, lower, upper for cma.CMAEvolutionStrategy -- always [0,1]^k since
    every template parameter is itself normalized and clamped inside build()."""
    x0 = np.array([ps.default for ps in tmpl.params])
    lo = np.array([ps.lo for ps in tmpl.params])
    hi = np.array([ps.hi for ps in tmpl.params])
    return x0, lo, hi


if __name__ == "__main__":
    import argparse
    from validity_spec import build_spec, check_pose_ctrl
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    spec = build_spec(model)

    print(f"{'template':14s}{'side':7s}{'params':30s}{'violations'}")
    for label, name, side in PLAN:
        tmpl = TEMPLATES[name]
        x0, lo, hi = cma_bounds(tmpl)
        for depth in (0.0, 0.5, 1.0):
            x = x0.copy()
            x[0] = depth
            ctrl = params_to_ctrl(spec, tmpl, x, side)
            v = check_pose_ctrl(spec, ctrl)
            pstr = ", ".join(f"{ps.name}={x[i]:.2f}" for i, ps in enumerate(tmpl.params))
            print(f"{name:14s}{side:7s}{pstr:30s}{'OK' if not v else '; '.join(v[:2])}   [{label}]")
