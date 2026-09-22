#!/usr/bin/env python3
"""
validity_spec.py -- biomechanical validity spec + acceptance gate for the AXON-R
protective pose library (lower-limb exoskeleton worn by paralyzed users).

WHY THIS FILE EXISTS
    "Lowest peak force" is not the same as "a pose a patient could safely
    undergo".  This module defines validity independently of the optimizer, so
    CMA-ES can never win by collapsing the legs, crossing them, or slamming
    fragile parts into the floor.  It provides:

      1. Exo-safe joint envelope   (human ROM norms, per joint, sign-aware)
      2. Static pose checks        (envelope, leg crossing/self-clearance,
                                    transition speed)
      3. Contact classes + weights (head / pelvis / thigh-hip / knee-shank / foot)
      4. ContactTracker            (drop-in per-step tracker, records first
                                    non-foot contact class)
      5. Acceptance gate           (do-no-harm + held-out effectiveness; a pose
                                    that fails becomes "stand" = no pose)
      6. Audit CLI                 (run this FIRST, before touching CMA-ES)

    Every number marked  [CONFIRM]  is a conservative STARTING VALUE from
    general biomechanics knowledge, not a clinical or hardware-verified spec.
    Have your mentor / the exoskeleton datasheet confirm or replace them.

USAGE
    python validity_spec.py --model g1_pendulum.xml
    python validity_spec.py --model g1_pendulum.xml --pose-json pose_library.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import mujoco
import numpy as np

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

LEG_JOINT_TYPES = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")

# Exo-safe envelope in ANATOMICAL degrees, measured from model-zero (straight leg).
#   pos_name : which anatomical motion the "pos_max" side means (None = symmetric)
#   pos_max  : max deg in that motion          neg_max : max deg in the opposite one
# Roughly 70-80% of AAOS-style normative ROM, tighter where crossing/rotation is
# a risk.  Paralyzed users often have reduced bone density below the injury, so
# the envelope is deliberately conservative.                       [CONFIRM]
ENVELOPE_DEG: Dict[str, Dict[str, object]] = {
    "hip_pitch":   dict(pos_name="flexion",   pos_max=90.0,  neg_max=15.0),   # ext = 15
    "hip_roll":    dict(pos_name="abduction", pos_max=30.0,  neg_max=15.0),   # adduction = 15 (no crossing)
    "hip_yaw":     dict(pos_name=None,        pos_max=20.0,  neg_max=20.0),
    "knee":        dict(pos_name="flexion",   pos_max=100.0, neg_max=0.0),    # no hyperextension
    "ankle_pitch": dict(pos_name=None,        pos_max=25.0,  neg_max=25.0),
    "ankle_roll":  dict(pos_name=None,        pos_max=15.0,  neg_max=15.0),
}
# The neutral stance must never be declared invalid: widen each envelope so it
# always contains  stand +/- this margin.
STAND_MARGIN_DEG = 5.0

# Transition (minimum-jerk, Flash & Hogan): peak speed = 1.875 * delta / T.
T_TRANSITION_S = 0.30
# Confirmed max SAFE COMMANDED velocity while bearing a patient's leg (2026-09).
# NOTE: a much larger figure (~100-300 rad/s) was separately proposed; do not
# substitute it here unless confirmed to be this same safe-commanded-velocity
# quantity, not a bare motor no-load speed rating -- the two differ by 10-30x
# and using the wrong one would silently defeat the min-jerk transition safety
# check (reference doc section 8.2).                                [CONFIRM]
MAX_JOINT_SPEED_RAD_S = {t: 10.5 for t in LEG_JOINT_TYPES}

# Leg-crossing / self-collision rules (metres, measured at the pose's static
# kinematics with the base held at the standing keyframe).         [CONFIRM]
MIN_LEG_CLEARANCE_M = 0.01     # min geom-to-geom distance, left leg vs right leg
MIN_KNEE_SEPARATION_M = 0.08   # left knee must stay this far left of right knee
MIN_ANKLE_SEPARATION_M = 0.08  # same for ankles

# Contact classes and injury-risk weights.  Fragile bone (knee/shank/femur) is
# weighted above pelvis (gluteal soft tissue), so dumping load onto the knees
# is never "cheap".  Foot contact is excluded (feet are meant to touch).
BODY_CLASSES = ("head", "pelvis", "thigh_hip", "knee_shank", "foot", "other")
FORCE_WEIGHTS: Dict[str, float] = {                                  # [CONFIRM]
    "head": 2.0,
    "knee_shank": 1.5,
    "pelvis": 1.0,
    "thigh_hip": 1.0,
    "other": 0.5,
    "foot": 0.0,
}
# Substring rules, checked in order, on the (lower-cased) BODY name.
BODY_CLASS_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("head",       ("pendulum", "head", "torso", "bob")),
    ("pelvis",     ("pelvis",)),
    ("foot",       ("ankle", "foot", "toe")),
    ("knee_shank", ("knee", "shank", "shin")),
    ("thigh_hip",  ("hip", "thigh", "femur")),
]
CONTACT_EPS_N = 1.0  # a class "touches" the floor once its summed force exceeds this


# =============================================================================
# 2. SPEC BUILDING (reads YOUR model, so limits are never hard-coded to a guess)
# =============================================================================

def _joint_type(name: str) -> Optional[str]:
    n = name.lower()
    for t in LEG_JOINT_TYPES:
        if t in n:
            return t
    return None


def _side(name: str) -> Optional[str]:
    n = name.lower()
    if "left" in n:
        return "left"
    if "right" in n:
        return "right"
    return None


def _reset(model: mujoco.MjModel, data: mujoco.MjData, keyframe: int) -> None:
    if model.nkey > keyframe:
        mujoco.mj_resetDataKeyframe(model, data, keyframe)
    else:
        mujoco.mj_resetData(model, data)


def _root_body(model: mujoco.MjModel) -> int:
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_bodyid[j])
    return 1


def _body_name(model: mujoco.MjModel, b: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""


def classify_body(name: str) -> str:
    n = name.lower()
    for cls, keys in BODY_CLASS_RULES:
        if any(k in n for k in keys):
            return cls
    return "other"


@dataclass
class JointSpec:
    act_idx: int
    act_name: str
    joint_id: int
    joint_name: str
    jtype: Optional[str]
    side: Optional[str]
    is_leg: bool
    sign: int          # +1: positive model angle = anatomical "pos_name" motion
    model_lo: float    # rad, from the XML (or -inf)
    model_hi: float
    lo: float          # rad, final exo-safe envelope
    hi: float
    stand: float       # rad, keyframe target


@dataclass
class ValiditySpec:
    model: mujoco.MjModel
    keyframe: int
    joints: List[JointSpec]
    stand_ctrl: np.ndarray
    warnings: List[str] = field(default_factory=list)

    @property
    def nu(self) -> int:
        return self.model.nu

    def leg_joints(self) -> List[JointSpec]:
        return [j for j in self.joints if j.is_leg]


def _probe_signs(model: mujoco.MjModel, keyframe: int, warnings: List[str], eps: float = 0.2) -> Dict[int, int]:
    """Kinematic probe: which model-angle direction is flexion / abduction?
    Uses the pelvis frame (x = forward, y = left), so it works for any XML sign convention."""
    d0 = mujoco.MjData(model)
    _reset(model, d0, keyframe)
    mujoco.mj_forward(model, d0)
    root = _root_body(model)
    R = d0.xmat[root].reshape(3, 3)
    fwd, left = R[:, 0].copy(), R[:, 1].copy()
    base = d0.xpos.copy()

    by_key: Dict[Tuple[Optional[str], Optional[str]], int] = {}
    for j in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        by_key[(_side(nm), _joint_type(nm))] = j

    def moved(jid: int, body: int) -> np.ndarray:
        d = mujoco.MjData(model)
        _reset(model, d, keyframe)
        d.qpos[model.jnt_qposadr[jid]] += eps
        mujoco.mj_forward(model, d)
        return d.xpos[body] - base[body]

    signs: Dict[int, int] = {}
    for (side, jt), jid in by_key.items():
        if jt not in ("hip_pitch", "knee", "hip_roll") or side is None:
            continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jt in ("hip_pitch", "hip_roll"):
            probe_j = by_key.get((side, "knee"))
        else:
            probe_j = by_key.get((side, "ankle_pitch"))
        if probe_j is None:
            warnings.append(f"sign probe: no distal joint for {nm}; assuming +1")
            signs[jid] = 1
            continue
        delta = moved(jid, int(model.jnt_bodyid[probe_j]))
        if jt == "hip_pitch":
            v, sgn = float(delta @ fwd), 1          # flexion swings the knee forward
        elif jt == "knee":
            v, sgn = float(delta @ fwd), -1         # flexion swings the foot backward
        else:
            v, sgn = float(delta @ left) * (1 if side == "left" else -1), 1  # abduction = away from midline
        if abs(v) < 1e-4:
            warnings.append(f"sign probe inconclusive for {nm}; assuming +1")
            signs[jid] = 1
        else:
            signs[jid] = 1 if v * sgn > 0 else -1
    return signs


def build_spec(model: mujoco.MjModel, keyframe: int = 0) -> ValiditySpec:
    warnings: List[str] = []
    signs = _probe_signs(model, keyframe, warnings)

    if model.nkey > keyframe and model.key_ctrl.size:
        stand_ctrl = np.array(model.key_ctrl[keyframe], dtype=float)
    else:
        stand_ctrl = np.zeros(model.nu)
        warnings.append("no keyframe ctrl found; using zeros as 'stand' -- audit stand values!")

    joints: List[JointSpec] = []
    for a in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act{a}"
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            joints.append(JointSpec(a, aname, -1, "", None, None, False, 1, -np.inf, np.inf, -np.inf, np.inf, stand_ctrl[a]))
            continue
        jid = int(model.actuator_trnid[a, 0])
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        jt, side = _joint_type(jname), _side(jname)
        is_leg = jt is not None and "pendulum" not in jname.lower()
        m_lo, m_hi = (model.jnt_range[jid] if model.jnt_limited[jid] else (-np.inf, np.inf))
        sign = signs.get(jid, 1)
        lo, hi = m_lo, m_hi
        if is_leg:
            e = ENVELOPE_DEG[jt]
            pos, neg = float(e["pos_max"]), float(e["neg_max"])
            if e["pos_name"] is None or sign == 1:
                lo_d, hi_d = -neg, pos
            else:
                lo_d, hi_d = -pos, neg
            m = np.radians(STAND_MARGIN_DEG)
            lo_r = min(np.radians(lo_d), stand_ctrl[a] - m)
            hi_r = max(np.radians(hi_d), stand_ctrl[a] + m)
            lo, hi = max(lo_r, m_lo), min(hi_r, m_hi)
        joints.append(JointSpec(a, aname, jid, jname, jt, side, is_leg, sign,
                                float(m_lo), float(m_hi), float(lo), float(hi), float(stand_ctrl[a])))
    return ValiditySpec(model, keyframe, joints, stand_ctrl, warnings)


# =============================================================================
# 3. STATIC POSE VALIDITY  (call this on every candidate BEFORE simulating it)
# =============================================================================

def _leg_geoms(model: mujoco.MjModel) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {"left": [], "right": []}
    allg: Dict[str, List[int]] = {"left": [], "right": []}
    for g in range(model.ngeom):
        side = _side(_body_name(model, int(model.geom_bodyid[g])))
        if side is None:
            continue
        allg[side].append(g)
        if model.geom_contype[g] or model.geom_conaffinity[g]:
            out[side].append(g)
    for s in out:
        if not out[s]:
            out[s] = allg[s]  # collision geoms disabled in the XML -> fall back to all geoms
    return out


def check_pose_ctrl(spec: ValiditySpec, pose_ctrl: np.ndarray, T: float = T_TRANSITION_S) -> List[str]:
    """Return a list of violation strings; empty list == statically valid."""
    model = spec.model
    pose = np.asarray(pose_ctrl, dtype=float).ravel()
    if pose.size != model.nu:
        return [f"pose length {pose.size} != nu {model.nu}"]
    why: List[str] = []
    tol = 1e-3

    # (a) envelope + (b) transition speed
    for j in spec.leg_joints():
        v = pose[j.act_idx]
        if v < j.lo - tol or v > j.hi + tol:
            why.append(f"{j.joint_name}: {np.degrees(v):.1f} deg outside envelope "
                       f"[{np.degrees(j.lo):.1f}, {np.degrees(j.hi):.1f}]")
        peak = 1.875 * abs(v - j.stand) / T
        vmax = MAX_JOINT_SPEED_RAD_S[j.jtype]
        if peak > vmax:
            why.append(f"{j.joint_name}: peak transition speed {peak:.1f} rad/s > {vmax:.1f}")

    # (c) static kinematics: leg crossing / self-clearance
    d = mujoco.MjData(model)
    _reset(model, d, spec.keyframe)
    for j in spec.leg_joints():
        d.qpos[model.jnt_qposadr[j.joint_id]] = pose[j.act_idx]
    mujoco.mj_forward(model, d)

    root = _root_body(model)
    left_axis = d.xmat[root].reshape(3, 3)[:, 1]

    def body_of(side: str, jt: str) -> Optional[int]:
        for j in spec.leg_joints():
            if j.side == side and j.jtype == jt:
                return int(model.jnt_bodyid[j.joint_id])
        return None

    for jt, lim, label in (("knee", MIN_KNEE_SEPARATION_M, "knees"),
                           ("ankle_pitch", MIN_ANKLE_SEPARATION_M, "ankles")):
        bl, br = body_of("left", jt), body_of("right", jt)
        if bl is not None and br is not None:
            sep = float((d.xpos[bl] - d.xpos[br]) @ left_axis)
            if sep < lim:
                why.append(f"{label} too close / crossed: separation {sep:.3f} m < {lim:.3f} m")

    if hasattr(mujoco, "mj_geomDistance"):
        lg = _leg_geoms(model)
        best, ft = np.inf, np.zeros(6)
        for a in lg["left"]:
            for b in lg["right"]:
                best = min(best, mujoco.mj_geomDistance(model, d, a, b, 0.2, ft))
        if best < MIN_LEG_CLEARANCE_M:
            why.append(f"left/right legs collide or nearly touch: clearance {best:.3f} m < {MIN_LEG_CLEARANCE_M:.3f} m")
    return why


# =============================================================================
# 4. CONTACT TRACKING (drop-in, per step)
# =============================================================================

class ContactTracker:
    """Call reset() at trial start, update(data) after every mj_step.
    Sums floor-contact force per body class each step, keeps the per-class peak,
    and records which non-foot class touched the floor first."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.floor_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0}
        self.geom_cls = [classify_body(_body_name(model, int(model.geom_bodyid[g])))
                         for g in range(model.ngeom)]
        self._f6 = np.zeros(6)
        self.reset()

    def reset(self) -> None:
        self.peak = {c: 0.0 for c in BODY_CLASSES}
        self.first_time: Dict[str, Optional[float]] = {c: None for c in BODY_CLASSES}

    def update(self, data: mujoco.MjData) -> None:
        step = {c: 0.0 for c in BODY_CLASSES}
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.floor_geoms:
                other = g2
            elif g2 in self.floor_geoms:
                other = g1
            else:
                continue
            mujoco.mj_contactForce(self.model, data, i, self._f6)
            step[self.geom_cls[other]] += float(np.linalg.norm(self._f6[:3]))
        for cls, f in step.items():
            self.peak[cls] = max(self.peak[cls], f)
            if f > CONTACT_EPS_N and self.first_time[cls] is None:
                self.first_time[cls] = float(data.time)

    @property
    def first_nonfoot_class(self) -> Optional[str]:
        cand = [(t, c) for c, t in self.first_time.items() if t is not None and c != "foot"]
        return min(cand)[1] if cand else None


def weighted_force(peaks: Dict[str, float], weights: Dict[str, float] = FORCE_WEIGHTS) -> float:
    return float(sum(weights.get(c, 0.0) * f for c, f in peaks.items()))


# =============================================================================
# 5. ACCEPTANCE GATE  (a pose only enters the library if it passes)
# =============================================================================

@dataclass
class TrialResult:
    fell: bool
    peaks: Dict[str, float]                     # ContactTracker.peak
    first_contact_class: Optional[str] = None   # ContactTracker.first_nonfoot_class
    tracking_err_rad: float = 0.0               # peak tracking error (protected trials)

    @property
    def score(self) -> float:
        return weighted_force(self.peaks)


@dataclass
class GateConfig:                                                        # all [CONFIRM]
    min_fall_trials: int = 12          # held-out natural-fall pairs needed as evidence
    min_nofall_trials: int = 8         # standing / sub-threshold pairs needed as evidence
    max_new_fall_rate: float = 0.0     # protected falls where unprotected did NOT fall
    min_median_reduction: float = 0.15 # median (1 - prot/unprot) of weighted force, fall pairs
    regression_tol: float = 0.10       # a pair "regresses" if prot > unprot*(1+tol)
    max_regression_frac: float = 0.20
    head_tol_frac: float = 0.05        # median head force: prot <= unprot*(1+tol) + slack
    head_slack_n: float = 25.0
    max_fragile_first_contact_frac: float = 0.05  # knee/shank touching the floor first
    max_median_tracking_err_rad: float = 0.15     # matches TRACKING_ERROR_TOL_RAD


@dataclass
class GateReport:
    passed: bool
    checks: Dict[str, Tuple[bool, str]]

    def __str__(self) -> str:
        lines = [("PASS" if self.passed else "FAIL") + " -- pose gate"]
        for k, (ok, msg) in self.checks.items():
            lines.append(f"  [{'ok' if ok else 'XX'}] {k}: {msg}")
        return "\n".join(lines)


def evaluate_gate(fall_pairs: List[Tuple[TrialResult, TrialResult]],
                  nofall_pairs: List[Tuple[TrialResult, TrialResult]],
                  cfg: GateConfig = GateConfig()) -> GateReport:
    """fall_pairs   : (unprotected, protected) for HELD-OUT conditions where a natural fall occurs.
       nofall_pairs : (unprotected, protected) for standing / sub-threshold conditions where the
                      unprotected robot does NOT fall (this is the false-alarm / do-no-harm set)."""
    checks: Dict[str, Tuple[bool, str]] = {}

    n_f, n_n = len(fall_pairs), len(nofall_pairs)
    checks["evidence"] = (n_f >= cfg.min_fall_trials and n_n >= cfg.min_nofall_trials,
                          f"{n_f} fall pairs (need {cfg.min_fall_trials}), {n_n} no-fall pairs (need {cfg.min_nofall_trials})")

    if n_n:
        new_falls = sum(1 for u, p in nofall_pairs if p.fell and not u.fell)
        rate = new_falls / n_n
        checks["do_no_harm"] = (rate <= cfg.max_new_fall_rate,
                                f"pose caused a fall in {new_falls}/{n_n} cases the robot would have survived (limit {cfg.max_new_fall_rate:.0%})")
    if n_f:
        red = np.array([1.0 - p.score / max(u.score, 1e-6) for u, p in fall_pairs])
        med = float(np.median(red))
        checks["median_reduction"] = (med >= cfg.min_median_reduction,
                                      f"median weighted-force reduction {med:.1%} (need >= {cfg.min_median_reduction:.0%})")
        reg = float(np.mean([p.score > u.score * (1 + cfg.regression_tol) for u, p in fall_pairs]))
        checks["regressions"] = (reg <= cfg.max_regression_frac,
                                 f"{reg:.0%} of conditions worse than unprotected (limit {cfg.max_regression_frac:.0%})")
        hu = float(np.median([u.peaks.get("head", 0.0) for u, _ in fall_pairs]))
        hp = float(np.median([p.peaks.get("head", 0.0) for _, p in fall_pairs]))
        checks["head_not_worse"] = (hp <= hu * (1 + cfg.head_tol_frac) + cfg.head_slack_n,
                                    f"median head force {hp:.0f} N vs unprotected {hu:.0f} N")
        prot_fell = [p for _, p in fall_pairs if p.fell]
        if prot_fell:
            fr = float(np.mean([p.first_contact_class == "knee_shank" for p in prot_fell]))
            checks["fragile_first_contact"] = (fr <= cfg.max_fragile_first_contact_frac,
                                               f"knee/shank hits the floor first in {fr:.0%} of protected falls "
                                               f"(limit {cfg.max_fragile_first_contact_frac:.0%})")
        te = float(np.median([p.tracking_err_rad for _, p in fall_pairs]))
        checks["tracking_error"] = (te <= cfg.max_median_tracking_err_rad,
                                    f"median peak tracking error {te:.3f} rad (limit {cfg.max_median_tracking_err_rad:.2f})")

    return GateReport(all(ok for ok, _ in checks.values()) and bool(checks), checks)


def decide(pose_ctrl: np.ndarray, static_violations: List[str], report: GateReport,
           stand_ctrl: np.ndarray) -> Tuple[str, np.ndarray]:
    """Library entry: the candidate pose only if statically valid AND gate passed; otherwise 'stand'."""
    if not static_violations and report.passed:
        return "pose", np.asarray(pose_ctrl, dtype=float)
    return "stand", np.asarray(stand_ctrl, dtype=float)


# =============================================================================
# 6. AUDIT CLI
# =============================================================================

def _iter_pose_vectors(obj, nu: int, path: str = "") -> Iterable[Tuple[str, np.ndarray]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}/{k}"
            if (isinstance(v, list) and len(v) == nu and all(isinstance(x, (int, float)) for x in v)
                    and "ctrl" in str(k).lower()):
                yield p, np.array(v, dtype=float)
            else:
                yield from _iter_pose_vectors(v, nu, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_pose_vectors(v, nu, f"{path}[{i}]")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--keyframe", type=int, default=0)
    ap.add_argument("--pose-json", default=None, help="pose_library.json; any list under a key containing 'ctrl' with nu entries is audited")
    args = ap.parse_args(argv)

    model = mujoco.MjModel.from_xml_path(args.model)
    spec = build_spec(model, args.keyframe)
    deg = np.degrees

    print("=" * 100)
    print("JOINT ENVELOPE  (deg).  sign=+1 means positive model angle == flexion/abduction; VERIFY once visually.")
    print("=" * 100)
    print(f"{'actuator':28s}{'type':13s}{'sign':>5s}{'model range':>20s}{'exo envelope':>20s}{'stand':>9s}")
    for j in spec.joints:
        if not j.is_leg:
            print(f"{j.act_name:28s}{'(not a leg joint - pinned, not searched)':s}")
            continue
        print(f"{j.act_name:28s}{j.jtype:13s}{j.sign:>5d}"
              f"{f'[{deg(j.model_lo):6.1f},{deg(j.model_hi):6.1f}]':>20s}"
              f"{f'[{deg(j.lo):6.1f},{deg(j.hi):6.1f}]':>20s}{deg(j.stand):9.1f}")

    print("\nBODY CLASSES (check that head/pelvis/knee_shank/thigh_hip/foot look right):")
    for b in range(1, model.nbody):
        n = _body_name(model, b)
        print(f"  {n:34s} -> {classify_body(n)}")

    lg = _leg_geoms(model)
    print(f"\nleg geoms used for clearance: left={len(lg['left'])} right={len(lg['right'])} "
          f"(mj_geomDistance {'available' if hasattr(mujoco, 'mj_geomDistance') else 'MISSING -> only separation check runs'})")
    for w in spec.warnings:
        print("WARNING:", w)

    stand_viol = check_pose_ctrl(spec, spec.stand_ctrl)
    print("\nSTAND pose self-check:", "OK" if not stand_viol else "PROBLEM -> " + "; ".join(stand_viol))

    if args.pose_json:
        with open(args.pose_json) as f:
            lib = json.load(f)
        found = list(_iter_pose_vectors(lib, model.nu))
        print(f"\nPOSE LIBRARY AUDIT: {len(found)} pose vectors found in {args.pose_json}")
        bad = 0
        for path, vec in found:
            v = check_pose_ctrl(spec, vec)
            bad += bool(v)
            print(f"  {'OK  ' if not v else 'FAIL'} {path}")
            for msg in v[:4]:
                print(f"         - {msg}")
        print(f"\n{bad}/{len(found)} poses violate the static validity spec.")
        if not found:
            print("  (no vectors found -- call check_pose_ctrl(spec, pose_ctrl) from your own code)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
