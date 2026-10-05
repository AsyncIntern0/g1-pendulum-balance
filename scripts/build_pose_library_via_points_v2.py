#!/usr/bin/env python3
"""
build_pose_library.py (v2, template-free)
=========================================
Goal: find, for each scenario (or scenario group) x velocity bin, the leg pose
that REDUCES GROUND-IMPACT the most -- not one that prevents the fall.

What changed vs v1
    * No pose_templates.py. CMA-ES searches the leg actuators directly, in a
      [0,1]^d box that maps onto the exo-safe joint envelope from
      validity_spec. Nothing is pre-shaped as squat / widen-crouch / step, so
      the search can pick whatever joint combination helps each scenario.
    * The gate is UNCHANGED (validity_spec.evaluate_gate + GATE_CONFIG_BY_BIN)
      and still runs once, on held-out conditions the search never saw.
    * Units: --granularity scenario  -> one entry per Phase-1 scenario (15)
              --granularity group     -> the SCENARIO_TO_GROUP groups (10-13 are each their own
                                          singleton group now -- see the v2.4 note below)
    * The objective now also sees the do-no-harm gate check (a small no-fall
      tuning set, disjoint from the gate's no-fall set), so CMA-ES can no
      longer "win" by finding a pose that itself topples a standing robot.
    * Bins where the unprotected robot (almost) never falls in the tuning set
      are skipped with a logged reason instead of burning a whole CMA-ES run
      on a gate that would fail on evidence anyway.
    * Multi-restart CMA-ES (restart 0 from 'stand', later restarts from wider
      random starts) -- a single search from 'stand' can miss a disconnected
      good basin (see technical-learnings: recovery stuck near 0).
    * Partial results are written after every bin; --resume skips finished ones.
    * Fixed: condition sampling used Python's salted hash(), so the tuning /
      select / holdout sets changed between runs. Now seeded via crc32.

v2.4: via-point (trajectory) search, added because BOTH earlier levers are
now closed with real evidence (seeded search plateaus within 1-2% of unrelated
poses; widening the box moved real joint limits by <0.0001 rad) -- the ~30-32%
ceiling is a property of a SINGLE STATIC pose, not of the search around it.
    * --traj-points N (default 1 = old static-pose behaviour, unchanged) lets
      a bin search N via-point poses plus (N-1) switch-time fractions instead
      of one fixed target -- e.g. an immediate brace pose, then a settle pose.
      Implemented as a fully separate path (TrajectorySpace / make_objective_traj
      / tune_pose_traj / gate_pose_traj) that reuses check_pose_ctrl, weighted_force
      and _diverse_topk but does not touch the tested static-pose functions.
    * REQUIRES HARNESS SUPPORT I cannot add without seeing phase3_pose_jerk_v7.py:
      make_run_trial_traj calls run_protected_trial(..., pose_traj=[(t, ctrl), ...])
      first; if that raises TypeError (no such parameter yet) it falls back to a
      SINGLE static call with the LAST via-point's ctrl, prints a one-time warning,
      and every trajectory entry's audit records regressed_to_static=True so a
      "trajectory" run that silently produced only static results is visible in
      the output, not just in a console log you might not have kept.
    * Runtime: pose_lookup(...) is UNCHANGED (static entries only). A new
      pose_traj_lookup(...) returns the [(t, ctrl), ...] list for a "pose_traj"
      entry (or a 1-point list for "stand"/"pose"), so callers don't need two
      code paths for the two kinds of entry.
    * Scenarios 10-13 no longer share one "unknown" group -- see the
      SCENARIO_TO_GROUP fix below. Untested claim worth flagging: since a
      trajectory is a strictly larger search than a static pose (N=1 recovers
      the old objective exactly), it should never do WORSE than the current
      library at the same holdout size; if a --traj-points 2 run ever passes
      the gate at a LOWER reduction than the current static entry for the same
      bin, something in the trajectory wiring is broken, not the concept.

v2.3 (driven by the full-scenario widebox run: median reduction 29.1% ->
31.3%, entirely from the tracking-penalty fix, NOT the wider box -- on the
real model the box changed by < 0.0001 rad, so that lever is now closed too)
    * gate_diagnostics sanity check widened: a bin now ALSO gets a printed
      WARNING when the median protected force is ~0 N while the robot still
      falls in most trials (frac_protected_still_fall high). That combination
      is physically impossible for a genuine impact and usually means the
      contact/force measurement isn't seeing the real landing (e.g. the
      observation window ends before ground contact, or geom IDs don't match
      after the scenario moves something like the floor). The OLD warning
      (near-0 fall fraction => maybe the fall was prevented) is kept as a
      separate, second condition.
    * --exclude units now leave a placeholder entry per bin instead of
      vanishing silently ({"kind": "stand", "excluded": true, "reason": ...})
      so the output JSON is self-documenting: a missing key used to be
      ambiguous between "excluded on purpose" and "some other error".
    * --regate-from <existing library json> [--regate-holdout-n N]: re-runs
      ONLY gate_pose on every non-skipped, non-excluded entry's already-found
      pose, with a (usually larger) holdout set -- no CMA-ES, so it's cheap.
      For bins that failed only on 'evidence' (too few held-out falls in a
      small set), a bigger holdout can resolve them without new search.

v2.2 (driven by the seeded v2.1 run: seed poses from unrelated scenarios all
scored within ~1-2% of CMA-ES's own winner -- the search itself was not the
bottleneck, the SEARCH BOX likely is)
    * Old box-finder moved ONE joint at a time from 'stand' with every other
      joint frozen -- an axis-aligned probe of what may be a non-axis-aligned
      feasible region. A joint that is only allowed to move far when a SECOND
      joint compensates (a common biomechanical coupling: e.g. hip flexion
      paired with ankle dorsiflexion) would be reported as far more limited
      than it really is.
    * New default: after the axis-only pass, bisect --box-directions (default
      250) random COMBINED directions across all searched joints, both signs.
      Each joint's lo/hi is the widest extent seen in EITHER pass, so the box
      only ever grows -- old seed poses (which fit inside the smaller v2.1 box)
      still map into [0,1] correctly under the new one.
    * This still produces a hyper-rectangle (an over-approximation), not the
      true polytope, so some box corners can still be infeasible -- exactly
      like before, make_objective()'s graded penalty on check_pose_ctrl is
      what actually keeps CMA-ES out of them. Widening the box only helps if
      the true envelope truly extends further; --box-directions 0 reproduces
      the old axis-only box exactly, for an apples-to-apples before/after run.

v2.1 (driven by the first full run: 17 pass / 10 fail / 18 skipped, median
reduction plateaued at ~29-30% in almost every bin)
    * Every bin used its FULL 720-evaluation budget -> CMA-ES never converged.
      The audit now records per-restart best score + stop reason so you can
      see whether more --maxiter still pays.
    * --seed-from <previous library json>: every pose found last time (passed
      OR failed the gate -- searched_delta_from_stand_rad is stored for both)
      is scored on the new bin's tuning set and the best ones become extra
      CMA-ES starting points. Order-independent, so it is safe to run units in
      parallel. audit['seed_scores'] doubles as a ceiling test: if seeds from
      OTHER scenarios all land near the same ratio, ~30% is a plateau of
      static poses, not a search failure.
    * Objective is now 50/50 mean + median of the per-draw cost (the gate
      judges the MEDIAN reduction; pure mean was optimising the tails).
    * Tracking penalty: weight 4 -> 10 and it starts at 90% of the limit, so
      poses land inside the budget instead of at 0.154 vs 0.150.
    * --holdout-n (default 24) to enlarge the held-out set for bins that fail
      only on 'evidence'; --exclude to drop scenarios that never fall (s11,s13).
    * gate_diagnostics per entry (fraction of protected trials that still fall,
      median protected force) + a printed WARNING when reduction >= 95%.
      Last run had s09_floor_drop at 100% -- verify before trusting it.
    Backward compatible: every new argument is optional, so an existing
    parallel wrapper that calls tune_pose / gate_pose / build_library still works.

Joint-limit source (search box)
    1. If ValiditySpec exposes per-actuator lower/upper arrays under a
       recognised attribute name (see _LO_ATTRS/_HI_ATTRS), those are used.
    2. Otherwise the box is derived by bisecting validity_spec.check_pose_ctrl
       one joint at a time out from 'stand' (the checker itself is the oracle).
    Either way the resolved box is printed at start-up -- eyeball it once.

USAGE
    python build_pose_library.py --model g1_pendulum.xml --mock
    python build_pose_library.py --model g1_pendulum.xml --live \\
        --granularity scenario --popsize 12 --maxiter 30 --restarts 2 \\
        --out pose_library_v2.json [--only forward,s02_push] [--resume]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib
import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import mujoco


from validity_spec import (
    GateConfig, GateReport, TrialResult, ValiditySpec, build_spec,
    check_pose_ctrl, decide, evaluate_gate, weighted_force, classify_body,
)

VELOCITY_BINS = ("low", "mid", "high")  # match these to your existing bin edges


# =============================================================================
# V2-ALIGNED CONTACT TRACKER
# =============================================================================
# The optimization must measure impact using the same physical attribution as
# the v2 viewer.  We KEEP the six gate classes (head, pelvis, knee_shank,
# thigh_hip, other, foot), but fix the geometry attribution:
#   * exact geom "ground" is the only floor geom;
#   * exact foot geoms from Phase-1 are ignored;
#   * exact body "pelvis" is pelvis;
#   * exact body "pendulum_bob" is head;
#   * remaining non-foot bodies use the existing six-class name rules.
#
# Unlike the old ContactTracker, peak force is the MAXIMUM INDIVIDUAL ground
# contact for each class, matching the v2 viewer's contact_peak_forces logic.
# This prevents multiple simultaneous contacts from being silently summed into
# a larger class force that the v2 viewer would not report.

class V2AlignedContactTracker:
    """Six-class gate tracker with v2 physical contact attribution."""

    def __init__(self, model: mujoco.MjModel, p1):
        self.model = model

        ground_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground"))
        if ground_id < 0:
            raise RuntimeError('V2-aligned contact tracking requires an exact geom named "ground".')
        self.ground_geom_id = ground_id

        foot_ids = p1.get_foot_geom_ids(model)
        self.foot_geom_ids = {int(g) for g in foot_ids}

        pelvis_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
        head_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pendulum_bob"))
        if pelvis_id < 0:
            raise RuntimeError('V2-aligned contact tracking requires an exact body named "pelvis".')
        if head_id < 0:
            raise RuntimeError('V2-aligned contact tracking requires an exact body named "pendulum_bob".')

        self.pelvis_body_id = pelvis_id
        self.head_body_id = head_id
        self.geom_body_ids = np.asarray(model.geom_bodyid, dtype=int)
        self.geom_body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b)) or ""
            for b in self.geom_body_ids
        ]
        self._f6 = np.zeros(6, dtype=float)
        self.reset()

    def reset(self) -> None:
        self.peak = {
            "head": 0.0,
            "pelvis": 0.0,
            "knee_shank": 0.0,
            "thigh_hip": 0.0,
            "other": 0.0,
            "foot": 0.0,
        }
        self.first_time = {k: None for k in self.peak}

    def _class_for_geom(self, geom_id: int) -> str:
        if geom_id in self.foot_geom_ids:
            return "foot"

        body_id = int(self.geom_body_ids[geom_id])
        if body_id == self.pelvis_body_id:
            return "pelvis"
        if body_id == self.head_body_id:
            return "head"

        # Keep the existing gate's knee/thigh/other classes for all remaining
        # bodies.  The important correction is that pelvis/head are exact IDs,
        # so e.g. left_hip_yaw_link is NOT silently treated as pelvis.
        return classify_body(self.geom_body_names[geom_id])

    def update(self, data: mujoco.MjData) -> None:
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)

            if g1 == self.ground_geom_id:
                other = g2
            elif g2 == self.ground_geom_id:
                other = g1
            else:
                continue

            cls = self._class_for_geom(other)
            if cls == "foot":
                # Same physical exclusion as the v2 viewer.
                continue

            mujoco.mj_contactForce(self.model, data, i, self._f6)
            force_n = float(np.linalg.norm(self._f6[:3]))

            # v2 semantics: peak individual contact, not sum of all contacts
            # in the class at one timestep.
            if force_n > self.peak[cls]:
                self.peak[cls] = force_n

            if force_n > 1.0 and self.first_time[cls] is None:
                self.first_time[cls] = float(data.time)

    @property
    def first_nonfoot_class(self) -> Optional[str]:
        candidates = [
            (t, cls) for cls, t in self.first_time.items()
            if t is not None and cls != "foot"
        ]
        return min(candidates)[1] if candidates else None



# Skip a bin's CMA-ES run when fewer than this many of its tuning draws
# naturally fall when unprotected. Gate needs >= min_fall_trials (6-12) genuine
# falls out of 24 held-out draws, i.e. a fall rate of 25-50%; <2/12 in tuning
# means the bin is far below that.
MIN_TUNING_FALLS = 2
N_NOFALL_TUNE = 4  # do-no-harm draws inside the objective (gate uses its own, disjoint 10)

# Objective shaping (defaults; overridable per call and from the CLI)
OBJ_TRACKING_WEIGHT = 10.0   # v2: 4.0. First run missed the 0.15 rad limit by 0.004 in two bins.
OBJ_TRACKING_MARGIN = 0.90   # penalise from 90% of the limit, so the median lands inside it
OBJ_MEDIAN_WEIGHT = 0.5      # cost = (1-w)*mean + w*median over falling draws (gate uses the median)


# =============================================================================
# Search space: the exo-safe joint box, normalised to [0,1]^d
# =============================================================================

_LO_ATTRS = ("ctrl_lo", "ctrl_low", "ctrl_min", "ctrl_lower", "q_lo", "q_min", "joint_lo", "lo", "lower", "lb")
_HI_ATTRS = ("ctrl_hi", "ctrl_high", "ctrl_max", "ctrl_upper", "q_hi", "q_max", "joint_hi", "hi", "upper", "ub")


@dataclass
class SearchSpace:
    stand: np.ndarray        # full ctrl vector (len = model.nu)
    idx: np.ndarray          # ctrl indices that CMA-ES is allowed to move
    lo: np.ndarray           # per-searched-joint lower bound (rad)
    hi: np.ndarray           # per-searched-joint upper bound (rad)
    names: List[str]         # actuator names of the searched joints
    source: str              # where lo/hi came from (printed + stored in the JSON)

    @property
    def dim(self) -> int:
        return len(self.idx)

    def to_ctrl(self, x: np.ndarray) -> np.ndarray:
        ctrl = self.stand.copy()
        ctrl[self.idx] = self.lo + np.clip(x, 0.0, 1.0) * (self.hi - self.lo)
        return ctrl

    def x_stand(self) -> np.ndarray:
        return self.x_of_ctrl(self.stand)

    def x_of_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        span = np.maximum(self.hi - self.lo, 1e-9)
        return np.clip((np.asarray(ctrl, float)[self.idx] - self.lo) / span, 0.0, 1.0)

    def describe(self) -> str:
        rows = [f"  {n:28s} stand={self.stand[i]:+.3f}  lo={l:+.3f}  hi={h:+.3f}"
                for n, i, l, h in zip(self.names, self.idx, self.lo, self.hi)]
        return f"search box ({self.dim} joints, source: {self.source})\n" + "\n".join(rows)


def _spec_bounds(spec: ValiditySpec, nu: int, idx: np.ndarray):
    for lo_n, hi_n in zip(_LO_ATTRS, _HI_ATTRS):
        lo, hi = getattr(spec, lo_n, None), getattr(spec, hi_n, None)
        if lo is None or hi is None:
            continue
        try:
            lo, hi = np.asarray(lo, float).ravel(), np.asarray(hi, float).ravel()
        except (TypeError, ValueError):
            continue
        if lo.size == nu and hi.size == nu:
            return lo[idx], hi[idx], f"spec.{lo_n}/{hi_n}"
        if lo.size == len(idx) and hi.size == len(idx):
            return lo, hi, f"spec.{lo_n}/{hi_n}"
    return None


def _n_violations(v) -> int:
    try:
        return len(v)
    except TypeError:
        return 1 if v else 0


def _oracle_bounds_axis(spec: ValiditySpec, idx: np.ndarray, stand: np.ndarray, t_max: float = 1.5):
    """Largest single-joint excursion from 'stand' (each direction) that
    check_pose_ctrl still accepts, with every OTHER searched joint pinned at
    stand. Conservative: needs nothing from ValiditySpec beyond stand_ctrl +
    the checker, but under-estimates any joint whose real range depends on a
    compensating move elsewhere (see _oracle_bounds, which extends this)."""
    if _n_violations(check_pose_ctrl(spec, stand)):
        raise RuntimeError("check_pose_ctrl rejects spec.stand_ctrl itself -- cannot derive a search box")
    lo, hi = np.empty(len(idx)), np.empty(len(idx))
    for k, j in enumerate(idx):
        for sign, out in ((+1.0, hi), (-1.0, lo)):
            c = stand.copy()
            c[j] = stand[j] + sign * t_max
            if not _n_violations(check_pose_ctrl(spec, c)):
                a = t_max
            else:
                a, b = 0.0, t_max
                for _ in range(14):
                    m = 0.5 * (a + b)
                    c[j] = stand[j] + sign * m
                    if _n_violations(check_pose_ctrl(spec, c)):
                        b = m
                    else:
                        a = m
            out[k] = stand[j] + sign * a
    return lo, hi, f"oracle-axis: bisect one joint at a time from stand (cap +/-{t_max} rad)"


def _oracle_bounds(spec: ValiditySpec, idx: np.ndarray, stand: np.ndarray, t_max: float = 1.5,
                   n_directions: int = 250, seed: int = 0, verbose: bool = True):
    """Axis-only bounds (see _oracle_bounds_axis), THEN widened by bisecting
    n_directions random unit vectors spanning ALL searched joints at once
    (both signs). A joint's final lo/hi is the widest value seen in either
    pass -- the box only grows, so an old, smaller box's poses still map into
    [0,1] here. n_directions=0 reproduces the old axis-only box exactly.

    This is still a bounding hyper-rectangle, not the true feasible polytope,
    so corners can still be infeasible; CMA-ES's graded penalty (see
    make_objective) handles that, same as before."""
    lo, hi, axis_src = _oracle_bounds_axis(spec, idx, stand, t_max)
    n_widened = 0
    if n_directions > 0:
        rng = np.random.default_rng(seed)
        d = len(idx)
        for _ in range(n_directions):
            v = rng.normal(size=d)
            v /= np.linalg.norm(v) + 1e-12
            for sign in (+1.0, -1.0):
                c = stand.copy()
                c[idx] = stand[idx] + sign * t_max * v
                if not _n_violations(check_pose_ctrl(spec, c)):
                    a = t_max
                else:
                    a, b = 0.0, t_max
                    for _ in range(14):
                        m = 0.5 * (a + b)
                        c[idx] = stand[idx] + sign * m * v
                        if _n_violations(check_pose_ctrl(spec, c)):
                            b = m
                        else:
                            a = m
                point = stand[idx] + sign * a * v
                widened = (point < lo) | (point > hi)
                n_widened += int(np.count_nonzero(widened))
                lo, hi = np.minimum(lo, point), np.maximum(hi, point)
    src = (axis_src if n_directions == 0 else
          f"oracle: axis pass + {n_directions} random-direction bisections, widest-of-both per joint "
          f"(cap +/-{t_max} rad)")
    if verbose and n_directions > 0:
        print(f"  box-finder: {n_directions} random directions widened {n_widened} joint-bound(s) "
              f"beyond the axis-only pass")
    return lo, hi, src


def build_search_space(spec: ValiditySpec, model, box_directions: int = 250, box_seed: int = 0) -> SearchSpace:
    import mujoco
    nu = int(model.nu)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}" for i in range(nu)]
    # Pendulum actuators stay pinned at 'stand' (the wearer's torso is not
    # something the exoskeleton can move) -- same rule as v6/v7.
    idx = np.array([i for i, n in enumerate(names) if "pendulum" not in n.lower()], dtype=int)
    stand = np.asarray(spec.stand_ctrl, float).copy()

    found = _spec_bounds(spec, nu, idx)
    lo, hi, src = found if found else _oracle_bounds(spec, idx, stand, n_directions=box_directions, seed=box_seed)

    if hasattr(model, "actuator_ctrllimited"):  # never exceed the actuator's own ctrlrange
        for k, j in enumerate(idx):
            if model.actuator_ctrllimited[j]:
                lo[k] = max(lo[k], model.actuator_ctrlrange[j, 0])
                hi[k] = min(hi[k], model.actuator_ctrlrange[j, 1])
    lo = np.minimum(lo, stand[idx])
    hi = np.maximum(hi, stand[idx])

    keep = (hi - lo) > 1e-3  # freeze joints with no usable range
    return SearchSpace(stand=stand, idx=idx[keep], lo=lo[keep], hi=hi[keep],
                       names=[n for n, k in zip([names[i] for i in idx], keep) if k], source=src)


@dataclass
class TrajectorySpace:
    """N via-point poses in the SAME per-joint box as `base` (a SearchSpace),
    plus (N-1) switch-time fractions of `duration_s`. N=1 has zero switch-time
    parameters and is mathematically identical to searching `base` alone --
    trajectory mode is a strict superset of static mode, not a different
    search."""
    base: SearchSpace
    n_points: int
    duration_s: float

    @property
    def dim(self) -> int:
        return self.n_points * self.base.dim + max(0, self.n_points - 1)

    def x_stand(self) -> np.ndarray:
        pts = np.tile(self.base.x_stand(), self.n_points)
        fracs = np.linspace(0.0, 1.0, self.n_points + 1)[1:-1] if self.n_points > 1 else np.array([])
        return np.concatenate([pts, fracs])

    def to_traj(self, x: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        d = self.base.dim
        pts = x[: self.n_points * d].reshape(self.n_points, d)
        ctrls = [self.base.to_ctrl(p) for p in pts]
        # The N via-points are TARGETS reached sequentially during the
        # trigger-to-impact window.  Therefore the first target is NOT at
        # t=0: at t=0 the real robot is at q_start.  For N=3:
        #   q_start --min-jerk--> Pose A --min-jerk--> Pose B --min-jerk--> Pose C
        #                         t1                  t2                  T
        # Only N-1 switch/arrival times are optimized; the final pose is
        # deliberately fixed at the end of the available recovery window.
        if self.n_points == 1:
            times = [self.duration_s]
        else:
            # Keep every minimum-jerk segment executable.  With a 300 ms
            # recovery window, 30 ms per segment gives three 90-ms-or-longer
            # segments for N=3. CMA-ES still chooses the relative timing.
            min_segment_s = min(0.03, self.duration_s / (self.n_points + 1))
            g = min_segment_s / self.duration_s
            raw = np.sort(np.clip(x[self.n_points * d:], 0.0, 1.0))
            available = max(0.0, 1.0 - self.n_points * g)
            fracs = np.array([g * (i + 1) + raw[i] * available
                              for i in range(self.n_points - 1)], dtype=float)
            times = [float(f * self.duration_s) for f in fracs] + [self.duration_s]
        return list(zip(times, ctrls))

    def x_seed(self, seed_point0_ctrl: np.ndarray) -> np.ndarray:
        """Warm start: via-point 0 = the given (already-found) pose, every
        later via-point = stand, switch times evenly spaced."""
        x = self.x_stand()
        x[: self.base.dim] = self.base.x_of_ctrl(seed_point0_ctrl)
        return x


def check_traj(spec: ValiditySpec, traj: List[Tuple[float, np.ndarray]]) -> int:
    """Total violation count summed over every via-point's ctrl."""
    return sum(_n_violations(check_pose_ctrl(spec, ctrl)) for _, ctrl in traj)


# =============================================================================
# Trial runners
# =============================================================================

def run_one_trial_MOCK(spec: ValiditySpec, ctrl: np.ndarray, scenario: str,
                       condition: dict, passive_pendulum: bool = False) -> TrialResult:
    """Stand-in physics for exercising the CMA-ES + gate wiring only -- NOT real
    dynamics. Bigger deviation from 'stand' lowers fall probability and force."""
    rng = np.random.default_rng(condition.get("seed", 0))
    hazard = condition.get("hazard", 0.5)
    depth = float(np.clip(np.abs(np.asarray(ctrl) - np.asarray(spec.stand_ctrl)).mean() / 0.3, 0, 1))
    fell = rng.random() < hazard * (1.0 - 0.6 * depth)
    base = 3000.0 * (1.0 - 0.55 * depth) + rng.normal(0, 150)
    other = max(0.0, base) if fell else 0.0
    return TrialResult(fell=fell, peaks={"head": 0.0, "pelvis": 0.0, "other": other,
                                         "knee_shank": other * 0.3, "thigh_hip": 0.0, "foot": 50.0},
                       first_contact_class="other", tracking_err_rad=0.02 + 0.05 * depth)


def make_run_trial(model, p1, impact_ids, snapshot, trigger_lead_s: float = 0.3) -> Callable:
    """Binds the per-run-fixed arguments once and returns run_trial(spec, ctrl,
    scenario, condition). `condition` must carry 'scenario_tuple' (the actual
    p1.SCENARIOS entry for THIS draw) plus magnitude/direction_deg/timing_phase_s.

    Requires run_protected_trial to accept an optional `contact_tracker=None`
    kwarg and call `contact_tracker.update(d)` once per step. Falls back to the
    flat pelvis/head/other split with a one-time warning if it doesn't."""
    _warned = {"once": False}

    def run_trial(spec: ValiditySpec, ctrl: np.ndarray, scenario: str, condition: dict) -> TrialResult:
        from phase3_pose_jerk_v7_real import run_protected_trial  # local import: only needed in live mode

        scen_tuple = condition.get("scenario_tuple")
        if scen_tuple is None:
            raise KeyError("condition dict is missing 'scenario_tuple' -- use make_condition_iters")

        ct = V2AlignedContactTracker(model, p1)
        kwargs = dict(
            model=model, scenario=scen_tuple, magnitude=condition["magnitude"],
            direction_deg=condition["direction_deg"], timing_phase_s=condition["timing_phase_s"],
            pose_ctrl=np.asarray(ctrl), trigger_lead_s=condition.get("trigger_lead_s", trigger_lead_s),
            impact_ids=impact_ids, snapshot=snapshot, p1=p1,
        )
        try:
            r = run_protected_trial(**kwargs, contact_tracker=ct)
            peaks = dict(ct.peak)
            first_contact = ct.first_nonfoot_class
        except TypeError:
            if not _warned["once"]:
                print("[warn] run_protected_trial has no contact_tracker param yet -- falling back to "
                      "the flat pelvis/head/other split; the gate's fragile-contact check is blind.")
                _warned["once"] = True
            r = run_protected_trial(**kwargs)
            peaks = {"head": r.peak_force_head, "pelvis": r.peak_force_pelvis,
                     "other": r.peak_force_other, "knee_shank": 0.0, "thigh_hip": 0.0, "foot": 0.0}
            first_contact = None

        return TrialResult(fell=r.fell, peaks=peaks, first_contact_class=first_contact,
                           tracking_err_rad=r.peak_tracking_error_rad)

    return run_trial


def run_traj_trial_MOCK(spec: ValiditySpec, traj: List[Tuple[float, np.ndarray]], scenario: str,
                        condition: dict, passive_pendulum: bool = False) -> TrialResult:
    """Wiring-test stand-in: 'depth' is the mean deviation-from-stand across
    ALL via-points, so a trajectory with more/bigger via-points reads as more
    protective than a single static pose -- enough to exercise the larger
    parameter vector, NOT a claim about real dynamics."""
    rng = np.random.default_rng(condition.get("seed", 0))
    hazard = condition.get("hazard", 0.5)
    depths = [np.abs(np.asarray(c) - np.asarray(spec.stand_ctrl)).mean() / 0.3 for _, c in traj]
    depth = float(np.clip(np.mean(depths), 0, 1))
    fell = rng.random() < hazard * (1.0 - 0.6 * depth)
    base = 3000.0 * (1.0 - 0.55 * depth) + rng.normal(0, 150)
    other = max(0.0, base) if fell else 0.0
    return TrialResult(fell=fell, peaks={"head": 0.0, "pelvis": 0.0, "other": other,
                                         "knee_shank": other * 0.3, "thigh_hip": 0.0, "foot": 50.0},
                       first_contact_class="other", tracking_err_rad=0.02 + 0.05 * depth), False


def make_run_trial_traj(model, p1, impact_ids, snapshot, trigger_lead_s: float = 0.3) -> Callable:
    """Live counterpart to run_traj_trial_MOCK. Tries run_protected_trial(...,
    pose_traj=[(t, ctrl), ...]) first; if the harness doesn't accept that
    kwarg yet, falls back to a SINGLE static call using the LAST via-point's
    ctrl and flags the result so a silently-degraded run is visible in the
    saved JSON, not just a console warning you might not have kept."""
    _warned = {"traj": False}

    def run_trial_traj(spec: ValiditySpec, traj: List[Tuple[float, np.ndarray]], scenario: str,
                       condition: dict) -> Tuple[TrialResult, bool]:
        from phase3_pose_jerk_v7_real import run_protected_trial
        scen_tuple = condition.get("scenario_tuple")
        if scen_tuple is None:
            raise KeyError("condition dict is missing 'scenario_tuple' -- use make_condition_iters")
        ct = V2AlignedContactTracker(model, p1)
        base_kwargs = dict(model=model, scenario=scen_tuple, magnitude=condition["magnitude"],
                           direction_deg=condition["direction_deg"], timing_phase_s=condition["timing_phase_s"],
                           trigger_lead_s=condition.get("trigger_lead_s", trigger_lead_s),
                           impact_ids=impact_ids, snapshot=snapshot, p1=p1)
        try:
            r = run_protected_trial(**base_kwargs, pose_traj=traj, contact_tracker=ct)
            regressed = False
        except TypeError:
            if not _warned["traj"]:
                print("[warn] run_protected_trial has no pose_traj= parameter yet -- trajectory search is "
                      "regressing to a single static call with the LAST via-point's pose. Results are still "
                      "saved but ARE NOT a real trajectory evaluation; add pose_traj= support to the harness "
                      "(or share phase3_pose_jerk_v7.py) to get real via-point results.")
                _warned["traj"] = True
            r = run_protected_trial(**base_kwargs, pose_ctrl=np.asarray(traj[-1][1]), contact_tracker=ct)
            regressed = True
        return TrialResult(fell=r.fell, peaks=dict(ct.peak), first_contact_class=ct.first_nonfoot_class,
                           tracking_err_rad=r.peak_tracking_error_rad), regressed

    return run_trial_traj


# =============================================================================
# Units (scenario / group) and condition iterators
# =============================================================================

#   id  category   fn_name            nominal_dir -> group
SCENARIO_TO_GROUP: Dict[int, str] = {
    1: "forward",       # push 0deg
    2: "backward",      # push 180deg
    3: "left",          # push 90deg
    4: "right",         # push 270deg
    5: "left",          # push 45deg
    6: "right",         # push 315deg
    7: "floor_tilt",    # floor_tilt_pitch
    8: "floor_tilt",    # floor_tilt_roll
    9: "sudden_load",   # floor_drop
    10: "low_friction",     # own group: friction loss has nothing in common with 11-13
    11: "actuator_fault",   # own group: excluded by default (see MIN_TUNING_FALLS) -- never falls in this harness
    12: "actuator_stuck",   # own group: fails the tracking-error gate at 'high' -- a real, different limitation
    13: "asymmetric_gain",  # own group: excluded by default -- never falls in this harness
    14: "forward",      # trip
    15: "sudden_load",  # sudden_load
}


def group_scenario_ids(p1=None) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for sid, group in SCENARIO_TO_GROUP.items():
        out.setdefault(group, []).append(sid)
    return out


def make_units(granularity: str, p1=None) -> Dict[str, List[int]]:
    """unit key -> list of Phase-1 scenario ids searched together.
    'group'    : the 7 SCENARIO_TO_GROUP groups (a pose must work for every id in the group)
    'scenario' : one unit per scenario id, keyed s01_<fn_name>, s02_<fn_name>, ..."""
    if granularity == "group":
        return group_scenario_ids(p1)
    names = {t[0]: t[2] for t in p1.SCENARIOS} if p1 is not None else {}
    return {(f"s{sid:02d}_{names[sid]}" if sid in names else f"s{sid:02d}"): [sid]
            for sid in sorted(SCENARIO_TO_GROUP)}


def pose_lookup(library: dict, unit_key: str, bin_name: str) -> Optional[np.ndarray]:
    """Runtime helper: returns the pose_ctrl for (unit, bin), or None meaning
    'command stand_ctrl' (entry missing, or it did not pass the gate).
    Also falls back to an 'unknown/<bin>' entry if the unit is absent -- a pure
    safety net for a scenario id this library has never seen (e.g. a new
    failure mode added after this library was built), NOT a bucket that 10-13
    get folded into: each now has its own group (see SCENARIO_TO_GROUP), since
    lumping unrelated failure modes -- friction loss, a stuck actuator, an
    asymmetric gain fault -- under one shared pose was never physically sound,
    least of all now that poses are searched per-scenario instead of templated."""
    entry = library.get(f"{unit_key}/{bin_name}") or library.get(f"unknown/{bin_name}")
    if entry is None or entry["kind"] != "pose":
        return None
    return np.array(entry["pose_ctrl"])


def pose_traj_lookup(library: dict, unit_key: str, bin_name: str,
                     stand_ctrl: Optional[np.ndarray] = None) -> Optional[List[Tuple[float, np.ndarray]]]:
    """Runtime helper for a library built with --traj-points > 1 (or a mix of
    trajectory and static entries): returns [(t, ctrl), ...] for ANY entry kind
    ('pose_traj', 'pose', or 'stand'), so a caller doesn't need two code paths
    for the two kinds of library. A 'pose' entry becomes a 1-point trajectory
    held for the whole trial (t=0); a missing/'stand' entry returns a 1-point
    stand trajectory (using `stand_ctrl` if given, else the entry's own
    pose_ctrl/stand fallback) rather than None, since callers of THIS function
    generally want something to command outright, not a sentinel to branch on."""
    entry = library.get(f"{unit_key}/{bin_name}") or library.get(f"unknown/{bin_name}")
    if entry is not None and entry.get("kind") == "pose_traj" and entry.get("pose_traj"):
        return [(float(t), np.array(ctrl)) for t, ctrl in entry["pose_traj"]]
    if entry is not None and entry.get("kind") == "pose" and entry.get("pose_ctrl") is not None:
        return [(0.0, np.array(entry["pose_ctrl"]))]
    base = stand_ctrl if stand_ctrl is not None else (
        np.array(entry["pose_ctrl"]) if entry is not None and entry.get("pose_ctrl") is not None else None)
    return [(0.0, base)] if base is not None else None


def unit_key_for_scenario(library: dict, scenario_id: int) -> Optional[str]:
    """Map a classifier-predicted Phase-1 scenario id to the library unit key
    (works for both granularities because every entry stores its scenario_ids)."""
    for v in library.values():
        if scenario_id in v.get("scenario_ids", []):
            return v["unit"]
    return None


# --- mock iterators (wiring test only) ---------------------------------------

def iter_tuning_conditions_MOCK(unit: str, bin_name: str) -> List[dict]:
    return [{"seed": i, "hazard": 0.95} for i in range(12)]


def iter_select_conditions_MOCK(unit: str, bin_name: str) -> List[dict]:
    return [{"seed": 300 + i, "hazard": 0.95} for i in range(10)]


def iter_holdout_conditions_MOCK(unit: str, bin_name: str) -> List[dict]:
    return [{"seed": 100 + i, "hazard": 0.95} for i in range(24)]


def iter_nofall_conditions_MOCK(unit: str, bin_name: str, tag: str = "gate", n: Optional[int] = None) -> List[dict]:
    base, default_n = (200, 10) if tag == "gate" else (500, N_NOFALL_TUNE)
    return [{"seed": base + i, "hazard": 0.0} for i in range(n or default_n)]


# --- real iterators ------------------------------------------------------------

# Must match the inline jitters list inside p1.build_trial_plan.
JITTERS_DEG = [-30, -15, 15, 30]
# Velocity-bin -> fraction of each scenario's MAGNITUDE_RANGES span. Proxy
# (bigger magnitude ~ faster fall), not measured trigger velocity.    [CONFIRM]
BIN_MAGNITUDE_FRACTIONS = {"low": (0.0, 0.33), "mid": (0.33, 0.66), "high": (0.66, 1.0)}


def make_condition_iters(p1, registry: Dict[str, List[tuple]], holdout_n: int = 24):
    """Returns (iter_tuning, iter_select, iter_holdout, iter_nofall). `registry`
    maps unit key -> LIST of p1.SCENARIOS tuples; each condition draws a random
    one, so a group's sets genuinely span all of its member scenarios.
    iter_nofall(unit, bin, tag='gate'|'nofall_tune', n=None) -- different tags
    give disjoint draws (gate set vs. the small set the objective uses)."""

    def _sample(unit: str, bin_name: str, n: int, seed_tag: str, nofall: bool = False) -> List[dict]:
        candidates = registry[unit]
        # crc32, NOT hash(): Python's str hash is salted per process, which made
        # the tuning/select/holdout sets differ from run to run.
        rng = np.random.default_rng(zlib.crc32(f"{unit}|{bin_name}|{seed_tag}".encode()))
        conds: List[dict] = []
        if nofall:
            _, _, _, nominal_dir = candidates[0]
            conds.append({"scenario_tuple": candidates[0], "magnitude": 0.0,
                          "direction_deg": float(nominal_dir or 0.0), "timing_phase_s": 0.3})
            n -= 1
        for _ in range(n):
            scen_tuple = candidates[int(rng.integers(len(candidates)))]
            _, _, fn_name, nominal_dir = scen_tuple
            lo, hi = p1.MAGNITUDE_RANGES[fn_name]
            if nofall:
                mag = float(rng.uniform(0.0, 0.5 * lo))
            else:
                f0, f1 = BIN_MAGNITUDE_FRACTIONS[bin_name]
                mag = float(rng.uniform(lo + f0 * (hi - lo), lo + f1 * (hi - lo)))
            timing = float(rng.uniform(0.0, 0.8))
            direction = float((nominal_dir + rng.choice(JITTERS_DEG)) % 360) if nominal_dir is not None else 0.0
            conds.append({"scenario_tuple": scen_tuple, "magnitude": mag,
                          "direction_deg": direction, "timing_phase_s": timing})
        return conds

    iter_tuning = lambda u, b: _sample(u, b, 12, "tune")
    iter_select = lambda u, b: _sample(u, b, 10, "select")     # ranks the shortlist only
    iter_holdout = lambda u, b: _sample(u, b, holdout_n, "holdout")   # the gate, once
    iter_nofall = lambda u, b, tag="gate", n=None: _sample(
        u, b, n or (10 if tag == "gate" else N_NOFALL_TUNE), f"nofall_{tag}", nofall=True)
    return iter_tuning, iter_select, iter_holdout, iter_nofall


# Per-bin gate configs (UNCHANGED from v1). 'low' is below the calibrated
# 50%-fall crossover, so genuine falls are rarer and need less evidence.  [CONFIRM]
GATE_CONFIG_BY_BIN: Dict[str, GateConfig] = {
    "low": GateConfig(min_fall_trials=6),
    "mid": GateConfig(min_fall_trials=10),
    "high": GateConfig(),  # default min_fall_trials=12
}


# =============================================================================
# Objective + tuning
# =============================================================================

def make_objective(spec: ValiditySpec, space: SearchSpace, unit: str, bin_name: str,
                   run_trial: Callable, iter_cond: Callable, iter_nofall: Callable,
                   cfg: GateConfig, tracking_weight: float = OBJ_TRACKING_WEIGHT,
                   tracking_margin: float = OBJ_TRACKING_MARGIN,
                   median_weight: float = OBJ_MEDIAN_WEIGHT
                   ) -> Tuple[Optional[Callable[[np.ndarray], float]], int]:
    """Objective mirrors what evaluate_gate checks, on TUNING draws only:
        (1-w)*mean + w*median, over naturally-falling draws, of
            force ratio (protected / unprotected weighted force)   -> median_reduction
          + tracking_weight * tracking error above margin*limit     -> tracking_error
          + head-over-unprotected / unprotected score               -> head_not_worse
          + 0.5 if first non-foot contact is knee/shank             -> fragile_first_contact
        + fraction of no-fall draws where the POSE makes the robot fall -> do_no_harm
    Returns (objective, n_natural_falls). objective is None when fewer than
    MIN_TUNING_FALLS tuning draws fall unprotected (caller skips the bin)."""
    conditions = iter_cond(unit, bin_name)
    unprotected = [(c, run_trial(spec, spec.stand_ctrl, unit, c)) for c in conditions]
    falling = [(c, u) for c, u in unprotected if u.fell]
    if len(falling) < MIN_TUNING_FALLS:
        return None, len(falling)

    nofall = []
    for c in iter_nofall(unit, bin_name, "nofall_tune", N_NOFALL_TUNE):
        u = run_trial(spec, spec.stand_ctrl, unit, c)
        if not u.fell:  # only conditions where 'stand' is genuinely fine can show harm
            nofall.append(c)

    def objective(x: np.ndarray) -> float:
        ctrl = space.to_ctrl(x)
        nv = _n_violations(check_pose_ctrl(spec, ctrl))
        if nv:
            return 10.0 + nv  # graded (not a flat 1e6 wall) so CMA-ES still gets a slope out of it
        costs = []
        for cond, u in falling:
            p = run_trial(spec, ctrl, unit, cond)
            u_score = max(weighted_force(u.peaks), 1e-6)
            ratio = weighted_force(p.peaks) / u_score
            tracking_pen = tracking_weight * max(
                0.0, p.tracking_err_rad - tracking_margin * cfg.max_median_tracking_err_rad)
            head_over = max(0.0, p.peaks.get("head", 0.0)
                            - u.peaks.get("head", 0.0) * (1 + cfg.head_tol_frac) - cfg.head_slack_n)
            head_pen = head_over / u_score
            fragile_pen = 0.5 if p.first_contact_class == "knee_shank" else 0.0
            costs.append(ratio + tracking_pen + head_pen + fragile_pen)
        harm = 0.0
        if nofall:
            harm = float(np.mean([1.0 if run_trial(spec, ctrl, unit, c).fell else 0.0 for c in nofall]))
        agg = (1.0 - median_weight) * float(np.mean(costs)) + median_weight * float(np.median(costs))
        return agg + harm

    return objective, len(falling)


def _diverse_topk(evals: List[Tuple[np.ndarray, float]], k: int, min_dist: float) -> List[np.ndarray]:
    """Greedy top-k by tuning score, skipping candidates within min_dist (L2 in
    the normalised box) of one already picked. Drawn from the WHOLE trace, not
    the last generation, which converges to near-duplicates."""
    ranked = sorted(evals, key=lambda t: t[1])
    picked: List[np.ndarray] = []
    for x, _ in ranked:
        if all(float(np.linalg.norm(x - p)) >= min_dist for p in picked):
            picked.append(x)
        if len(picked) >= k:
            break
    if not picked and ranked:
        picked = [ranked[0][0]]
    return picked


def load_seed_ctrls(path: str, space: SearchSpace) -> List[np.ndarray]:
    """Every pose a previous run searched -- passed OR failed the gate -- as a
    full ctrl vector. Uses searched_delta_from_stand_rad (stored for every
    non-skipped entry), so failed-gate poses are recovered too, not just the
    'pose_ctrl' field (which is 'stand' for a failed bin). De-duplicated."""
    with open(path) as f:
        lib = json.load(f)
    seeds: List[np.ndarray] = []
    for k, v in lib.items():
        if k.startswith("_"):
            continue
        dl = v.get("searched_delta_from_stand_rad")
        if not dl or any(n not in dl for n in space.names):
            continue
        ctrl = space.stand.copy()
        ctrl[space.idx] = np.clip(space.stand[space.idx] + np.array([dl[n] for n in space.names]),
                                  space.lo, space.hi)
        if not any(np.allclose(ctrl, q, atol=1e-4) for q in seeds):
            seeds.append(ctrl)
    return seeds


def tune_pose(spec: ValiditySpec, space: SearchSpace, unit: str, bin_name: str, run_trial: Callable,
              iter_tuning: Callable, iter_select: Callable, iter_nofall: Callable, cfg: GateConfig,
              popsize: int = 12, maxiter: int = 30, restarts: int = 2, seed: int = 0, top_k: int = 4,
              seed_ctrls: Optional[Sequence[np.ndarray]] = None, n_seed_starts: int = 2,
              obj_kwargs: Optional[dict] = None, max_seed_eval: int = 40
              ) -> Tuple[Optional[np.ndarray], dict]:
    """Returns (chosen_x, audit). chosen_x is None (with audit['skipped']) when
    the tuning set has too few natural falls to be worth searching.

    Starts: restart 0 = 'stand' (sigma 0.2); then, if seed_ctrls are given, the
    `n_seed_starts` best-scoring seeds on THIS bin's tuning set (sigma 0.12,
    local refinement); then random perturbations of stand until there are at
    least `restarts` starts. Every evaluated point is logged; a diverse top-k is
    re-ranked on the disjoint 'select' set; the select winner goes to the gate
    (exactly one candidate, exactly one look at the held-out set)."""
    import cma
    okw = obj_kwargs or {}
    objective, n_falls = make_objective(spec, space, unit, bin_name, run_trial, iter_tuning, iter_nofall,
                                        cfg, **okw)
    if objective is None:
        return None, {"skipped": f"only {n_falls} natural fall(s) in the tuning set (< {MIN_TUNING_FALLS})"}

    rng = np.random.default_rng(seed)
    x_stand = space.x_stand()
    lo, hi = np.zeros(space.dim), np.ones(space.dim)
    evals: List[Tuple[np.ndarray, float]] = []
    restart_of: List[int] = []

    # (x0, sigma0, label) start list
    starts: List[Tuple[np.ndarray, float, str]] = [(x_stand, 0.2, "stand")]
    seed_scores: List[float] = []
    if seed_ctrls:
        cand = [space.x_of_ctrl(c) for c in list(seed_ctrls)[:max_seed_eval]]
        scores = _eval_population(cand, objective, pool=pool)
        for x, f in zip(cand, scores):  # seeds are real evaluations: keep them in the shortlist pool
            evals.append((x, float(f)))
            restart_of.append(-1)
        order = np.argsort(scores)
        seed_scores = [float(scores[i]) for i in order]
        for i in order[:n_seed_starts]:
            starts.append((cand[i], 0.12, "seed"))
    r_extra = 1
    while len(starts) < restarts:
        sg = 0.2 + 0.1 * r_extra
        starts.append((np.clip(x_stand + rng.normal(0.0, sg, space.dim), 0.02, 0.98), sg, "random"))
        r_extra += 1

    per_restart: List[dict] = []
    for r, (x0, sigma0, label) in enumerate(starts):
        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            "bounds": [lo.tolist(), hi.tolist()], "popsize": popsize, "maxiter": maxiter,
            "seed": seed + 1 + r, "verbose": -9})
        best = float("inf")
        while not es.stop():
            xs = es.ask()
            fs = _eval_population(xs, objective, pool=pool)
            es.tell(xs, fs)
            for x, f in zip(xs, fs):
                evals.append((np.clip(np.array(x, dtype=float), lo, hi), float(f)))
                restart_of.append(r)
                best = min(best, float(f))
        stop = es.stop()
        per_restart.append({"start": label, "best_tuning_score": best,
                            "stopped_by": sorted(stop.keys()), "hit_maxiter": "maxiter" in stop})

    shortlist = _diverse_topk(evals, top_k, min_dist=0.05 * np.sqrt(space.dim))
    select_obj, _ = make_objective(spec, space, unit, bin_name, run_trial, iter_select, iter_nofall, cfg, **okw)
    if select_obj is None:  # select set had no natural falls -- fall back to tuning rank
        best_x = shortlist[0]
        scored_scores: List[float] = []
        best_select = float("nan")
    else:
        scored = sorted(((x, select_obj(x)) for x in shortlist), key=lambda t: t[1])
        best_x, best_select = scored[0]
        scored_scores = [float(s) for _, s in scored]

    ranked = sorted(range(len(evals)), key=lambda i: evals[i][1])
    win_i = next(i for i in ranked if np.allclose(evals[i][0], best_x))
    audit = {
        "n_tuning_natural_falls": n_falls,
        "n_evaluations": len(evals),
        "n_shortlist": len(shortlist),
        "selected_rank_on_tuning": ranked.index(win_i),
        "winning_restart": restart_of[win_i],   # -1 = a seed pose won without further search
        "tuning_score_of_winner": evals[win_i][1],
        "select_score": float(best_select),
        "shortlist_select_scores": scored_scores,
        "per_restart": per_restart,             # hit_maxiter=True everywhere => not converged, raise --maxiter
        "seed_scores": seed_scores[:10],        # sorted best-first; all ~equal => plateau, not a search failure
    }
    return best_x, audit


def make_objective_traj(spec: ValiditySpec, space: TrajectorySpace, unit: str, bin_name: str,
                        run_trial_traj: Callable, iter_cond: Callable, iter_nofall: Callable,
                        cfg: GateConfig, tracking_weight: float = OBJ_TRACKING_WEIGHT,
                        tracking_margin: float = OBJ_TRACKING_MARGIN, median_weight: float = OBJ_MEDIAN_WEIGHT
                        ) -> Tuple[Optional[Callable[[np.ndarray], float]], int]:
    """Trajectory counterpart of make_objective: identical cost shape (ratio +
    tracking penalty + head-not-worse penalty + fragile-contact penalty,
    blended mean/median, plus a do-no-harm term), evaluated against a
    STAND-run baseline (a single endpoint at duration_s using stand_ctrl via run_trial_traj)
    exactly like the static path uses spec.stand_ctrl. `_n_regressed` is
    stashed on the returned closure so the caller can report how often the
    harness fell back to a static call (see make_run_trial_traj)."""
    conditions = iter_cond(unit, bin_name)
    stand_traj = [(float(space.duration_s), np.asarray(spec.stand_ctrl).copy())]
    unprotected = [(c, run_trial_traj(spec, stand_traj, unit, c)[0]) for c in conditions]
    falling = [(c, u) for c, u in unprotected if u.fell]
    if len(falling) < MIN_TUNING_FALLS:
        return None, len(falling)

    nofall = []
    for c in iter_nofall(unit, bin_name, "nofall_tune", N_NOFALL_TUNE):
        u, _ = run_trial_traj(spec, stand_traj, unit, c)
        if not u.fell:
            nofall.append(c)

    n_regressed = [0]

    def objective(x: np.ndarray) -> float:
        traj = space.to_traj(x)
        nv = check_traj(spec, traj)
        if nv:
            return 10.0 + nv
        costs = []
        for cond, u in falling:
            p, regressed = run_trial_traj(spec, traj, unit, cond)
            n_regressed[0] += int(regressed)
            u_score = max(weighted_force(u.peaks), 1e-6)
            ratio = weighted_force(p.peaks) / u_score
            tracking_pen = tracking_weight * max(
                0.0, p.tracking_err_rad - tracking_margin * cfg.max_median_tracking_err_rad)
            head_over = max(0.0, p.peaks.get("head", 0.0)
                            - u.peaks.get("head", 0.0) * (1 + cfg.head_tol_frac) - cfg.head_slack_n)
            head_pen = head_over / u_score
            fragile_pen = 0.5 if p.first_contact_class == "knee_shank" else 0.0
            costs.append(ratio + tracking_pen + head_pen + fragile_pen)
        harm = 0.0
        if nofall:
            harm = float(np.mean([1.0 if run_trial_traj(spec, traj, unit, c)[0].fell else 0.0 for c in nofall]))
        agg = (1.0 - median_weight) * float(np.mean(costs)) + median_weight * float(np.median(costs))
        return agg + harm

    objective.n_regressed = n_regressed  # type: ignore[attr-defined]
    return objective, len(falling)



# =============================================================================
# CPU-parallel trajectory objective workers
# =============================================================================
# Each worker owns a separate MuJoCo model.  The parent only sends normalized
# CMA-ES vectors (x) to workers; the expensive physics trials happen in the
# worker process.  A per-bin objective context is installed once before a
# generation is evaluated, so the stand/nofall baseline is NOT recomputed for
# every candidate.
_traj_worker_objective = None
_traj_worker_runner = None


def _traj_worker_init(model_path: str, p1_module_name: str, passive_pendulum: bool,
                      trigger_lead_s: float):
    global _traj_worker_runner
    import importlib
    p1 = importlib.import_module(p1_module_name)
    from phase3_pose_jerk_v7_real import snapshot_model_state, load_instrumented_model, impact_body_ids
    model = load_instrumented_model(model_path, passive_pendulum=passive_pendulum)
    snapshot = snapshot_model_state(model)
    impact_ids = impact_body_ids(model)
    _traj_worker_runner = make_run_trial_traj(model, p1, impact_ids, snapshot,
                                               trigger_lead_s=trigger_lead_s)


def _traj_worker_set_context(ctx):
    """Install one bin's objective in this worker before candidate evaluation.

    Windows multiprocessing uses spawn, so the context must contain only
    pickleable data. Import the Phase-1 module locally from its module name.
    """
    global _traj_worker_objective
    import importlib
    p1 = importlib.import_module(ctx["p1_module_name"])
    registry = ctx["registry"]
    unit = ctx["unit"]
    bin_name = ctx["bin_name"]
    iter_tuning, _, _, iter_nofall = make_condition_iters(
        p1, registry, holdout_n=ctx["holdout_n"])
    objective, n_falls = make_objective_traj(
        ctx["spec"], ctx["space"], unit, bin_name, _traj_worker_runner,
        iter_tuning, iter_nofall, ctx["cfg"], **ctx["obj_kwargs"])
    _traj_worker_objective = objective
    return n_falls


def _traj_worker_eval(x):
    if _traj_worker_objective is None:
        raise RuntimeError("trajectory worker objective was not initialized")
    return float(_traj_worker_objective(np.asarray(x, dtype=float)))


def _eval_population(xs, objective, pool=None):
    """Evaluate one CMA-ES population serially or across CPU workers."""
    if pool is None:
        return [float(objective(np.asarray(x))) for x in xs]
    return list(pool.map(_traj_worker_eval, [np.asarray(x, dtype=float) for x in xs]))

def tune_pose_traj(spec: ValiditySpec, space: TrajectorySpace, unit: str, bin_name: str,
                   run_trial_traj: Callable, iter_tuning: Callable, iter_select: Callable, iter_nofall: Callable,
                   cfg: GateConfig, popsize: int = 12, maxiter: int = 30, restarts: int = 2, seed: int = 0,
                   top_k: int = 4, seed_point0_ctrls: Optional[Sequence[np.ndarray]] = None,
                   n_seed_starts: int = 2, obj_kwargs: Optional[dict] = None,
                   pool=None, worker_context: Optional[dict] = None
                   ) -> Tuple[Optional[np.ndarray], dict, List[dict]]:
    """Optimize a trajectory on tuning conditions, then retain the diverse Top-K.

    IMPORTANT: the 24-condition holdout gate is NOT used here to choose a single
    candidate.  The complete diverse Top-K is returned to the caller so every
    shortlisted candidate can face the same holdout acceptance gate.

    The selection conditions are used only to rank the Top-K candidates.  They
    do not eliminate candidates before gating.  If the selection set has no
    natural falls, tuning rank is used as the deterministic fallback ordering.
    """
    import cma
    okw = obj_kwargs or {}
    objective, n_falls = make_objective_traj(spec, space, unit, bin_name, run_trial_traj, iter_tuning,
                                             iter_nofall, cfg, **okw)
    if pool is not None:
        if worker_context is None:
            raise ValueError("pool was supplied without worker_context")
        # Build the identical objective inside each worker.  This also avoids
        # trying to pickle a closure that captures a MuJoCo model.
        pool.map(_traj_worker_set_context, [worker_context] * getattr(pool, "_processes", 1))
    if objective is None:
        return None, {"skipped": f"only {n_falls} natural fall(s) in the tuning set (< {MIN_TUNING_FALLS})"}, []

    rng = np.random.default_rng(seed)
    x_stand = space.x_stand()
    lo, hi = np.zeros(space.dim), np.ones(space.dim)
    evals: List[Tuple[np.ndarray, float]] = []
    restart_of: List[int] = []

    starts: List[Tuple[np.ndarray, float, str]] = [(x_stand, 0.2, "stand")]
    seed_scores: List[float] = []
    if seed_point0_ctrls:
        cand = [space.x_seed(c) for c in list(seed_point0_ctrls)[:40]]
        scores = _eval_population(cand, objective, pool=pool)
        for x, f in zip(cand, scores):
            evals.append((x, float(f))); restart_of.append(-1)
        order = np.argsort(scores)
        seed_scores = [float(scores[i]) for i in order]
        for i in order[:n_seed_starts]:
            starts.append((cand[i], 0.12, "seed"))
    r_extra = 1
    while len(starts) < restarts:
        sg = 0.2 + 0.1 * r_extra
        starts.append((np.clip(x_stand + rng.normal(0.0, sg, space.dim), 0.001, 0.999), sg, "random"))
        r_extra += 1

    per_restart: List[dict] = []
    for r, (x0, sigma0, label) in enumerate(starts):
        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            "bounds": [lo.tolist(), hi.tolist()], "popsize": popsize, "maxiter": maxiter,
            "seed": seed + 1 + r, "verbose": -9})
        best = float("inf")
        while not es.stop():
            xs = es.ask()
            fs = _eval_population(xs, objective, pool=pool)
            es.tell(xs, fs)
            for x, f in zip(xs, fs):
                evals.append((np.clip(np.array(x, dtype=float), lo, hi), float(f)))
                restart_of.append(r)
                best = min(best, float(f))
        stop = es.stop()
        per_restart.append({"start": label, "best_tuning_score": best,
                            "stopped_by": sorted(stop.keys()), "hit_maxiter": "maxiter" in stop})

    shortlist = _diverse_topk(evals, top_k, min_dist=0.05 * np.sqrt(space.dim))

    # Rank the diverse Top-K on the independent selection conditions, but DO NOT
    # collapse the shortlist to one candidate. Every retained candidate will be
    # evaluated by gate_pose_traj() on the full holdout set.
    select_obj, _ = make_objective_traj(spec, space, unit, bin_name, run_trial_traj, iter_select, iter_nofall,
                                        cfg, **okw)
    tuning_score_by_x = {id(x): float(f) for x, f in evals}
    candidate_rows: List[dict] = []
    if select_obj is None:
        # No natural falls in selection data: preserve tuning order rather than
        # inventing a select score or discarding candidates.
        ordered = list(shortlist)
        scored_scores: List[float] = []
        best_select = float("nan")
    else:
        scored = sorted(((x, select_obj(x)) for x in shortlist), key=lambda t: t[1])
        ordered = [x for x, _ in scored]
        scored_scores = [float(s) for _, s in scored]
        best_select = float(scored[0][1]) if scored else float("nan")

    # Assign deterministic shortlist ranks after selection ordering.
    for rank, x in enumerate(ordered, start=1):
        tuning_score = tuning_score_by_x.get(id(x))
        if tuning_score is None:
            # _diverse_topk returns references to eval arrays in the current
            # implementation; keep a robust fallback for future changes.
            matches = [f for ex, f in evals if np.allclose(ex, x)]
            tuning_score = float(matches[0]) if matches else float("nan")
        candidate_rows.append({
            "rank": rank,
            "x": np.asarray(x, dtype=float),
            "tuning_score": float(tuning_score),
            "select_score": (float(scored_scores[rank - 1]) if scored_scores else float("nan")),
        })

    # The first candidate remains the preferred candidate if multiple candidates
    # pass the gate; the actual acceptance decision is made later in _build_library_traj.
    best_x = candidate_rows[0]["x"] if candidate_rows else None
    ranked = sorted(range(len(evals)), key=lambda i: evals[i][1])
    win_i = next(i for i in ranked if np.allclose(evals[i][0], best_x)) if best_x is not None else 0
    audit = {
        "n_tuning_natural_falls": n_falls, "n_evaluations": len(evals), "n_shortlist": len(shortlist),
        "selected_rank_on_tuning": ranked.index(win_i) if best_x is not None else None,
        "winning_restart": restart_of[win_i] if best_x is not None else None,
        "tuning_score_of_preferred_candidate": evals[win_i][1] if best_x is not None else None,
        "select_score": float(best_select),
        "shortlist_select_scores": scored_scores,
        "per_restart": per_restart, "seed_scores": seed_scores[:10],
        "n_regressed_to_static_calls": objective.n_regressed[0],  # type: ignore[attr-defined]
        "gate_all_shortlist": True,
        "gate_candidate_count": len(candidate_rows),
    }
    return best_x, audit, candidate_rows


def gate_pose_traj(spec: ValiditySpec, space: TrajectorySpace, x: np.ndarray, unit: str, bin_name: str,
                   run_trial_traj: Callable, iter_holdout: Callable, iter_nofall: Callable,
                   cfg: GateConfig, diag: Optional[dict] = None) -> Tuple[GateReport, List[Tuple[float, np.ndarray]]]:
    """Trajectory counterpart of gate_pose. Returns (report, traj)."""
    traj = space.to_traj(x)
    stand_traj = [(float(space.duration_s), np.asarray(spec.stand_ctrl).copy())]
    fall_pairs, nofall_pairs = [], []
    n_regressed = 0
    for cond in iter_holdout(unit, bin_name):
        u, _ = run_trial_traj(spec, stand_traj, unit, cond)
        p, reg = run_trial_traj(spec, traj, unit, cond)
        n_regressed += int(reg)
        fall_pairs.append((u, p))
    for cond in iter_nofall(unit, bin_name):
        u, _ = run_trial_traj(spec, stand_traj, unit, cond)
        p, reg = run_trial_traj(spec, traj, unit, cond)
        n_regressed += int(reg)
        nofall_pairs.append((u, p))
    if diag is not None:
        genuine = [(u, p) for u, p in fall_pairs if u.fell]
        diag["n_regressed_to_static_calls"] = n_regressed
        if genuine:
            diag["n_genuine_falls"] = len(genuine)
            diag["frac_protected_still_fall"] = float(np.mean([p.fell for _, p in genuine]))
            diag["median_protected_weighted_force_n"] = float(np.median([weighted_force(p.peaks) for _, p in genuine]))
            diag["median_unprotected_weighted_force_n"] = float(np.median([weighted_force(u.peaks) for u, _ in genuine]))
    return evaluate_gate(fall_pairs, nofall_pairs, cfg), traj


# =============================================================================
# Gate + pipeline
# =============================================================================

def gate_pose(spec: ValiditySpec, space: SearchSpace, x: np.ndarray, unit: str, bin_name: str,
              run_trial: Callable, iter_holdout: Callable, iter_nofall: Callable,
              cfg: GateConfig, diag: Optional[dict] = None) -> Tuple[GateReport, np.ndarray]:
    """Returns (report, ctrl) exactly as before. If `diag` is a dict it is filled
    with sanity numbers computed from the SAME trials (no extra simulation)."""
    ctrl = space.to_ctrl(x)
    fall_pairs, nofall_pairs = [], []
    for cond in iter_holdout(unit, bin_name):
        fall_pairs.append((run_trial(spec, spec.stand_ctrl, unit, cond), run_trial(spec, ctrl, unit, cond)))
    for cond in iter_nofall(unit, bin_name):  # default tag 'gate'
        nofall_pairs.append((run_trial(spec, spec.stand_ctrl, unit, cond), run_trial(spec, ctrl, unit, cond)))
    if diag is not None:
        genuine = [(u, p) for u, p in fall_pairs if u.fell]
        if genuine:
            diag["n_genuine_falls"] = len(genuine)
            diag["frac_protected_still_fall"] = float(np.mean([p.fell for _, p in genuine]))
            diag["median_protected_weighted_force_n"] = float(np.median([weighted_force(p.peaks) for _, p in genuine]))
            diag["median_unprotected_weighted_force_n"] = float(np.median([weighted_force(u.peaks) for u, _ in genuine]))
    return evaluate_gate(fall_pairs, nofall_pairs, cfg), ctrl


def _json_default(o):
    """numpy scalars/arrays from gate checks or audits -> plain JSON types."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def excluded_placeholder_entries(excluded_units: Dict[str, List[int]], space: SearchSpace,
                                 reason: str, bins: Optional[Sequence[str]] = None) -> Dict[str, dict]:
    """One 'stand' entry per (excluded unit, bin) so the output JSON is
    self-documenting -- a missing key used to be ambiguous between 'excluded
    on purpose' and 'some other error'. `excluded: True` distinguishes this
    from a genuine 0-natural-fall skip, which also lands at 'stand' but was
    actually evaluated."""
    run_bins = tuple(bins) if bins else VELOCITY_BINS
    out: Dict[str, dict] = {}
    for unit, sids in excluded_units.items():
        for bin_name in run_bins:
            out[f"{unit}/{bin_name}"] = {
                "unit": unit, "scenario_ids": sids, "velocity_bin": bin_name,
                "kind": "stand", "excluded": True, "gate_passed": False, "skipped_reason": reason,
                "pose_ctrl": space.stand.tolist(), "gate_checks": {}, "static_violations": [],
                "candidate_selection": {"skipped": reason},
            }
    return out


def _save(path: str, library: dict, space: SearchSpace) -> None:
    payload = {"_meta": {"search_box_source": space.source, "joints": space.names,
                         "lo": space.lo.tolist(), "hi": space.hi.tolist(),
                         "stand_ctrl": space.stand.tolist()}, **library}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def build_library(spec: ValiditySpec, space: SearchSpace, units: Dict[str, List[int]], run_trial: Callable,
                  iter_tuning: Callable, iter_select: Callable, iter_holdout: Callable, iter_nofall: Callable,
                  popsize: int, maxiter: int, restarts: int, cfg: Optional[GateConfig] = None,
                  out_path: Optional[str] = None, existing: Optional[dict] = None,
                  seed_ctrls: Optional[Sequence[np.ndarray]] = None,
                  obj_kwargs: Optional[dict] = None,
                  bins: Optional[Sequence[str]] = None,
                  traj_space: Optional[TrajectorySpace] = None,
                  run_trial_traj: Optional[Callable] = None) -> dict:
    """`bins` restricts the run to a subset of VELOCITY_BINS (default: all).
    A parallel launcher uses it to run one (unit, bin) job per process.

    `traj_space` + `run_trial_traj`: when both are given, every bin is
    searched as an N-via-point trajectory instead of one static pose (see
    TrajectorySpace / tune_pose_traj / gate_pose_traj). `seed_ctrls`, if given,
    warm-start via-point 0 of the trajectory search (see TrajectorySpace.x_seed)
    -- they need not come from a trajectory run; a prior STATIC library is the
    intended seed source, since N=1 of a trajectory search recovers the static
    objective exactly."""
    if traj_space is not None:
        return _build_library_traj(spec, traj_space, units, run_trial_traj, iter_tuning, iter_select,
                                   iter_holdout, iter_nofall, popsize, maxiter, restarts, cfg=cfg,
                                   out_path=out_path, existing=existing, seed_point0_ctrls=seed_ctrls,
                                   obj_kwargs=obj_kwargs, bins=bins)

    run_bins = tuple(bins) if bins else VELOCITY_BINS
    bad = [b for b in run_bins if b not in VELOCITY_BINS]
    if bad:
        raise ValueError(f"unknown velocity bin(s) {bad}; expected a subset of {VELOCITY_BINS}")
    library: Dict[str, dict] = dict(existing or {})
    for unit, sids in units.items():
        for bin_name in run_bins:
            key = f"{unit}/{bin_name}"
            if key in library:
                print(f"[{key:34s}] resumed (already in output)")
                continue
            bin_cfg = cfg if cfg is not None else GATE_CONFIG_BY_BIN[bin_name]
            t0 = time.time()
            x, audit = tune_pose(spec, space, unit, bin_name, run_trial, iter_tuning, iter_select, iter_nofall,
                                 bin_cfg, popsize=popsize, maxiter=maxiter, restarts=restarts,
                                 seed_ctrls=seed_ctrls, obj_kwargs=obj_kwargs)
            entry = {"unit": unit, "scenario_ids": sids, "velocity_bin": bin_name,
                     "candidate_selection": audit}
            if x is None:
                entry.update(kind="stand", gate_passed=False, skipped_reason=audit["skipped"],
                             pose_ctrl=spec.stand_ctrl.tolist(), gate_checks={}, static_violations=[])
                print(f"[{key:34s}] stand  SKIPPED: {audit['skipped']} ({time.time()-t0:.1f}s)")
            else:
                diag: dict = {}
                report, ctrl = gate_pose(spec, space, x, unit, bin_name, run_trial, iter_holdout, iter_nofall,
                                         bin_cfg, diag=diag)
                static_v = check_pose_ctrl(spec, ctrl)
                kind, final_ctrl = decide(ctrl, static_v, report, spec.stand_ctrl)
                entry.update(
                    kind=kind, gate_passed=report.passed, pose_ctrl=final_ctrl.tolist(),
                    searched_delta_from_stand_rad={n: float(ctrl[i] - space.stand[i])
                                                   for n, i in zip(space.names, space.idx)},
                    gate_checks={k: {"ok": ok, "msg": msg} for k, (ok, msg) in report.checks.items()},
                    static_violations=static_v, gate_diagnostics=diag)
                unconverged = all(r["hit_maxiter"] for r in audit["per_restart"])
                print(f"[{key:34s}] {kind:6s} gate={'PASS' if report.passed else 'FAIL'} "
                      f"evals={audit['n_evaluations']} restart={audit['winning_restart']}"
                      f"{' UNCONVERGED' if unconverged else ''} ({time.time()-t0:.1f}s)")
                _n_genuine = diag.get("n_genuine_falls", 0)
                if diag.get("frac_protected_still_fall", 1.0) < 0.1 and _n_genuine:
                    print(f"    WARNING {key}: only {diag['frac_protected_still_fall']:.0%} of protected falls still "
                          f"fall (median protected force {diag['median_protected_weighted_force_n']:.0f} N vs "
                          f"{diag['median_unprotected_weighted_force_n']:.0f} N) -- a ~100% 'reduction' means the "
                          "fall is being prevented or the contact tracker isn't registering; verify in the viewer.")
                if (_n_genuine and diag.get("frac_protected_still_fall", 0) >= 0.5
                        and diag.get("median_protected_weighted_force_n", 1.0) < 1.0):
                    print(f"    WARNING {key}: median protected force is ~0 N "
                          f"({diag['median_protected_weighted_force_n']:.2f} N) while the robot STILL falls in "
                          f"{diag['frac_protected_still_fall']:.0%} of trials -- physically implausible for a real "
                          "impact. Likely cause: the contact/force measurement isn't capturing the actual landing "
                          "for this scenario (observation window ends before ground contact, or geom IDs don't "
                          "match after something in the scenario moves, e.g. floor_drop). Do not trust this "
                          "reduction number until verified in the viewer.")
            entry["tuning_seconds"] = round(time.time() - t0, 1)
            library[key] = entry
            if out_path:
                _save(out_path, library, space)  # partial results survive a crash / Ctrl-C
    return library


def _traj_json(traj: List[Tuple[float, np.ndarray]]) -> List[list]:
    return [[float(t), np.asarray(ctrl).tolist()] for t, ctrl in traj]


def _save_traj(path: str, library: dict, space: TrajectorySpace) -> None:
    payload = {"_meta": {"search_box_source": space.base.source, "joints": space.base.names,
                         "lo": space.base.lo.tolist(), "hi": space.base.hi.tolist(),
                         "stand_ctrl": space.base.stand.tolist(),
                         "n_points": space.n_points, "duration_s": space.duration_s}, **library}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _build_library_traj(spec: ValiditySpec, space: TrajectorySpace, units: Dict[str, List[int]],
                        run_trial_traj: Callable, iter_tuning: Callable, iter_select: Callable,
                        iter_holdout: Callable, iter_nofall: Callable, popsize: int, maxiter: int, restarts: int,
                        cfg: Optional[GateConfig] = None, out_path: Optional[str] = None,
                        existing: Optional[dict] = None, seed_point0_ctrls: Optional[Sequence[np.ndarray]] = None,
                        obj_kwargs: Optional[dict] = None, bins: Optional[Sequence[str]] = None,
                        pool=None, worker_context_factory: Optional[Callable] = None) -> dict:
    """Trajectory counterpart of build_library's main loop -- same structure,
    entries are 'pose_traj' (list of [t, ctrl]) instead of a single 'pose_ctrl'
    (which is still populated, = the LAST via-point, for any consumer that
    only reads the old field)."""
    run_bins = tuple(bins) if bins else VELOCITY_BINS
    bad = [b for b in run_bins if b not in VELOCITY_BINS]
    if bad:
        raise ValueError(f"unknown velocity bin(s) {bad}; expected a subset of {VELOCITY_BINS}")
    library: Dict[str, dict] = dict(existing or {})
    for unit, sids in units.items():
        for bin_name in run_bins:
            key = f"{unit}/{bin_name}"
            if key in library:
                print(f"[{key:34s}] resumed (already in output)")
                continue
            bin_cfg = cfg if cfg is not None else GATE_CONFIG_BY_BIN[bin_name]
            t0 = time.time()
            x, audit, gate_candidates = tune_pose_traj(
                spec, space, unit, bin_name, run_trial_traj, iter_tuning, iter_select,
                iter_nofall, bin_cfg, popsize=popsize, maxiter=maxiter, restarts=restarts,
                seed_point0_ctrls=seed_point0_ctrls, obj_kwargs=obj_kwargs,
                pool=pool,
                worker_context=(worker_context_factory(unit, bin_name)
                                if worker_context_factory is not None else None))
            entry = {"unit": unit, "scenario_ids": sids, "velocity_bin": bin_name, "mode": "trajectory",
                     "n_points": space.n_points, "duration_s": space.duration_s, "candidate_selection": audit}
            if x is None:
                stand_traj = [(float(space.duration_s), np.asarray(spec.stand_ctrl).copy())]
                entry.update(kind="stand", gate_passed=False, skipped_reason=audit["skipped"],
                            pose_traj=_traj_json(stand_traj), pose_ctrl=spec.stand_ctrl.tolist(),
                            gate_checks={}, static_violations=[])
                print(f"[{key:34s}] stand  SKIPPED: {audit['skipped']} ({time.time()-t0:.1f}s)")
            else:
                # IMPORTANT: gate EVERY diverse Top-K candidate.  Selection only
                # ranks the candidates; it must not discard a candidate before the
                # actual 24-condition acceptance gate.  This prevents a candidate
                # that ranks second/third/fourth on selection data from being lost
                # when the preferred candidate fails median reduction, tracking,
                # regression, or another gate criterion.
                candidate_gate_results: List[dict] = []
                passing_candidates: List[Tuple[dict, object, List[Tuple[float, np.ndarray]], dict, List[str]]] = []
                for cand in gate_candidates:
                    cand_diag: dict = {}
                    cand_report, cand_traj = gate_pose_traj(
                        spec, space, cand["x"], unit, bin_name, run_trial_traj,
                        iter_holdout, iter_nofall, bin_cfg, diag=cand_diag)
                    cand_nv = check_traj(spec, cand_traj)
                    gate_passed = bool(cand_report.passed and not cand_nv)
                    gate_row = {
                        "rank": int(cand["rank"]),
                        "tuning_score": float(cand["tuning_score"]),
                        "select_score": float(cand["select_score"]),
                        "gate_passed": gate_passed,
                        "gate_checks": {k: {"ok": ok, "msg": msg}
                                         for k, (ok, msg) in cand_report.checks.items()},
                        "static_violations": cand_nv,
                        "gate_diagnostics": cand_diag,
                        "pose_traj": _traj_json(cand_traj),
                    }
                    candidate_gate_results.append(gate_row)
                    if gate_passed:
                        passing_candidates.append((cand, cand_report, cand_traj, cand_diag, cand_nv))

                # Deterministic final choice: candidates are already ordered by
                # selection score. Gate results decide eligibility; selection score
                # breaks ties among eligible candidates. We do NOT optimize on the
                # holdout values.
                if passing_candidates:
                    passing_candidates.sort(key=lambda z: (z[0]["select_score"], z[0]["rank"]))
                    chosen, report, traj, diag, nv = passing_candidates[0]
                    kind, final_traj = "pose_traj", traj
                    final_gate_passed = True
                    final_gate_checks = {k: {"ok": ok, "msg": msg}
                                         for k, (ok, msg) in report.checks.items()}
                    final_diag = diag
                    final_nv = nv
                    selected_rank = int(chosen["rank"])
                else:
                    # No shortlisted trajectory satisfied the acceptance gate.
                    # Fall back to stand rather than deploying a failing pose.
                    kind = "stand"
                    final_traj = [(float(space.duration_s), np.asarray(spec.stand_ctrl).copy())]
                    final_gate_passed = False
                    final_gate_checks = {}
                    final_diag = {"n_candidates_gated": len(gate_candidates),
                                  "n_candidates_passed": 0}
                    final_nv = []
                    selected_rank = None

                entry.update(
                    kind=kind, gate_passed=final_gate_passed, pose_traj=_traj_json(final_traj),
                    pose_ctrl=np.asarray(final_traj[-1][1]).tolist(),
                    switch_times_s=[float(t) for t, _ in final_traj],
                    gate_checks=final_gate_checks, static_violations=final_nv,
                    gate_diagnostics=final_diag,
                    candidate_gate_results=candidate_gate_results,
                    selected_gate_candidate_rank=selected_rank)
                unconverged = all(r["hit_maxiter"] for r in audit["per_restart"])
                n_reg = audit.get("n_regressed_to_static_calls", 0) + sum(
                    r.get("gate_diagnostics", {}).get("n_regressed_to_static_calls", 0)
                    for r in candidate_gate_results)
                n_pass = sum(1 for r in candidate_gate_results if r["gate_passed"])
                print(f"[{key:34s}] {kind:10s} gate={'PASS' if final_gate_passed else 'FAIL'} "
                      f"candidates={len(gate_candidates)} passed={n_pass} evals={audit['n_evaluations']}"
                      f"{' UNCONVERGED' if unconverged else ''}"
                      f"{f' REGRESSED-TO-STATIC x{n_reg}' if n_reg else ''} ({time.time()-t0:.1f}s)")
                for r in candidate_gate_results:
                    print(f"    candidate rank={r['rank']} gate={'PASS' if r['gate_passed'] else 'FAIL'} "
                          f"select={r['select_score']:.4f} tuning={r['tuning_score']:.4f}")
                if n_reg:
                    print(f"    NOTE {key}: the harness fell back to a static pose_ctrl call {n_reg} time(s) -- "
                          "this bin's result is NOT a real trajectory evaluation until run_protected_trial "
                          "accepts pose_traj=.")
            entry["tuning_seconds"] = round(time.time() - t0, 1)
            library[key] = entry
            if out_path:
                _save_traj(out_path, library, space)
    return library


def regate_library(spec: ValiditySpec, space: SearchSpace, existing: dict, run_trial: Callable,
                   iter_holdout: Callable, iter_nofall: Callable,
                   cfg_by_bin: Dict[str, GateConfig], out_path: Optional[str] = None) -> dict:
    """Re-runs ONLY gate_pose on every entry's already-found pose against a
    (usually larger) holdout set -- no CMA-ES, so it's cheap. Skipped and
    excluded entries are carried over unchanged (there is no pose to re-gate).
    Overwrites each entry's gate_checks/gate_passed/kind/pose_ctrl/gate_diagnostics
    in place; candidate_selection (the original tuning record) is left as-is
    plus a note that it was re-gated."""
    library: Dict[str, dict] = dict(existing)
    for key, entry in existing.items():
        if entry.get("excluded") or entry.get("skipped_reason") or not entry.get("searched_delta_from_stand_rad"):
            continue  # nothing was searched for this bin -- carry over unchanged
        unit, bin_name = entry["unit"], entry["velocity_bin"]
        dl = entry["searched_delta_from_stand_rad"]
        if any(n not in dl for n in space.names):
            print(f"[{key:34s}] SKIPPED regate: searched_delta_from_stand_rad doesn't match this space's joints")
            continue
        ctrl = space.stand.copy()
        ctrl[space.idx] = np.clip(space.stand[space.idx] + np.array([dl[n] for n in space.names]),
                                  space.lo, space.hi)
        x = space.x_of_ctrl(ctrl)
        cfg = cfg_by_bin[bin_name]
        diag: dict = {}
        t0 = time.time()
        report, ctrl = gate_pose(spec, space, x, unit, bin_name, run_trial, iter_holdout, iter_nofall, cfg, diag=diag)
        static_v = check_pose_ctrl(spec, ctrl)
        kind, final_ctrl = decide(ctrl, static_v, report, spec.stand_ctrl)
        new_entry = dict(entry)
        new_entry.update(kind=kind, gate_passed=report.passed, pose_ctrl=final_ctrl.tolist(),
                         gate_checks={k: {"ok": ok, "msg": msg} for k, (ok, msg) in report.checks.items()},
                         static_violations=static_v, gate_diagnostics=diag, regated=True,
                         regate_seconds=round(time.time() - t0, 1))
        flip = "" if kind == entry.get("kind") else f"  ({entry.get('kind')} -> {kind})"
        print(f"[{key:34s}] regated {kind:6s} gate={'PASS' if report.passed else 'FAIL'}{flip} "
              f"({time.time()-t0:.1f}s)")
        library[key] = new_entry
        if out_path:
            _save(out_path, library, space)
    return library


# =============================================================================
# CLI
# =============================================================================

def main(argv=None) -> int:
    import mujoco
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="pose_library.json")
    ap.add_argument("--granularity", choices=("scenario", "group"), default="scenario")
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=30)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1,
                    help="CPU worker processes for trajectory candidate evaluation. "
                         "1 = serial; >1 evaluates each CMA-ES generation in parallel. "
                         "Each worker owns its own MuJoCo model.")
    ap.add_argument("--only", default="", help="comma-separated unit keys (or key prefixes) to run")
    ap.add_argument("--box-directions", type=int, default=250,
                    help="random combined-joint directions used to widen the oracle search box beyond the "
                         "axis-only pass (0 reproduces the old v2.1 box exactly); ignored if ValiditySpec "
                         "exposes explicit per-actuator bounds")
    ap.add_argument("--box-seed", type=int, default=0, help="seed for --box-directions sampling")
    ap.add_argument("--only-bin", choices=VELOCITY_BINS, default="",
                    help="run just one velocity bin (used by the parallel launcher, one job per unit/bin)")
    ap.add_argument("--exclude", default="", help="comma-separated unit keys/prefixes to drop (e.g. s11,s13)")
    ap.add_argument("--holdout-n", type=int, default=24,
                    help="held-out draws for the gate (the first 24 are identical to the default set)")
    ap.add_argument("--seed-from", default="", help="previous library JSON; its poses seed extra CMA-ES starts")
    ap.add_argument("--regate-from", default="",
                    help="previous library JSON; re-run ONLY the gate (no search) on every entry's pose against "
                         "--regate-holdout-n held-out draws. Cheap way to resolve bins that failed only on "
                         "'evidence'. Ignores --seed-from, --box-*, --popsize/--maxiter/--restarts.")
    ap.add_argument("--regate-holdout-n", type=int, default=48, help="held-out draws to use for --regate-from")
    ap.add_argument("--passive-pendulum", action="store_true",
                    help="match phase3_pose_jerk_v7's passive-pendulum model option in CPU workers")
    ap.add_argument("--p1-module", default="generate_fall_dataset_final",
                    help="Phase-1 scenario module imported inside CPU workers")
    ap.add_argument("--traj-points", type=int, default=1,
                    help="search N via-point poses (+ N-1 switch-time fractions) instead of 1 static pose. "
                         "1 (default) is the old static-pose behaviour, unchanged. REQUIRES run_protected_trial "
                         "to accept a pose_traj= kwarg -- if it doesn't, results silently regress to a static "
                         "call using the last via-point (flagged per-bin as REGRESSED-TO-STATIC in the output).")
    ap.add_argument("--traj-duration-s", type=float, default=0.30,
                    help="trigger-to-impact recovery window used by --traj-points (default: 0.30 s); "
                         "the final via-point is reached at this time")
    ap.add_argument("--tracking-weight", type=float, default=OBJ_TRACKING_WEIGHT)
    ap.add_argument("--tracking-margin", type=float, default=OBJ_TRACKING_MARGIN)
    ap.add_argument("--median-weight", type=float, default=OBJ_MEDIAN_WEIGHT)
    ap.add_argument("--resume", action="store_true", help="keep finished bins already present in --out")
    ap.add_argument("--mock", action="store_true", help="fake physics: exercises CMA-ES + gate wiring only")
    ap.add_argument("--live", action="store_true", help="real harness via make_run_trial")
    args = ap.parse_args(argv)
    if args.traj_points > 1 and not (0.05 <= args.traj_duration_s <= 0.30):
        ap.error("--traj-duration-s must be between 0.05 and 0.30 s for the current 300 ms trigger budget")

    model = mujoco.MjModel.from_xml_path(args.model)
    spec = build_spec(model)

    if args.mock:
        units = make_units(args.granularity)
        run_trial, it, isel, ih, inf = (run_one_trial_MOCK, iter_tuning_conditions_MOCK,
                                        iter_select_conditions_MOCK, iter_holdout_conditions_MOCK,
                                        iter_nofall_conditions_MOCK)
        print("*** MOCK MODE: physics is fake, this only exercises CMA-ES + gate wiring ***\n")
    elif args.live:
        import generate_fall_dataset_final as p1
        from phase3_pose_jerk_v7_real import snapshot_model_state, load_instrumented_model, impact_body_ids
        id_to_tuple = {t[0]: t for t in p1.SCENARIOS}
        units = make_units(args.granularity, p1)
        registry = {u: [id_to_tuple[s] for s in ids] for u, ids in units.items()}
        model = load_instrumented_model(args.model)
        impact_ids = impact_body_ids(model)
        snapshot = snapshot_model_state(model)
        run_trial = make_run_trial(model, p1, impact_ids, snapshot)
        it, isel, ih, inf = make_condition_iters(p1, registry, holdout_n=args.holdout_n)
        print("*** LIVE MODE *** -- check SCENARIO_TO_GROUP matches your intent.\n")
    else:
        print("Pass --mock to test the pipeline, or --live to run against the real harness.")
        return 1

    if args.regate_from:
        with open(args.regate_from) as f:
            existing = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        space = build_search_space(spec, model, box_directions=args.box_directions, box_seed=args.box_seed)
        print(space.describe(), "\n")
        if args.live:
            it, isel, ih, inf = make_condition_iters(p1, registry, holdout_n=args.regate_holdout_n)
        else:
            ih = lambda u, b: iter_holdout_conditions_MOCK(u, b)  # mock has a fixed-size holdout; fine for wiring
        n_re = sum(1 for v in existing.values()
                  if not v.get("excluded") and not v.get("skipped_reason") and v.get("searched_delta_from_stand_rad"))
        print(f"re-gating {n_re} entrie(s) from {args.regate_from} with holdout_n={args.regate_holdout_n} "
              f"(no CMA-ES search)\n")
        library = regate_library(spec, space, existing, run_trial, ih, inf, GATE_CONFIG_BY_BIN, out_path=args.out)
        n_pose = sum(1 for v in library.values() if v["kind"] == "pose")
        n_flip = sum(1 for k, v in library.items() if v.get("regated") and v.get("kind") != existing[k].get("kind"))
        print(f"\n{n_pose}/{len(library)} bins now produce a gate-passed pose ({n_flip} status change(s) from "
              f"regating). wrote {args.out}")
        return 0

    if args.only:
        wanted = [w.strip() for w in args.only.split(",") if w.strip()]
        units = {u: s for u, s in units.items() if any(u == w or u.startswith(w) for w in wanted)}
        if not units:
            print(f"--only {args.only!r} matched no units. Exiting.")
            return 1

    excluded_units: Dict[str, List[int]] = {}
    if args.exclude:
        drop = [w.strip() for w in args.exclude.split(",") if w.strip()]
        excluded_units = {u: s_ for u, s_ in units.items() if any(u == w or u.startswith(w) for w in drop)}
        units = {u: s_ for u, s_ in units.items() if u not in excluded_units}
        if not units:
            print(f"--exclude {args.exclude!r} removed every unit. Exiting.")
            return 1
        print(f"excluded {len(excluded_units)} unit(s) via --exclude {args.exclude!r}: "
              f"{', '.join(sorted(excluded_units))} (placeholder 'stand' entries will be written for these)\n")

    base_space = build_search_space(spec, model, box_directions=args.box_directions, box_seed=args.box_seed)
    print(base_space.describe(), "\n")
    obj_kwargs = {"tracking_weight": args.tracking_weight, "tracking_margin": args.tracking_margin,
                  "median_weight": args.median_weight}

    pool = None
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    if args.workers > 1 and args.traj_points <= 1:
        print("[info] --workers is only applied to trajectory CMA-ES; static mode remains unchanged.")
    if args.workers > 1 and args.traj_points > 1:
        if args.mock:
            print("[info] --mock uses lightweight fake physics; keeping trajectory evaluation serial.")
        else:
            model_abspath = os.path.abspath(args.model)
            pool = mp.Pool(
                processes=args.workers,
                initializer=_traj_worker_init,
                initargs=(model_abspath, args.p1_module, args.passive_pendulum
                          if hasattr(args, "passive_pendulum") else False,
                          0.3),
            )
            print(f"Started a pool of {args.workers} CPU worker processes for trajectory CMA-ES.")

    existing = {}
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            existing = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    if args.traj_points > 1:
        traj_space = TrajectorySpace(base=base_space, n_points=args.traj_points, duration_s=args.traj_duration_s)
        seed_ctrls = load_seed_ctrls(args.seed_from, base_space) if args.seed_from else None
        if args.seed_from:
            print(f"loaded {len(seed_ctrls)} distinct seed poses from {args.seed_from} "
                  f"(each warm-starts a trajectory seed)\n")
        if args.mock:
            run_trial_traj = run_traj_trial_MOCK
        else:
            run_trial_traj = make_run_trial_traj(model, p1, impact_ids, snapshot)
        print(f"TRAJECTORY MODE: {args.traj_points} via-points over {args.traj_duration_s}s "
              f"(dim={traj_space.dim} per bin, vs {base_space.dim} for a static pose)\n")

        def _worker_context_factory(unit, bin_name):
            # IMPORTANT: do not put the imported p1 module itself in this
            # dictionary. Windows multiprocessing pickles this context.
            return {
                "spec": spec, "space": traj_space, "unit": unit, "bin_name": bin_name,
                "cfg": GATE_CONFIG_BY_BIN[bin_name], "obj_kwargs": obj_kwargs,
                "p1_module_name": args.p1_module, "registry": registry,
                "holdout_n": args.holdout_n,
            }

        try:
            library = _build_library_traj(
                spec, traj_space, units, run_trial_traj, it, isel, ih, inf,
                args.popsize, args.maxiter, args.restarts, cfg=None, out_path=args.out,
                existing=existing, seed_point0_ctrls=seed_ctrls, obj_kwargs=obj_kwargs,
                bins=([args.only_bin] if args.only_bin else None), pool=pool,
                worker_context_factory=(_worker_context_factory if pool is not None else None))
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        if excluded_units:
            reason = f"excluded via --exclude {args.exclude!r} (this scenario produced no/negligible natural falls)"
            ph = excluded_placeholder_entries(excluded_units, base_space, reason,
                                              bins=([args.only_bin] if args.only_bin else None))
            for k, v in ph.items():
                v["pose_traj"] = _traj_json([(float(traj_space.duration_s), base_space.stand)])
                v["mode"] = "trajectory"
            library.update(ph)
            _save_traj(args.out, library, traj_space)
        n_pose = sum(1 for v in library.values() if v["kind"] == "pose_traj")
        n_reg = sum(v.get("gate_diagnostics", {}).get("n_regressed_to_static_calls", 0) for v in library.values())
        print(f"\n{n_pose}/{len(library)} bins produced a gate-passed trajectory; "
              f"{len(library)-n_pose} fell back to 'stand'."
              + (f" WARNING: {n_reg} total regressed-to-static calls across the run -- these bins' results are "
                 "NOT real trajectory evaluations (see the v2.4 docstring note)." if n_reg else ""))
        print(f"wrote {args.out}")
        return 0

    space = base_space
    seed_ctrls = load_seed_ctrls(args.seed_from, space) if args.seed_from else None
    if args.seed_from:
        print(f"loaded {len(seed_ctrls)} distinct seed poses from {args.seed_from}\n")
    n_cond = 12 + N_NOFALL_TUNE  # + up to 40 seed evaluations per bin when --seed-from is used
    print(f"units={len(units)} x bins={1 if args.only_bin else len(VELOCITY_BINS)}; per bin ~ "
          f"{args.restarts * args.popsize * args.maxiter} objective evals x up to {n_cond} trials each "
          f"(time ONE trial, multiply, before launching the full run)\n")

    library = build_library(spec, space, units, run_trial, it, isel, ih, inf, args.popsize, args.maxiter,
                            args.restarts, cfg=None, out_path=args.out, existing=existing,
                            seed_ctrls=seed_ctrls, obj_kwargs=obj_kwargs,
                            bins=([args.only_bin] if args.only_bin else None))

    if excluded_units:
        reason = f"excluded via --exclude {args.exclude!r} (this scenario produced no/negligible natural falls)"
        library.update(excluded_placeholder_entries(excluded_units, space, reason,
                                                     bins=([args.only_bin] if args.only_bin else None)))
        _save(args.out, library, space)

    n_pose = sum(1 for v in library.values() if v["kind"] == "pose")
    n_excl = sum(1 for v in library.values() if v.get("excluded"))
    n_skip = sum(1 for v in library.values() if v.get("skipped_reason") and not v.get("excluded"))
    print(f"\n{n_pose}/{len(library)} bins produced a gate-passed pose; "
          f"{len(library)-n_pose} fell back to 'stand' ({n_skip} skipped for lack of natural falls, "
          f"{n_excl} excluded via --exclude).")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
