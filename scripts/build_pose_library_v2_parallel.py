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
              --granularity group     -> the 7 SCENARIO_TO_GROUP groups
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
import sys
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from validity_spec import (
    GateConfig, GateReport, TrialResult, ValiditySpec, build_spec,
    check_pose_ctrl, decide, evaluate_gate, weighted_force,
)

VELOCITY_BINS = ("low", "mid", "high")  # match these to your existing bin edges

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
    from validity_spec import ContactTracker
    _warned = {"once": False}

    def run_trial(spec: ValiditySpec, ctrl: np.ndarray, scenario: str, condition: dict) -> TrialResult:
        from phase3_pose_jerk_v7 import run_protected_trial  # local import: only needed in live mode

        scen_tuple = condition.get("scenario_tuple")
        if scen_tuple is None:
            raise KeyError("condition dict is missing 'scenario_tuple' -- use make_condition_iters")

        ct = ContactTracker(model)
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
    10: "unknown",      # low_friction
    11: "unknown",      # actuator_fault
    12: "unknown",      # actuator_stuck
    13: "unknown",      # asymmetric_gain
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
    Also falls back to the 'unknown' group entry if the unit is absent."""
    entry = library.get(f"{unit_key}/{bin_name}") or library.get(f"unknown/{bin_name}")
    if entry is None or entry["kind"] != "pose":
        return None
    return np.array(entry["pose_ctrl"])


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
        scores = [objective(x) for x in cand]
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
            fs = [objective(np.array(x)) for x in xs]
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
                  bins: Optional[Sequence[str]] = None) -> dict:
    """`bins` restricts the run to a subset of VELOCITY_BINS (default: all).
    A parallel launcher uses it to run one (unit, bin) job per process."""
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
    ap.add_argument("--tracking-weight", type=float, default=OBJ_TRACKING_WEIGHT)
    ap.add_argument("--tracking-margin", type=float, default=OBJ_TRACKING_MARGIN)
    ap.add_argument("--median-weight", type=float, default=OBJ_MEDIAN_WEIGHT)
    ap.add_argument("--workers", type=int, default=1,
                    help="CPU parallel workers; independent unit/bin jobs run in separate processes")
    ap.add_argument("--resume", action="store_true", help="keep finished bins already present in --out")
    ap.add_argument("--mock", action="store_true", help="fake physics: exercises CMA-ES + gate wiring only")
    ap.add_argument("--live", action="store_true", help="real harness via make_run_trial")
    args = ap.parse_args(argv)

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
        from phase3_pose_jerk_v7 import snapshot_model_state, load_instrumented_model, impact_body_ids
        id_to_tuple = {t[0]: t for t in p1.SCENARIOS}
        units = make_units(args.granularity, p1)
        registry = {u: [id_to_tuple[s] for s in ids] for u, ids in units.items()}
        model = load_instrumented_model(args.model)
        impact_ids = impact_body_ids(model)
        snapshot = snapshot_model_state(model)
        run_trial = make_run_trial(model, p1, impact_ids, snapshot)
        it, isel, ih, inf = make_condition_iters(p1, registry, holdout_n=args.holdout_n)
        print("*** LIVE MODE *** -- check SCENARIO_TO_GROUP (esp. 'unknown') matches your intent.\n")
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

    space = build_search_space(spec, model, box_directions=args.box_directions, box_seed=args.box_seed)
    print(space.describe(), "\n")
    seed_ctrls = load_seed_ctrls(args.seed_from, space) if args.seed_from else None
    if args.seed_from:
        print(f"loaded {len(seed_ctrls)} distinct seed poses from {args.seed_from}\n")
    obj_kwargs = {"tracking_weight": args.tracking_weight, "tracking_margin": args.tracking_margin,
                  "median_weight": args.median_weight}
    n_cond = 12 + N_NOFALL_TUNE  # + up to 40 seed evaluations per bin when --seed-from is used
    print(f"units={len(units)} x bins={1 if args.only_bin else len(VELOCITY_BINS)}; per bin ~ "
          f"{args.restarts * args.popsize * args.maxiter} objective evals x up to {n_cond} trials each "
          f"(time ONE trial, multiply, before launching the full run)\n")

    existing = {}
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            existing = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    if args.workers < 1:
        ap.error("--workers must be >= 1")

    if args.workers > 1 and not args.only_bin:
        # Isolated subprocesses each own their MuJoCo model/data and output file.
        jobs = [(unit, b) for unit in units for b in VELOCITY_BINS
                if f"{unit}/{b}" not in existing]
        if not jobs:
            library = dict(existing)
        else:
            print(f"CPU parallelism: {args.workers} worker processes for {len(jobs)} unit/bin jobs")
            with tempfile.TemporaryDirectory(prefix="pose_lib_workers_") as tmpdir:
                def run_job(job_index, unit, bin_name):
                    job_out = os.path.join(tmpdir, f"job_{job_index:04d}.json")
                    cmd = [sys.executable, os.path.abspath(__file__),
                           "--model", args.model, "--out", job_out,
                           "--granularity", args.granularity,
                           "--popsize", str(args.popsize), "--maxiter", str(args.maxiter),
                           "--restarts", str(args.restarts), "--only", unit,
                           "--only-bin", bin_name, "--box-directions", str(args.box_directions),
                           "--box-seed", str(args.box_seed), "--holdout-n", str(args.holdout_n),
                           "--tracking-weight", str(args.tracking_weight),
                           "--tracking-margin", str(args.tracking_margin),
                           "--median-weight", str(args.median_weight)]
                    cmd.append("--mock" if args.mock else "--live")
                    if args.seed_from:
                        cmd.extend(["--seed-from", args.seed_from])
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode:
                        raise RuntimeError(f"Parallel job {unit}/{bin_name} failed (exit {result.returncode}):\\n"
                                           f"{result.stdout}\\n{result.stderr}")
                    with open(job_out, encoding="utf-8") as jf:
                        payload = json.load(jf)
                    return {k: v for k, v in payload.items() if not k.startswith("_")}

                results = {}
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = [pool.submit(run_job, i, u, b) for i, (u, b) in enumerate(jobs)]
                    for future in as_completed(futures):
                        results.update(future.result())
                library = {**existing, **results}
    else:
        library = build_library(spec, space, units, run_trial, it, isel, ih, inf, args.popsize, args.maxiter,
                                args.restarts, cfg=None, out_path=args.out, existing=existing,
                                seed_ctrls=seed_ctrls, obj_kwargs=obj_kwargs,
                                bins=([args.only_bin] if args.only_bin else None))

    if args.workers > 1 and not args.only_bin:
        _save(args.out, library, space)

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
