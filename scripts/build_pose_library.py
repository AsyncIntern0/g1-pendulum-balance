#!/usr/bin/env python3
"""
build_pose_library.py -- orchestrates: template params -> CMA-ES -> gate ->
pose_library.json, replacing phase3_pose_design.py's free 12-dim search.

INTEGRATION -- 3 things to adapt to your existing codebase
    This file is self-contained and runnable as-is (it ships a MOCK trial
    function so you can see the whole pipeline run before touching your real
    harness). To go live, replace the three functions marked ADAPT below with
    calls into your existing phase3_pose_jerk_v7.py:

    1. run_one_trial(spec, ctrl, scenario, condition, passive_pendulum=False)
       -> should call YOUR run_protected_trial (protected) or the equivalent
          unprotected pass, and return a validity_spec.TrialResult built from
          the SAME ContactTracker peaks your scoring already computes, plus
          `tracking_err_rad` from your existing peak-tracking-error logic.
    2. iter_tuning_conditions(scenario) / iter_holdout_conditions(scenario)
       -> should pull from your existing COND_TIMINGS/COND_JITTERS, but
          SPLIT into two disjoint sets: one CMA-ES optimizes against, one
          reserved for the gate (see technical-learnings: recovery_rate was
          fragile precisely because only ~4 conditions were ever sampled --
          this split is what makes the gate's "held-out" claim honest).
    3. iter_nofall_conditions(scenario)
       -> standing + sub-threshold disturbance magnitudes where the
          UNPROTECTED robot does not fall. This is the do-no-harm set from
          the viewer bug you found (0/0/0 unprotected vs 2749N protected).

WHAT THIS FILE DOES ON ITS OWN (no adaptation needed to see it work)
    - Runs CMA-ES over each PLAN entry's 1-2 template params (not 12 raw
      joint angles) using `cma`.
    - Objective = mean weighted_force over the TUNING conditions only.
    - After tuning, runs the gate (validity_spec.evaluate_gate) against
      HELD-OUT fall conditions + no-fall conditions.
    - Writes pose_library.json: `decide()` result per (group, velocity bin),
      i.e. the pose if it passed the gate, else 'stand' with a reason logged.

USAGE (mock mode, to see the pipeline run)
    python build_pose_library.py --model g1_pendulum.xml --mock

USAGE (live, after adapting the 3 functions above)
    python build_pose_library.py --model g1_pendulum.xml \\
        --out pose_library.json --popsize 8 --maxiter 25
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from validity_spec import (
    GateConfig, GateReport, TrialResult, ValiditySpec, build_spec,
    check_pose_ctrl, decide, evaluate_gate, weighted_force,
)
from pose_templates import PLAN, TEMPLATES, Template, cma_bounds, params_to_ctrl

VELOCITY_BINS = ("low", "mid", "high")  # match these to your existing bin edges


# =============================================================================
# ADAPT #1: run a single trial and return a TrialResult
# =============================================================================

def run_one_trial_MOCK(spec: ValiditySpec, ctrl: np.ndarray, scenario: str,
                       condition: dict, passive_pendulum: bool = False) -> TrialResult:
    """Stand-in physics: deeper knee flexion -> lower force but small chance of
    an induced fall if depth is extreme, JUST so the CMA-ES + gate wiring below
    is exercised end-to-end. Delete this once run_one_trial is adapted."""
    rng = np.random.default_rng(condition.get("seed", 0))
    knee_depth = float(np.clip((ctrl[3] - spec.stand_ctrl[3]) / max(1e-6, (
        TEMPLATES["lower_squat"].params[0].hi)), 0, 1)) if ctrl.size > 3 else 0.0
    base = 3000.0 * (1.0 - 0.55 * knee_depth) + rng.normal(0, 150)
    induced_fall = knee_depth > 0.9 and rng.random() < 0.15
    fell = induced_fall
    other = max(0.0, base) if not fell else max(0.0, base * 1.3)
    return TrialResult(fell=fell, peaks={"head": 0.0, "pelvis": 0.0, "other": other,
                                         "knee_shank": other * 0.3, "thigh_hip": 0.0, "foot": 50.0},
                       first_contact_class="other", tracking_err_rad=0.02 + 0.05 * knee_depth)


def run_one_trial_ADAPT(spec: ValiditySpec, ctrl: np.ndarray, scenario: str,
                        condition: dict, passive_pendulum: bool = False) -> TrialResult:
    """TODO: wire to your real harness. Sketch of what belongs here:

        from phase3_pose_jerk_v7 import run_protected_trial, snapshot_model_state, ...
        with snapshot_model_state(spec.model):
            result = run_protected_trial(spec.model, pose_ctrl=ctrl, scenario=scenario,
                                         **condition, passive_pendulum=passive_pendulum)
        tracker_peaks = result.contact_peaks   # <- replace with validity_spec.ContactTracker
        return TrialResult(fell=result.fell, peaks=tracker_peaks,
                           first_contact_class=result.first_nonfoot_class,
                           tracking_err_rad=result.peak_tracking_error_rad)

    IMPORTANT: peaks must come from validity_spec.ContactTracker (or classes
    matching BODY_CLASSES), not your old flat head/pelvis/other split, or the
    gate's fragile-first-contact and weighted-force checks won't line up.
    """
    raise NotImplementedError("wire this to phase3_pose_jerk_v7.run_protected_trial")


# =============================================================================
# ADAPT #2 / #3: condition iterators
# =============================================================================

def iter_tuning_conditions_MOCK(scenario: str, bin_name: str) -> List[dict]:
    return [{"seed": i} for i in range(6)]


def iter_holdout_conditions_MOCK(scenario: str, bin_name: str) -> List[dict]:
    return [{"seed": 100 + i} for i in range(14)]


def iter_nofall_conditions_MOCK(scenario: str, bin_name: str) -> List[dict]:
    return [{"seed": 200 + i} for i in range(10)]


# TODO ADAPT: replace the three *_MOCK functions above with reads from your
# existing COND_TIMINGS / COND_JITTERS (split tuning vs held-out) and a new
# standing/sub-threshold condition generator for the no-fall set.


# =============================================================================
# Objective + tuning
# =============================================================================

def make_objective(spec: ValiditySpec, tmpl: Template, side: str, scenario: str, bin_name: str,
                   run_trial: Callable, iter_tuning: Callable) -> Callable[[np.ndarray], float]:
    conditions = iter_tuning(scenario, bin_name)

    def objective(x: np.ndarray) -> float:
        ctrl = params_to_ctrl(spec, tmpl, x, side)
        v = check_pose_ctrl(spec, ctrl)
        if v:
            return 1e6  # should never trigger given clamped templates; safety net only
        scores = []
        for cond in conditions:
            r = run_trial(spec, ctrl, scenario, cond)
            scores.append(weighted_force(r.peaks))
        return float(np.mean(scores))

    return objective


def tune_pose(spec: ValiditySpec, tmpl: Template, side: str, scenario: str, bin_name: str,
              run_trial: Callable, iter_tuning: Callable,
              popsize: int = 8, maxiter: int = 25, seed: int = 0) -> np.ndarray:
    x0, lo, hi = cma_bounds(tmpl)
    objective = make_objective(spec, tmpl, side, scenario, bin_name, run_trial, iter_tuning)

    if len(x0) == 1:
        # cma's internal per-coordinate sigma bookkeeping needs n>=2; a 1-D
        # template parameter is cheap enough to grid-search directly instead.
        grid = np.linspace(lo[0], hi[0], max(popsize * 2, 9))
        scores = [objective(np.array([g])) for g in grid]
        return np.array([grid[int(np.argmin(scores))]])

    import cma
    es = cma.CMAEvolutionStrategy(x0, 0.25, {
        "bounds": [lo.tolist(), hi.tolist()], "popsize": popsize, "maxiter": maxiter, "seed": seed,
        "verbose": -9,
    })
    while not es.stop():
        xs = es.ask()
        es.tell(xs, [objective(np.array(x)) for x in xs])
    return np.clip(es.result.xbest, lo, hi)


# =============================================================================
# Gate evaluation
# =============================================================================

def gate_pose(spec: ValiditySpec, tmpl: Template, side: str, x: np.ndarray, scenario: str, bin_name: str,
             run_trial: Callable, iter_holdout: Callable, iter_nofall: Callable,
             cfg: GateConfig) -> Tuple[GateReport, np.ndarray]:
    ctrl = params_to_ctrl(spec, tmpl, x, side)

    fall_pairs, nofall_pairs = [], []
    for cond in iter_holdout(scenario, bin_name):
        u = run_trial(spec, spec.stand_ctrl, scenario, cond)
        p = run_trial(spec, ctrl, scenario, cond)
        fall_pairs.append((u, p))
    for cond in iter_nofall(scenario, bin_name):
        u = run_trial(spec, spec.stand_ctrl, scenario, cond)
        p = run_trial(spec, ctrl, scenario, cond)
        nofall_pairs.append((u, p))

    report = evaluate_gate(fall_pairs, nofall_pairs, cfg)
    return report, ctrl


# =============================================================================
# Full pipeline -> pose_library.json
# =============================================================================

def build_library(spec: ValiditySpec, run_trial: Callable, iter_tuning: Callable,
                  iter_holdout: Callable, iter_nofall: Callable,
                  popsize: int, maxiter: int, cfg: GateConfig) -> dict:
    library: Dict[str, dict] = {}
    for scenario, tmpl_name, side in PLAN:
        tmpl = TEMPLATES[tmpl_name]
        for bin_name in VELOCITY_BINS:
            key = f"{scenario}/{bin_name}"
            t0 = time.time()
            x = tune_pose(spec, tmpl, side, scenario, bin_name, run_trial, iter_tuning,
                         popsize=popsize, maxiter=maxiter)
            report, ctrl = gate_pose(spec, tmpl, side, x, scenario, bin_name,
                                     run_trial, iter_holdout, iter_nofall, cfg)
            static_v = check_pose_ctrl(spec, ctrl)
            kind, final_ctrl = decide(ctrl, static_v, report, spec.stand_ctrl)
            library[key] = {
                "scenario": scenario, "velocity_bin": bin_name,
                "template": tmpl_name, "side": side,
                "params": {ps.name: float(x[i]) for i, ps in enumerate(tmpl.params)},
                "pose_ctrl": final_ctrl.tolist(),
                "kind": kind,  # "pose" or "stand" (fallback)
                "gate_passed": report.passed,
                "gate_checks": {k: {"ok": ok, "msg": msg} for k, (ok, msg) in report.checks.items()},
                "static_violations": static_v,
                "tuning_seconds": round(time.time() - t0, 1),
            }
            print(f"[{key:24s}] {kind:6s} gate={'PASS' if report.passed else 'FAIL'} "
                 f"({time.time()-t0:.1f}s)")
    return library


# =============================================================================
# CLI
# =============================================================================

def main(argv=None) -> int:
    import mujoco
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="build_pose/pose_library.json")
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=25)
    ap.add_argument("--mock", action="store_true", help="use the built-in mock trial fn to test the pipeline")
    args = ap.parse_args(argv)

    model = mujoco.MjModel.from_xml_path(args.model)
    spec = build_spec(model)

    if args.mock:
        run_trial, it, ih, inf = (run_one_trial_MOCK, iter_tuning_conditions_MOCK,
                                  iter_holdout_conditions_MOCK, iter_nofall_conditions_MOCK)
        print("*** MOCK MODE: physics is fake, this only exercises CMA-ES + gate wiring ***\n")
    else:
        print("Live mode requires run_one_trial_ADAPT / iter_*_ADAPT to be wired to your "
             "harness -- see the module docstring. Exiting.")
        return 1

    cfg = GateConfig()
    library = build_library(spec, run_trial, it, ih, inf, args.popsize, args.maxiter, cfg)

    n_pose = sum(1 for v in library.values() if v["kind"] == "pose")
    print(f"\n{n_pose}/{len(library)} bins produced a pose; {len(library)-n_pose} fell back to 'stand'.")
    with open(args.out, "w") as f:
        json.dump(library, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
