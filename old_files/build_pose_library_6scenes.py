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

def iter_tuning_conditions_MOCK(scenario: str, bin_name: str) -> List[dict]:
    return [{"seed": i} for i in range(6)]


def iter_holdout_conditions_MOCK(scenario: str, bin_name: str) -> List[dict]:
    return [{"seed": 100 + i} for i in range(14)]


def iter_nofall_conditions_MOCK(scenario: str, bin_name: str) -> List[dict]:
    return [{"seed": 200 + i} for i in range(10)]


def run_one_trial_ADAPT(spec: ValiditySpec, ctrl: np.ndarray, scenario: str,
                        condition: dict, passive_pendulum: bool = False) -> TrialResult:
    """DEPRECATED sketch -- see make_run_trial() below, which wires directly
    to the real run_protected_trial signature (from phase3_pose_jerk_v7.py,
    pasted 2026-09) and is what build_library actually uses in --live mode.
    Kept only so the module docstring's numbered list still points somewhere.
    """
    raise NotImplementedError("use make_run_trial(...) instead -- see below")


# NOTE: SCENARIO_REGISTRY (mapping PLAN label -> real p1.SCENARIOS tuple) is
# now built at runtime in main()'s --live branch from SCENARIO_INDEX_MAP
# below, once p1 is actually importable. make_run_trial still takes a
# scenario_registry dict as before -- nothing else changes.


def make_run_trial(model, p1, impact_ids, snapshot, scenario_registry: Dict[str, tuple],
                   trigger_lead_s: float = 0.3) -> Callable:
    """Binds the per-run-fixed arguments (model, p1, impact_ids, snapshot,
    trigger_lead_s) once, and returns a run_trial(spec, ctrl, scenario,
    condition) closure matching what tune_pose/gate_pose expect. `condition`
    dicts only need to carry the PER-CONDITION values: magnitude,
    direction_deg, timing_phase_s (i.e. exactly what your existing
    COND_TIMINGS/COND_JITTERS already vary per trial).

    Requires run_protected_trial to accept an optional `contact_tracker=None`
    kwarg and call `contact_tracker.update(d)` once per step (2-line addition,
    see build_pose_library.py's module docstring for the exact diff). Falls
    back to the old flat pelvis/head/other split with a one-time warning if
    that param isn't present yet, so this still runs before you've patched it
    -- but the gate's fragile-first-contact check won't be meaningful until
    you have.
    """
    from validity_spec import ContactTracker
    _warned = {"once": False}

    def run_trial(spec: ValiditySpec, ctrl: np.ndarray, scenario: str, condition: dict) -> TrialResult:
        from phase3_pose_jerk_v7 import run_protected_trial  # local import: only needed in live mode

        scen_tuple = scenario_registry.get(scenario)
        if scen_tuple is None:
            raise KeyError(f"SCENARIO_REGISTRY['{scenario}'] is not filled in -- see TODO ADAPT above")

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
                print("[warn] run_protected_trial has no contact_tracker param yet -- "
                     "falling back to the flat pelvis/head/other split. Patch it "
                     "(see module docstring) for the gate's fragile-contact check "
                     "to mean anything.")
                _warned["once"] = True
            r = run_protected_trial(**kwargs)
            peaks = {"head": r.peak_force_head, "pelvis": r.peak_force_pelvis,
                    "other": r.peak_force_other, "knee_shank": 0.0, "thigh_hip": 0.0, "foot": 0.0}
            first_contact = None

        return TrialResult(fell=r.fell, peaks=peaks, first_contact_class=first_contact,
                           tracking_err_rad=r.peak_tracking_error_rad)

    return run_trial


# =============================================================================
# ADAPT #2 / #3: condition iterators
# =============================================================================

# =============================================================================
# Real condition generators, built from generate_fall_dataset_final.py
# (pasted 2026-09): SCENARIOS list, MAGNITUDE_RANGES, and the timing/jitter
# grid used in build_trial_plan (TIMING_LEVELS=5 over [0, 0.8]s, jitters
# [-30,-15,15,30] deg for directional scenarios).
# =============================================================================

# index into p1.SCENARIOS for each PLAN group. p1.SCENARIOS[i] =
#   1 forward push(0deg) | 2 backward push(180) | 3 left push(90) | 4 right push(270)
#   8 floor_tilt_roll (surface)  | 15 sudden_load (trip)
# picked floor_tilt_roll (not _pitch, index 6) to match the pose-12 anomaly
# bin already on file in technical-learnings; swap to index 6 if you'd rather
# tune against floor_tilt_pitch instead (or add both as separate PLAN rows).
SCENARIO_INDEX_MAP: Dict[str, int] = {
    "forward": 0, "backward": 1, "left": 2, "right": 3,
    "sudden_load": 14, "floor_tilt": 7,
}
# Must match the inline jitters list inside p1.build_trial_plan -- it isn't
# exported as a module-level name there, so keep this in sync by hand if that
# list ever changes.
JITTERS_DEG = [-30, -15, 15, 30]
# Velocity-bin -> fraction of each scenario's MAGNITUDE_RANGES span to sample
# from. This is a simple proxy (bigger magnitude ~ faster fall), not measured
# from actual trigger velocities -- swap for a real velocity-quantile split
# once you have Stage-3 velocity readings per condition.               [CONFIRM]
BIN_MAGNITUDE_FRACTIONS = {"low": (0.0, 0.33), "mid": (0.33, 0.66), "high": (0.66, 1.0)}


def make_condition_iters(p1, scenario_registry: Dict[str, tuple]) -> Tuple[Callable, Callable, Callable]:
    """Returns (iter_tuning, iter_holdout, iter_nofall) closures bound to the
    real p1 module and a filled scenario_registry (see SCENARIO_INDEX_MAP)."""

    def _sample(scenario: str, bin_name: str, n: int, seed_tag: str, nofall: bool = False) -> List[dict]:
        scen_id, category, fn_name, nominal_dir = scenario_registry[scenario]
        lo, hi = p1.MAGNITUDE_RANGES[fn_name]
        rng = np.random.default_rng(abs(hash((scenario, bin_name, seed_tag))) % (2**32))
        if nofall:
            # sub-threshold: below the calibrated ~onset magnitude `lo` (recall
            # calibrate_magnitude sets lo ~= crossover*0.5, i.e. already a
            # below-the-fall-rate-target point) plus one true standing case.
            mag_lo, mag_hi = 0.0, 0.5 * lo
            conds = [{"magnitude": 0.0, "direction_deg": float(nominal_dir or 0.0), "timing_phase_s": 0.3}]
            n -= 1
        else:
            f0, f1 = BIN_MAGNITUDE_FRACTIONS[bin_name]
            mag_lo, mag_hi = lo + f0 * (hi - lo), lo + f1 * (hi - lo)
            conds = []
        for _ in range(n):
            mag = float(rng.uniform(mag_lo, mag_hi))
            timing = float(rng.uniform(0.0, 0.8))
            if nominal_dir is not None:
                direction = float((nominal_dir + rng.choice(JITTERS_DEG)) % 360)
            else:
                direction = 0.0
            conds.append({"magnitude": mag, "direction_deg": direction, "timing_phase_s": timing})
        return conds

    def iter_tuning(scenario: str, bin_name: str) -> List[dict]:
        return _sample(scenario, bin_name, 6, "tune")

    def iter_holdout(scenario: str, bin_name: str) -> List[dict]:
        return _sample(scenario, bin_name, 14, "holdout")

    def iter_nofall(scenario: str, bin_name: str) -> List[dict]:
        return _sample(scenario, bin_name, 10, "nofall", nofall=True)

    return iter_tuning, iter_holdout, iter_nofall


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
    ap.add_argument("--out", default="pose_library.json")
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=25)
    ap.add_argument("--mock", action="store_true", help="use the built-in mock trial fn to test the pipeline")
    ap.add_argument("--live", action="store_true", help="use the real harness via make_run_trial (see SCENARIO_REGISTRY)")
    args = ap.parse_args(argv)

    model = mujoco.MjModel.from_xml_path(args.model)
    spec = build_spec(model)

    if args.mock:
        run_trial, it, ih, inf = (run_one_trial_MOCK, iter_tuning_conditions_MOCK,
                                  iter_holdout_conditions_MOCK, iter_nofall_conditions_MOCK)
        print("*** MOCK MODE: physics is fake, this only exercises CMA-ES + gate wiring ***\n")
    elif args.live:
            import generate_fall_dataset_final as p1
            from phase3_pose_jerk_v7 import (
                snapshot_model_state,
                load_instrumented_model,
                impact_body_ids,
            )

            model = load_instrumented_model(args.model)

            registry = {
                label: p1.SCENARIOS[idx]
                for label, idx in SCENARIO_INDEX_MAP.items()
            }

            impact_ids = impact_body_ids(model)
            snapshot = snapshot_model_state(model)

            run_trial = make_run_trial(
                model, p1, impact_ids, snapshot, registry
            )

            it, ih, inf = make_condition_iters(p1, registry)

            print(
                "*** LIVE MODE *** -- verify SCENARIO_INDEX_MAP picks the scenarios "
                "you intend (esp. 'floor_tilt' -> floor_tilt_roll, not _pitch) "
                "before trusting output.\n"
            )
    else:
        print("Pass --mock to test the pipeline, or --live to run against your real harness "
             "(after filling in SCENARIO_REGISTRY and the condition iterators). Exiting.")
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
