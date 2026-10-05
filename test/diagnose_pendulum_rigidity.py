"""
diagnose_pendulum_rigidity.py
==============================
One-shot diagnostic for tonight: for ONE scenario/condition/pose, runs all
4 combinations of {unprotected, protected} x {rigid pendulum (kp=800 hold,
current default), passive pendulum (gain/bias zeroed)} and prints peak
head/pelvis/other force for each.

This answers the question that matters before touching anything else:
  - If UNPROTECTED already dumps ~100% of force onto head with rigid
    pendulum -> the "impact hits only the pendulum" behavior is a
    pre-existing rig/geometry characteristic (present since Phase 1's own
    baseline), not something the v6 leg-only search introduced. The fix is
    then about whether a passive pendulum is a better physical model (rows
    3-4 below), not about re-tuning CMA-ES.
  - If UNPROTECTED is fine (force distributed pelvis/other, some head) but
    PROTECTED (with a v6 pose) concentrates everything on head -> the v6
    LEG POSTURE itself is doing something that exposes the head (e.g.
    raising the hips clear while leaving the torso as the only thing that
    reaches the ground) -> a genuinely different problem, more about
    W_HEAD/objective shaping than pendulum physics.

Usage:
  python diagnose_pendulum_rigidity.py \
      --model g1_pendulum.xml --p1-module generate_fall_dataset_final \
      --pose-library pose_library_v6.json --pose-id 1 --bin-index 0 \
      --condition-index 2

`--condition-index` picks which of the 4 sampled (timing, jitter)
conditions to run (0-3, matching COND_TIMINGS x COND_JITTERS order in
phase3_pose_jerk_v7.py) -- pick the one whose diagnostics in the JSON
show the bad head-only spike you want to reproduce.
"""
import argparse
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--p1-module", default="generate_fall_dataset_final")
    ap.add_argument("--phase3-module", default="phase3_pose_jerk_v7",
                     help="Import name of the patched phase3 script (no .py).")
    ap.add_argument("--pose-library", required=True,
                     help="e.g. pose_library_v6.json -- the run showing the bad spike.")
    ap.add_argument("--pose-id", type=int, required=True)
    ap.add_argument("--bin-index", type=int, default=0,
                     help="Index into that pose's velocity_control_points.")
    ap.add_argument("--condition-index", type=int, default=0,
                     help="Which of the 4 sampled (timing, jitter) conditions to run.")
    ap.add_argument("--lead-time-s", type=float, default=0.30,
                     help="Trigger lead time in seconds, matching the run "
                          "that produced --pose-library (POSE_TRANSITION_BUDGET_S "
                          "region -- 0.30 is phase3_pose_jerk_v7's default budget).")
    args = ap.parse_args()

    sys.path.insert(0, ".")
    import importlib
    p3 = importlib.import_module(args.phase3_module)
    p1 = importlib.import_module(args.p1_module)

    with open(args.pose_library) as f:
        lib = json.load(f)
    pose_entry = None
    for p in lib["poses"]:
        if p["pose_id"] == args.pose_id:
            pose_entry = p
            break
    if pose_entry is None:
        raise SystemExit(f"pose_id {args.pose_id} not found in {args.pose_library}")
    vc = pose_entry["velocity_control_points"][args.bin_index]
    pose_ctrl = np.array(vc["pose"], dtype=float)
    # velocity_control_points don't carry their own scenario_id (a merged
    # pose's curve is shared across every scenario it covers) -- use the
    # first covered scenario as a concrete, reproducible stand-in. Good
    # enough for tonight's yes/no diagnostic; it does not need to be the
    # exact original scenario that produced this exact score.
    scen_id = pose_entry["covers_scenario_ids"][0]
    magnitude = vc["magnitude_used"]
    print(f"Pose {args.pose_id}, bin {args.bin_index}: scenario_id={scen_id}, "
          f"v={vc['velocity_mps']:.3f} m/s, magnitude={magnitude:.3f}, "
          f"stored score={vc['score']:.1f}, stored diagnostics[condition_index]="
          f"{vc['diagnostics'][args.condition_index] if args.condition_index < len(vc['diagnostics']) else 'N/A'}")

    # Resolve the actual scenario tuple from generate_fall_dataset_final's
    # own scenario table -- do NOT hand-guess (category, fn_name, direction).
    scenario = None
    for s in p1.SCENARIOS if hasattr(p1, "SCENARIOS") else []:
        if s[0] == scen_id:
            scenario = s
            break
    if scenario is None:
        raise SystemExit(
            f"Could not resolve scenario_id={scen_id} against {args.p1_module}'s "
            "own scenario table -- check the attribute name (expected SCENARIOS) "
            "and adjust this script rather than guessing the scenario tuple."
        )

    timing = p3.COND_TIMINGS[args.condition_index // len(p3.COND_JITTERS)]
    jitter = p3.COND_JITTERS[args.condition_index % len(p3.COND_JITTERS)]
    scen_id_, category, fn_name, nominal_dir = scenario
    direction_deg = nominal_dir + jitter

    def run(passive, protected):
        model = p3.load_instrumented_model(args.model, passive_pendulum=passive)
        snapshot = p3.snapshot_model_state(model)
        impact_ids = p3.impact_body_ids(model)
        stand = model.key_ctrl[0].copy()
        ctrl_to_use = pose_ctrl if protected else stand
        # trigger_lead_s = a huge negative-equivalent for "unprotected": we
        # instead just command `stand` as the "pose" so the minimum-jerk
        # trajectory is a no-op (q_start ~= stand already), reusing the
        # exact same stepping/scoring code path for both conditions.
        result = p3.run_protected_trial(
            model, scenario, magnitude, direction_deg, timing,
            ctrl_to_use, args.lead_time_s, impact_ids, snapshot, p1)
        return result

    print(f"\nScenario {scen_id} / {fn_name}, direction={direction_deg:.1f} deg, "
          f"timing={timing}, jitter={jitter}, magnitude={magnitude:.3f}\n")
    header = f"{'condition':<28}{'head N':>10}{'pelvis N':>10}{'other N':>10}{'fell':>7}"
    print(header)
    print("-" * len(header))
    for label, passive, protected in [
        ("unprotected / rigid", False, False),
        ("unprotected / passive", True, False),
        ("protected(v6 pose) / rigid", False, True),
        ("protected(v6 pose) / passive", True, True),
    ]:
        result = run(passive, protected)
        print(f"{label:<28}{result.peak_force_head:>10.1f}{result.peak_force_pelvis:>10.1f}"
              f"{result.peak_force_other:>10.1f}{str(result.fell):>7}")

    print(
        "\nRead this as: if rows 1-2 already show head absorbing ~all the "
        "force with rigid pendulum, that's a pre-existing rig characteristic "
        "-- compare rows 1 vs 2 to see whether a passive pendulum changes "
        "the UNPROTECTED case too. If rows 1-2 look reasonable but row 3 "
        "concentrates on head, the v6 leg posture itself is the cause -- "
        "row 4 shows whether the passive-pendulum fix resolves it."
    )


if __name__ == "__main__":
    main()
