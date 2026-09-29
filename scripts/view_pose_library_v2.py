"""
Phase-3 pose-library visual comparison for g1_pendulum.xml.

Run from project root:
    python scripts/view_pose_library_fall.py

Terminal controls:
    u = unprotected
    p = protected
    r = protected again
    q = quit

Important:
- Uses the Phase-1 SCENARIO_BUILDERS mechanism correctly:
      disturb_fn = builder(model, magnitude, direction=...)
      disturb_fn(model, data, t_rel)
- Finds the natural impact first.
- Uses the actual qvel[0:3] at the 0.30 s pre-impact trigger for the
  pose-library lookup, matching Phase-3's velocity-at-trigger concept.
- Protected mode uses the selected library joint_angles with the same
  minimum-jerk transition.
- Both modes use a fresh reset and the same disturbance.
- The MuJoCo viewer runs in real time and remains open after impact so
  the fall and resulting pose can be inspected visually.
"""

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


LEAD_TIME_S = 0.30
POSE_TRANSITION_S = 0.30
POST_IMPACT_HOLD_S = 1.5
MAX_SIM_TIME_S = 4.0


def minimum_jerk(t, T, q_start, q_target):
    if T <= 0:
        return q_target.copy()
    tau = min(max(t / T, 0.0), 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return q_start + (q_target - q_start) * s


def load_module(name_or_path):
    name_or_path = str(name_or_path)
    if name_or_path.endswith(".py"):
        name_or_path = name_or_path[:-3]

    path = Path(name_or_path)
    if path.exists():
        spec = importlib.util.spec_from_file_location(
            path.stem, str(path.resolve())
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module: {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = mod
        spec.loader.exec_module(mod)
        return mod

    return importlib.import_module(name_or_path)


def resolve_path(root, value):
    p = Path(value)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def find_scenario(p1, scenario_id):
    for scenario in p1.SCENARIOS:
        if int(scenario[0]) == scenario_id:
            return scenario
    raise ValueError(f"Scenario {scenario_id} not found in Phase-1 SCENARIOS.")


def scenario_parts(scenario):
    scen_id, category, fn_name, nominal_dir = scenario
    direction = nominal_dir if nominal_dir is not None else 0
    return scen_id, category, fn_name, direction


def make_disturbance(p1, model, fn_name, magnitude, direction, timing):
    """
    This is the important Phase-1 interface.

    The scenario builder creates the disturbance function. The returned
    function is called every simulation step with (model, data, t_rel).
    """
    builder = p1.SCENARIO_BUILDERS.get(fn_name)
    if builder is None:
        raise KeyError(
            f"SCENARIO_BUILDERS has no entry for '{fn_name}'."
        )

    # Phase-1's actual interface.
    disturb_fn = builder(model, magnitude, direction=direction)

    if not callable(disturb_fn):
        raise TypeError(
            f"Scenario builder '{fn_name}' did not return a callable disturbance."
        )

    return disturb_fn


def apply_standing_control(p1, model, data):
    """
    DEPRECATED / DO NOT USE AS THE ONLY STANDING-HOLD MECHANISM.

    generate_fall_dataset_final.py defines NEITHER `apply_standing_control`
    NOR `stand_control` -- both getattr() lookups below always miss, so this
    function has always silently done nothing. Every other script in this
    pipeline (generate_fall_dataset_final.py, phase3_pose_jerk_v7.py) holds
    the stand pose explicitly with `data.ctrl[:] = model.key_ctrl[0].copy()`
    every step instead of going through any p1 API. Kept here (now a no-op
    by design, not by accident) only so an old call site doesn't crash;
    every call site below now sets ctrl explicitly and no longer depends on
    this function doing anything.
    """
    fn = getattr(p1, "apply_standing_control", None)
    if callable(fn):
        try:
            fn(model, data)
            return
        except TypeError:
            pass

    fn = getattr(p1, "stand_control", None)
    if callable(fn):
        try:
            fn(model, data)
        except TypeError:
            pass


def reset_data(model):
    """Reset to the STANDING keyframe (index 0), matching
    generate_fall_dataset_final.py / phase3_pose_jerk_v7.py exactly.

    The previous version called mj_resetData(), which zeroes qpos/qvel
    entirely -- NOT the standing pose. Combined with apply_standing_control()
    silently no-oping (see above), this meant every "unprotected" run in
    this viewer, and both passes of natural_impact_and_trigger() that
    compute trigger_time/velocity_at_trigger, were simulated from a
    non-standing zero pose with ZERO commanded torque throughout -- not a
    real "no protection" baseline, and not comparable to anything Phase 1
    or Phase 3 produces. That silent mismatch, not the pose library or the
    pendulum physics, is what made "unprotected" register 0 force while
    "protected" (the only condition where ctrl was ever actually set)
    looked like it was causing the fall."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    print("RESET ERROR:",
      np.abs(data.qpos - model.key_qpos[0]).max())
    mujoco.mj_forward(model, data)
    return data


def geom_name(model, geom_id):
    if geom_id < 0:
        return ""
    return mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
    ) or ""


def find_ground_and_feet(model, p1):
    ground_id = -1
    foot_ids = set()

    for gid in range(model.ngeom):
        name = geom_name(model, gid).lower()

        if ground_id < 0 and ("ground" in name or "floor" in name):
            ground_id = gid

        if any(x in name for x in ("foot", "toe", "ankle")):
            foot_ids.add(gid)

    fn = getattr(p1, "get_foot_geom_ids", None)
    if callable(fn):
        try:
            ids = fn(model)
            if ids is not None:
                foot_ids.update(int(x) for x in ids)
        except Exception:
            pass

    return ground_id, foot_ids


def contact_force(model, data, contact_id):
    wrench = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, contact_id, wrench)
    return float(np.linalg.norm(wrench[:3]))


def classify_contact(name):
    n = name.lower()

    if any(x in n for x in ("head", "bob", "pendulum")):
        return "head"

    if any(x in n for x in ("pelvis", "hip", "waist")):
        return "pelvis"

    return "other"


def measure_ground_contacts(model, data, ground_id, foot_ids):
    """
    Return current ground-contact forces grouped as head/pelvis/other.

    Foot contacts are deliberately excluded from the impact groups.
    """
    forces = {
        "head": 0.0,
        "pelvis": 0.0,
        "other": 0.0,
    }

    active = {
        "head": False,
        "pelvis": False,
        "other": False,
    }

    for i in range(data.ncon):
        c = data.contact[i]

        if ground_id >= 0:
            if c.geom1 != ground_id and c.geom2 != ground_id:
                continue

            body_geom = c.geom2 if c.geom1 == ground_id else c.geom1
        else:
            # No named ground geom was found. Treat non-foot contacts as
            # candidate ground contacts, as a fallback.
            if c.geom1 in foot_ids or c.geom2 in foot_ids:
                continue
            body_geom = c.geom1

        if body_geom in foot_ids:
            continue

        part = classify_contact(geom_name(model, body_geom))
        f = contact_force(model, data, i)

        forces[part] += f
        active[part] = True

    return forces, active


def natural_impact_and_trigger(
    p1, model, scenario, magnitude, direction, timing
):
    """
    Reproduce the natural Phase-1 disturbance and find first non-foot ground
    contact. At trigger = impact - 0.30 s, record qvel[0:3], matching the
    Phase-3 velocity-at-trigger measurement concept.
    """
    scen_id, category, fn_name, _ = scenario

    data = reset_data(model)
    print("RESET ERROR:",
      np.abs(data.qpos - model.key_qpos[0]).max())
    stand_ctrl = model.key_ctrl[0].copy()
    print("CTRL ERROR:",
      np.abs(data.ctrl - model.key_ctrl[0]).max())
    disturb_fn = make_disturbance(
        p1, model, fn_name, magnitude, direction, timing
    )

    ground_id, foot_ids = find_ground_and_feet(model, p1)
    dt = float(model.opt.timestep)

    impact_time = None
    velocity_at_trigger = None
    trigger_time = None

    # First pass: find natural impact time.
    for _ in range(int(MAX_SIM_TIME_S / dt)):
        t_rel = float(data.time)

        data.ctrl[:] = stand_ctrl

        # Same returned Phase-1 disturbance function used at every step.
        disturb_fn(model, data, t_rel)

        mujoco.mj_step(model, data)


        _, active = measure_ground_contacts(
            model, data, ground_id, foot_ids
        )

        if any(active.values()):
            impact_time = float(data.time)
            break

    if impact_time is None:
        raise RuntimeError(
            "Natural fall did not produce a non-foot ground contact "
            f"within {MAX_SIM_TIME_S:.1f}s."
        )

    trigger_time = max(0.0, impact_time - LEAD_TIME_S)

    # Second pass: same disturbance again, stopping exactly at trigger.
    data = reset_data(model)
    disturb_fn = make_disturbance(
        p1, model, fn_name, magnitude, direction, timing
    )

    while data.time < trigger_time:
        t_rel = float(data.time)
        data.ctrl[:] = stand_ctrl
        disturb_fn(model, data, t_rel)

        step_before = float(data.time)
        mujoco.mj_step(model, data)
 
                

        if data.time == step_before:
            break

    velocity_at_trigger = float(np.linalg.norm(data.qvel[0:3]))

    return impact_time, trigger_time, velocity_at_trigger


def direction_label_from_phase3(p3, scenario):
    fn = getattr(p3, "_direction_label", None)
    if callable(fn):
        try:
            return fn(scenario)
        except Exception:
            pass

    _, _, _, direction = scenario_parts(scenario)

    mapping = {
        0: "forward",
        90: "left",
        180: "backward",
        270: "right",
        45: "forward-left",
        315: "forward-right",
    }

    return mapping.get(direction, str(direction))


def select_pose_from_v2_library(library, scenario_id, velocity_bin):
    """Select the v2 entry matching the Phase-1 scenario id and chosen bin."""
    matches = []
    for key, entry in library.items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        ids = entry.get("scenario_ids", [])
        if scenario_id in ids and entry.get("velocity_bin") == velocity_bin:
            matches.append((key, entry))
    if not matches:
        raise KeyError(
            f"No library entry for scenario id {scenario_id}, bin {velocity_bin}."
        )
    # A scenario may map to a single grouped unit. Fail clearly if ambiguous.
    if len(matches) > 1:
        raise ValueError(f"Ambiguous library matches: {[k for k, _ in matches]}")
    key, entry = matches[0]
    pose = np.asarray(entry.get("pose_ctrl", []), dtype=float)
    if pose.ndim != 1:
        raise ValueError(f"{key}: pose_ctrl must be a 1-D vector")
    return pose, key, entry


def print_force_summary(label, peak):
    print(f"\n[{label} PEAK FORCES]")
    print(f"  Head   : {peak['head']:.2f} N")
    print(f"  Pelvis : {peak['pelvis']:.2f} N")
    print(f"  Other  : {peak['other']:.2f} N")


def run_viewer(
    p1,
    model,
    scenario,
    magnitude,
    direction,
    timing,
    protected,
    pose,
    pose_id,
    trigger_time,
):
    """
    Run one complete real-time visual experiment.
    """
    scen_id, category, fn_name, _ = scenario_parts(scenario)

    data = reset_data(model)
    stand_ctrl = model.key_ctrl[0].copy()
    disturb_fn = make_disturbance(
        p1, model, fn_name, magnitude, direction, timing
    )

    ground_id, foot_ids = find_ground_and_feet(model, p1)

    q_start = None
    triggered = False
    first_contact = False
    impact_time = None

    peak = {
        "head": 0.0,
        "pelvis": 0.0,
        "other": 0.0,
    }

    # Viewer camera.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.7
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15

        wall_start = time.perf_counter()

        print("\n" + "=" * 68)
        print("PROTECTED" if protected else "UNPROTECTED")
        print("=" * 68)
        print(f"Scenario : {scen_id} / {category}")
        print(f"Magnitude: {magnitude:.3f}")
        print(f"Direction: {direction}")

        if protected:
            print(f"Pose ID  : {pose_id}")
            print(f"Trigger  : {trigger_time:.3f}s")

        while viewer.is_running() and data.time < MAX_SIM_TIME_S:
            sim_t = float(data.time)

            # Real-time pacing.
            target_wall = wall_start + sim_t
            remaining = target_wall - time.perf_counter()
            if remaining > 0:
                time.sleep(min(remaining, 0.004))

            # Keep standing/PD control exactly as Phase-1 normally does.
            data.ctrl[:] = stand_ctrl

            # The SAME disturbance mechanism as Phase-1.
            disturb_fn(model, data, sim_t)

            # Trigger protection before the natural impact.
            if protected and not triggered and sim_t >= trigger_time:
                q_start = data.qpos[7:7 + model.nu].copy()
                triggered = True
                print(
                    f"[TRIGGER] pose {pose_id} at {sim_t:.3f}s"
                )

            if protected and triggered:
                elapsed = sim_t - trigger_time
                q_cmd = minimum_jerk(
                    elapsed,
                    POSE_TRANSITION_S,
                    q_start,
                    pose,
                )
                data.ctrl[:model.nu] = q_cmd

            mujoco.mj_step(model, data)


            forces, active = measure_ground_contacts(
                model, data, ground_id, foot_ids
            )

            for key in peak:
                peak[key] = max(peak[key], forces[key])

            if any(active.values()) and not first_contact:
                first_contact = True
                impact_time = float(data.time)

                print(
                    f"[CONTACT] non-foot ground contact at "
                    f"{impact_time:.3f}s"
                )
                print(
                    f"[CONTACT FORCE] "
                    f"head={forces['head']:.2f} N, "
                    f"pelvis={forces['pelvis']:.2f} N, "
                    f"other={forces['other']:.2f} N"
                )

            viewer.sync()

            # Continue showing the post-impact state.
            if (
                first_contact
                and data.time >= impact_time + POST_IMPACT_HOLD_S
            ):
                break

        viewer.sync()

    print_force_summary(
        "PROTECTED" if protected else "UNPROTECTED",
        peak,
    )

    return peak


def main():
    ap = argparse.ArgumentParser(description="Compare unprotected and pose-library protected MuJoCo fall runs.")
    ap.add_argument("--model", required=True, help="Path to g1_pendulum.xml")
    ap.add_argument("--library", required=True, help="Path to pose_lib_widebox.json")
    ap.add_argument("--p1-module", default="generate_fall_dataset_final")
    ap.add_argument("--scenario", type=int, default=9, help="Phase-1 scenario ID")
    ap.add_argument("--bin", dest="velocity_bin", choices=("low", "mid", "high"), default=None,
                    help="Pose-library velocity bin; if omitted, prompt interactively")
    ap.add_argument("--magnitude", type=float, default=100.0)
    ap.add_argument("--timing", type=float, default=0.0)
    args = ap.parse_args()

    root = Path.cwd()
    model_path = resolve_path(root, args.model)
    library_path = resolve_path(root, args.library)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not library_path.exists():
        raise FileNotFoundError(f"Pose library not found: {library_path}")

    p1 = load_module(args.p1_module)
    scenario = find_scenario(p1, args.scenario)
    scen_id, category, fn_name, direction = scenario_parts(scenario)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with open(library_path, "r", encoding="utf-8") as f:
        library = json.load(f)

    velocity_bin = args.velocity_bin
    if velocity_bin is None:
        print("Available velocity bins: low / mid / high")
        velocity_bin = input("Select velocity bin: ").strip().lower()
    if velocity_bin not in ("low", "mid", "high"):
        raise ValueError("Velocity bin must be low, mid, or high.")

    pose, entry_key, entry = select_pose_from_v2_library(library, int(scen_id), velocity_bin)
    if pose.size != model.nu:
        raise ValueError(f"Selected pose has {pose.size} controls, but model.nu={model.nu}.")
    pose_id = entry_key + (" [gate-passed]" if entry.get("gate_passed") else " [stand/fallback]")

    print("\n" + "=" * 68)
    print("PHASE-3 POSE LIBRARY: PROTECTED vs UNPROTECTED")
    print("=" * 68)
    print(f"Scenario : {scen_id} / {category} ({fn_name})")
    print(f"Bin      : {velocity_bin}")
    print(f"Entry    : {entry_key}")
    print(f"Kind     : {entry.get('kind', 'unknown')}")
    print(f"Gate     : {entry.get('gate_passed', False)}")
    if entry.get("skipped_reason"):
        print(f"Note     : {entry['skipped_reason']}")
    print(f"Magnitude: {args.magnitude:.3f}")

    natural_impact, trigger_time, measured_velocity = natural_impact_and_trigger(
        p1, model, scenario, args.magnitude, direction, args.timing
    )
    print(f"Natural impact time : {natural_impact:.3f}s")
    print(f"Trigger time        : {trigger_time:.3f}s")
    print(f"Velocity at trigger : {measured_velocity:.3f} m/s")
    print("\nTerminal controls: u=unprotected, p=protected, r=protected again, q=quit")

    unprotected_peak = None
    protected_peak = None
    while True:
        try:
            choice = input("Choice [u/p/r/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "q"
        if choice == "q":
            print("Viewer closed.")
            break
        if choice == "u":
            unprotected_peak = run_viewer(p1, model, scenario, args.magnitude, direction,
                                          args.timing, False, None, None, trigger_time)
        elif choice in ("p", "r"):
            protected_peak = run_viewer(p1, model, scenario, args.magnitude, direction,
                                        args.timing, True, pose, pose_id, trigger_time)
        else:
            print("Use u, p, r, or q.")
        if unprotected_peak is not None and protected_peak is not None:
            print("\n" + "=" * 68)
            print("PEAK FORCE COMPARISON (N)")
            print("=" * 68)
            print(f"{'Body':12s}{'Unprotected':>16s}{'Protected':>16s}")
            for part in ("head", "pelvis", "other"):
                print(f"{part:12s}{unprotected_peak[part]:16.2f}{protected_peak[part]:16.2f}")


if __name__ == "__main__":
    main()
