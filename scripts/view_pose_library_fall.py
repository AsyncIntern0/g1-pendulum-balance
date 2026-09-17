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
    Phase-1 commonly provides standing control. If present, use it exactly.
    Otherwise leave controls at zero.
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
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
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

        apply_standing_control(p1, model, data)

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
        apply_standing_control(p1, model, data)
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


def select_pose(p3, library, fn_name, direction_label, velocity):
    fn = getattr(p3, "select_pose", None)
    if not callable(fn):
        raise RuntimeError(
            "phase3_pose_jerk_v5.py with select_pose() could not be loaded."
        )

    # Use the project's actual Stage-4 selection function.
    return fn(library, fn_name, direction_label, velocity)


def extract_selected_pose(selected):
    if not isinstance(selected, dict):
        raise TypeError(
            "Stage-4 select_pose() returned an unexpected type: "
            f"{type(selected).__name__}"
        )

    if "joint_angles" not in selected:
        raise KeyError(
            "Selected pose does not contain 'joint_angles'."
        )

    pose = np.asarray(selected["joint_angles"], dtype=float)

    if pose.ndim != 1:
        raise ValueError(
            f"Selected joint_angles must be 1-D; got shape {pose.shape}."
        )

    return pose, selected


def get_pose_id(selected):
    return selected.get("pose_id", "?")


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
            apply_standing_control(p1, model, data)

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
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default=r"C:\Users\Asyncronix\Downloads\Asyncronix_Intern"
                r"\g1-pendulum-balance\unitree_g1\g1_pendulum.xml",
    )

    ap.add_argument(
        "--library",
        default=r"C:\Users\Asyncronix\Downloads\Asyncronix_Intern"
                r"\g1-pendulum-balance\phase3_out_v5\pose_library_v5.json",
    )

    ap.add_argument(
        "--p1-module",
        default="generate_fall_dataset_final",
    )

    ap.add_argument(
        "--p3-module",
        default="phase3_pose_jerk_v5",
    )

    ap.add_argument(
        "--scenario",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--magnitude",
        type=float,
        default=100,
    )

    ap.add_argument(
        "--velocity",
        type=float,
        default=None,
        help=(
            "Optional override for Stage-4 lookup only. "
            "Normally leave unset so velocity is measured at trigger."
        ),
    )

    ap.add_argument(
        "--timing",
        type=float,
        default=0.0,
    )

    args = ap.parse_args()

    root = Path.cwd()

    model_path = resolve_path(root, args.model)
    library_path = resolve_path(root, args.library)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not library_path.exists():
        raise FileNotFoundError(f"Library not found: {library_path}")

    p1 = load_module(args.p1_module)
    p3 = load_module(args.p3_module)

    scenario = find_scenario(p1, args.scenario)
    scen_id, category, fn_name, nominal_direction = scenario_parts(
        scenario
    )

    model = mujoco.MjModel.from_xml_path(str(model_path))

    with open(library_path, "r", encoding="utf-8") as f:
        library = json.load(f)

    direction_label = direction_label_from_phase3(p3, scenario)

    print("\n" + "=" * 68)
    print("PHASE-3 POSE LIBRARY VISUAL COMPARISON")
    print("=" * 68)
    print(f"Model    : {model_path}")
    print(f"Library  : {library_path}")
    print(f"Scenario : {scen_id}")
    print(f"Cause    : {category}")
    print(f"Magnitude: {args.magnitude:.3f}")
    print(f"Direction: {direction_label}")

    # Find natural impact and then measure qvel[0:3] at exactly 0.30 s
    # before that impact.
    natural_impact, trigger_time, measured_velocity = (
        natural_impact_and_trigger(
            p1,
            model,
            scenario,
            args.magnitude,
            nominal_direction,
            args.timing,
        )
    )

    lookup_velocity = (
        float(args.velocity)
        if args.velocity is not None
        else measured_velocity
    )

    selected = select_pose(
        p3,
        library,
        fn_name,
        direction_label,
        lookup_velocity,
    )

    pose, meta = extract_selected_pose(selected)

    if pose.size != model.nu:
        raise ValueError(
            f"Selected pose has {pose.size} joint angles, "
            f"but the XML has model.nu={model.nu} actuators."
        )

    pose_id = get_pose_id(meta)

    print(f"Natural impact time : {natural_impact:.3f}s")
    print(f"Trigger time        : {trigger_time:.3f}s")
    print(
        f"Velocity at trigger: {measured_velocity:.3f} m/s"
    )

    if args.velocity is not None:
        print(
            f"Velocity used for lookup (--velocity): "
            f"{lookup_velocity:.3f} m/s"
        )
    else:
        print(
            f"Velocity used for lookup: "
            f"{lookup_velocity:.3f} m/s"
        )

    print(f"Pose ID             : {pose_id}")

    if isinstance(meta, dict):
        if "velocity_range_covered_mps" in meta:
            print(
                "Pose velocity range: "
                f"{meta['velocity_range_covered_mps']}"
            )
        if "velocity_out_of_range" in meta:
            print(
                "Velocity out of range: "
                f"{meta['velocity_out_of_range']}"
            )

    print("\nMuJoCo viewer controls are through this terminal:")
    print("  u = unprotected")
    print("  p = protected")
    print("  r = protected again")
    print("  q = quit")

    # Save the exact experiment values so u/p/r use identical conditions.
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
            unprotected_peak = run_viewer(
                p1,
                model,
                scenario,
                args.magnitude,
                nominal_direction,
                args.timing,
                False,
                None,
                None,
                trigger_time,
            )

        elif choice in ("p", "r"):
            protected_peak = run_viewer(
                p1,
                model,
                scenario,
                args.magnitude,
                nominal_direction,
                args.timing,
                True,
                pose,
                pose_id,
                trigger_time,
            )

        else:
            print("Use u, p, r, or q.")

        if unprotected_peak is not None and protected_peak is not None:
            print("\n" + "=" * 68)
            print("FORCE COMPARISON")
            print("=" * 68)
            print(f"{'':16s}{'UNPROTECTED':>16s}{'PROTECTED':>16s}")
            print(
                f"{'Head':16s}"
                f"{unprotected_peak['head']:16.2f}"
                f"{protected_peak['head']:16.2f}"
            )
            print(
                f"{'Pelvis':16s}"
                f"{unprotected_peak['pelvis']:16.2f}"
                f"{protected_peak['pelvis']:16.2f}"
            )
            print(
                f"{'Other':16s}"
                f"{unprotected_peak['other']:16.2f}"
                f"{protected_peak['other']:16.2f}"
            )
            print("=" * 68)


if __name__ == "__main__":
    main()
