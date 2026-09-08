"""
generate_fall_dataset.py  (FINAL)
==================================
Phase 1 dataset generation for AXON-R fall detection/classification.

Baseline controller: position actuators held FIXED at the 'stand' keyframe
ctrl values for the whole episode (passive PD/impedance hold, kp=300 legs,
kp=800 pendulum) — not zero control, not a trained recovery policy. Small
disturbances recover on their own via this stiffness; large ones overwhelm
it. This is what gives an honest, non-fabricated stable/fall split.

Episodes are VARIABLE LENGTH — they end as soon as either:
  - FALL is confirmed: a new non-foot geom contacts the ground (checked
    against a settle-phase baseline contact set, so already-touching geoms
    at rest don't get mislabeled), or
  - STABLE is confirmed: pelvis height/tilt/ang-vel stay inside a band for
    a continuous window after the disturbance,
with MAX_EPISODE_TIME as a hard safety-net cap only (not the real signal).

=====================================================================
CHANGELOG — mechanism fixes from the previous version (all root-caused
against real per-scenario fall-rate curves, not guessed):
=====================================================================
1. MAGNITUDE_RANGES / CALIBRATION_MAX_MAGNITUDE are keyed PER SCENARIO
   FUNCTION, never per category. Categories bundled incompatible units
   (radians vs meters vs kg vs a 0-1 severity fraction) into one shared
   range, which silently corrupted calibration for whichever scenario in
   that category didn't match the others (this caused floor_drop, trip,
   and every actuator scenario to get miscalibrated ranges in earlier runs).

2. floor_drop: pulling a foot DOWN with a force does nothing — the ground
   is rigid, so the pull just increases contact normal force; the foot
   never loses support. Fixed to genuinely disable collision under one
   foot for a short window (contype/conaffinity -> 0), which is what
   "the floor gives way under one foot" actually requires.

3. trip: kicking a single joint's velocity gets absorbed almost instantly
   by that same joint's own strong position gain (kp=300) before it can
   propagate into a whole-body toppling moment. Fixed to apply an external
   force to the foot/shin body instead (same mechanism family as `push`,
   just with a different, more destabilizing lever arm).

4. low_friction: the companion "slip" push was a fixed weak force,
   independent of what push magnitude the robot actually needed to fall.
   Fixed so `magnitude` directly IS the companion push force (same N
   scale as `push`), with friction always cut hard and fixed — this also
   makes it a clean, direct comparison against the `push` scenario later.

5. actuator_fault / asymmetric_gain: originally only reduced stiffness
   (kp), leaving damping (kv) untouched, so a "weakened" joint just sagged
   slowly into a new stable equilibrium — a static double-support stance
   is forgiving enough that this rarely crosses into real instability.
   Fixed to also reduce damping at high severity, so full failure means a
   genuinely floppy, uncontrolled joint that can actually buckle.

6. actuator_delay retired and REPLACED by actuator_stuck. Delay only means
   something if the underlying commanded signal changes over time,  but
   this baseline's ctrl is a constant setpoint (holds `stand` the whole
   episode) — delaying a constant is a no-op by construction, not a
   calibration problem. actuator_stuck (a joint suddenly forced to a
   wrong angle, e.g. a sensor/decoder fault) is physically real and
   testable under a static baseline.

Run:
    python generate_fall_dataset.py --model g1_pendulum.xml --out dataset/ --workers 8
    python generate_fall_dataset.py --model g1_pendulum.xml --calibrate
"""
import tempfile
import argparse
import csv
import os
import time
from dataclasses import dataclass, asdict
from multiprocessing import Pool

import numpy as np
import mujoco


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

SETTLE_TIME = 0.5           # s, let the robot settle on the keyframe before disturbing
MAX_EPISODE_TIME = 4.0      # s, hard cap safety net (post-settle) — not the real stop signal
STABLE_WINDOW = 1.0         # s, continuous time required to confirm "stable"
LOG_HZ = 200                 # Hz, resample sensor stream to a realistic IMU rate
STABLE_PELVIS_Z_MIN = 0.55   # m, below this pelvis height counts toward instability
STABLE_ANGVEL_MAX = 2.0      # rad/s, gyro magnitude above this counts toward instability
PUSH_DURATION = 0.10         # s, how long an impulsive external force is applied

# 15 scenarios: (id, category, scenario_fn, nominal_direction_deg_or_None)
SCENARIOS = [
    (1,  "push",       "push",            0),      # forward
    (2,  "push",       "push",            180),    # backward
    (3,  "push",       "push",            90),     # left
    (4,  "push",       "push",            270),    # right
    (5,  "push",       "push",            45),     # front-left diagonal
    (6,  "push",       "push",            315),    # front-right diagonal
    (7,  "surface",    "floor_tilt_pitch", None),
    (8,  "surface",    "floor_tilt_roll",  None),
    (9,  "surface",    "floor_drop",       None),
    (10, "surface",    "low_friction",     None),
    (11, "actuator",   "actuator_fault",   None),
    (12, "actuator",   "actuator_stuck",   None),   # replaces actuator_delay — see changelog
    (13, "actuator",   "asymmetric_gain",  None),
    (14, "trip",       "trip",             None),
    (15, "trip",       "sudden_load",      None),
]

MAGNITUDE_LEVELS = 5
TIMING_LEVELS = 5
DIRECTION_JITTER_LEVELS = 4
N_VARIATIONS = MAGNITUDE_LEVELS * TIMING_LEVELS * DIRECTION_JITTER_LEVELS  # = 100

# Magnitude ranges keyed PER SCENARIO FUNCTION (see changelog #1). Starting
# points only — always run --calibrate on your real assets before the full
# generation, these will not be right out of the box.
MAGNITUDE_RANGES = {
    "push": (77.168, 231.504),
    "floor_tilt_pitch": (0.051, 0.153),
    "floor_tilt_roll": (0.105, 0.314),
    "floor_drop": (0.134, 0.401),
    "low_friction": (57.129, 171.387),
    "actuator_fault": (0.300, 1.000),
    "actuator_stuck": (0.184, 0.552),
    "asymmetric_gain": (0.300, 1.000),
    "trip": (114.277, 342.832),
    "sudden_load": (9.571, 28.714),
}

# Hard search ceilings so --calibrate can't wander into nonsensical (tilt
# past ~90deg) or saturated (severity > 1.0) territory, or waste time.
CALIBRATION_MAX_MAGNITUDE = {
    "push": 400.0,
    "floor_tilt_pitch": 1.2,
    "floor_tilt_roll": 1.2,
    "floor_drop": 1.0,
    "low_friction": 400.0,
    "actuator_fault": 1.0,
    "actuator_stuck": 1.3,     # keep well inside joint limits
    "asymmetric_gain": 1.0,
    "trip": 400.0,
    "sudden_load": 60.0,
}


@dataclass
class TrialLabel:
    trial_id: str
    scenario_id: int
    scenario_category: str
    scenario_fn: str
    magnitude_level: int
    magnitude_value: float
    timing_level: int
    timing_phase_s: float
    direction_jitter_level: int
    direction_deg: float
    seed: int
    fell: bool
    stable: bool
    fall_direction: str
    time_to_ground_contact: float
    peak_pelvis_ang_vel: float
    peak_pelvis_lin_acc: float
    episode_length_s: float
    recoverable_note: str


# ─────────────────────────────────────────────────────────────────
# DISTURBANCE INJECTORS
# ─────────────────────────────────────────────────────────────────

def get_geom_ids_for_body(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return [i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id]


def make_push_fn(pelvis_id, direction_deg, magnitude):
    theta = np.radians(direction_deg)
    force_vec = magnitude * np.array([np.cos(theta), np.sin(theta), 0.0])

    def fn(m, d, t_rel):
        if t_rel < PUSH_DURATION:
            d.xfrc_applied[pelvis_id, :3] = force_vec
        else:
            d.xfrc_applied[pelvis_id, :3] = 0.0
    return fn


def make_floor_tilt_fn(axis, magnitude):
    magnitude = float(np.clip(magnitude, 0.0, 1.2))  # never let sin/cos wrap past sanity
    g = 9.81
    if axis == "pitch":
        new_gravity = np.array([-g * np.sin(magnitude), 0.0, -g * np.cos(magnitude)])
    else:
        new_gravity = np.array([0.0, -g * np.sin(magnitude), -g * np.cos(magnitude)])

    def fn(m, d, t_rel):
        if t_rel < 0.02:
            m.opt.gravity[:] = new_gravity
    return fn


def make_floor_drop_fn(foot_geom_ids, disable_duration):
    # FIX (changelog #2): genuinely remove contact under one foot for a
    # window, rather than pulling on a rigidly-supported foot (which just
    # increases normal force and does nothing).
    saved = {}

    def fn(m, d, t_rel):
        if abs(t_rel - 0.0) < 1e-6:
            for gid in foot_geom_ids:
                saved[gid] = (int(m.geom_contype[gid]), int(m.geom_conaffinity[gid]))
                m.geom_contype[gid] = 0
                m.geom_conaffinity[gid] = 0
        elif t_rel >= disable_duration and saved:
            for gid, (ct, ca) in saved.items():
                m.geom_contype[gid] = ct
                m.geom_conaffinity[gid] = ca
            saved.clear()
    return fn


def make_low_friction_fn(pelvis_id, magnitude):
    # FIX (changelog #4): magnitude IS the companion push force (Newtons,
    # same scale as `push`); friction is always cut hard and fixed, so the
    # comparison against plain `push` at the same force is direct and clean.
    def fn(m, d, t_rel):
        if abs(t_rel - 0.0) < 1e-6:
            ground_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
            m.geom_friction[ground_gid, 0] = 0.03
        if t_rel < PUSH_DURATION:
            d.xfrc_applied[pelvis_id, 0] = magnitude
        else:
            d.xfrc_applied[pelvis_id, 0] = 0.0
    return fn


def make_actuator_fault_fn(severity):
    # FIX (changelog #5): reduce BOTH stiffness (kp) and damping (kv), not
    # just stiffness — otherwise the joint just damps into a new, still
    # stable, equilibrium instead of genuinely buckling.
    joint_priority = ["right_ankle_pitch", "right_ankle_roll", "right_knee",
                       "right_hip_pitch", "right_hip_roll"]
    n_fail = max(1, int(round(severity * len(joint_priority))))
    failing = joint_priority[:n_fail]
    factor = max(0.02, 1.0 - severity)

    def fn(m, d, t_rel):
        if abs(t_rel - 0.0) < 1e-6:
            for name in failing:
                aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                m.actuator_gainprm[aid, 0] *= factor
                m.actuator_biasprm[aid, 1] *= factor
                m.actuator_biasprm[aid, 2] *= factor   # damping term — see changelog #5
    return fn


def make_actuator_stuck_fn(actuator_name, offset_rad):
    # NEW (changelog #6): joint suddenly forced to a wrong commanded angle
    # (e.g. an encoder/decoder fault), at full stiffness — physically real
    # and testable even under a static, non-reactive baseline.
    state = {"id": None, "orig": None}

    def fn(m, d, t_rel):
        if state["id"] is None:
            state["id"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            state["orig"] = d.ctrl[state["id"]]
        if t_rel >= 0.0:
            d.ctrl[state["id"]] = state["orig"] + offset_rad
    return fn


def make_asymmetric_gain_fn(reduce_side, severity):
    factor = max(0.02, 1.0 - severity)

    def fn(m, d, t_rel):
        if abs(t_rel - 0.0) < 1e-6:
            prefix = "left" if reduce_side == "left" else "right"
            for i in range(m.nu):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if name and name.startswith(prefix):
                    m.actuator_gainprm[i, 0] *= factor
                    m.actuator_biasprm[i, 1] *= factor
                    m.actuator_biasprm[i, 2] *= factor   # damping term — see changelog #5
    return fn


def make_trip_fn(foot_body_id, magnitude):
    # FIX (changelog #3): external force on the foot/shin body instead of a
    # joint-velocity kick, which a single joint's own stiffness absorbed
    # almost instantly before it could destabilize the whole body.
    def fn(m, d, t_rel):
        if t_rel < PUSH_DURATION:
            d.xfrc_applied[foot_body_id, 0] = -magnitude  # pulled backward, as if caught
        else:
            d.xfrc_applied[foot_body_id, 0] = 0.0
    return fn


def make_sudden_load_fn(pendulum_bob_body_id, added_mass_kg):
    def fn(m, d, t_rel):
        if abs(t_rel - 0.0) < 1e-6:
            g = 9.81
            d.xfrc_applied[pendulum_bob_body_id, 2] = -added_mass_kg * g
    return fn


SCENARIO_BUILDERS = {
    "push": lambda m, mag, direction, **kw: make_push_fn(
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"), direction, mag),
    "floor_tilt_pitch": lambda m, mag, **kw: make_floor_tilt_fn("pitch", mag),
    "floor_tilt_roll":  lambda m, mag, **kw: make_floor_tilt_fn("roll", mag),
    "floor_drop":       lambda m, mag, **kw: make_floor_drop_fn(
        get_geom_ids_for_body(m, "right_ankle_roll_link"), mag),
    "low_friction":     lambda m, mag, **kw: make_low_friction_fn(
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"), mag),
    "actuator_fault":   lambda m, mag, **kw: make_actuator_fault_fn(np.clip(mag, 0.0, 1.0)),
    "actuator_stuck":   lambda m, mag, **kw: make_actuator_stuck_fn("right_hip_pitch", mag),
    "asymmetric_gain":  lambda m, mag, **kw: make_asymmetric_gain_fn("left", np.clip(mag, 0.0, 1.0)),
    "trip":             lambda m, mag, **kw: make_trip_fn(
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"), mag),
    "sudden_load":      lambda m, mag, **kw: make_sudden_load_fn(
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pendulum_bob"), mag),
}


# ─────────────────────────────────────────────────────────────────
# CORE ROLLOUT
# ─────────────────────────────────────────────────────────────────

def get_foot_geom_ids(model):
    ids = set()
    for i in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i])
        if bname and "ankle_roll_link" in bname:
            ids.add(i)
    return ids


def snapshot_contacts(model, data, ground_id):
    contacts = set()
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = con.geom1, con.geom2
        if ground_id in (g1, g2):
            other = g2 if g1 == ground_id else g1
            contacts.add(other)
    return contacts


def run_trial(model_path, scenario, magnitude, direction_deg, timing_phase_s,
              seed, trial_id, log_dir):
    scen_id, category, fn_name, nominal_dir = scenario
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    mujoco.mj_resetDataKeyframe(m, d, 0)
    stand_ctrl = m.key_ctrl[0].copy()
    d.ctrl[:] = stand_ctrl

    ground_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    foot_ids = get_foot_geom_ids(m)

    imu_gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu-pelvis-angular-velocity")
    imu_acc_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu-pelvis-linear-acceleration")
    gyro_adr = m.sensor_adr[imu_gyro_id]
    acc_adr = m.sensor_adr[imu_acc_id]

    dt = m.opt.timestep
    log_every = max(1, int(round(1.0 / (LOG_HZ * dt))))

    settle_steps = int(SETTLE_TIME / dt)
    for _ in range(settle_steps):
        d.ctrl[:] = stand_ctrl
        mujoco.mj_step(m, d)
    baseline_contacts = snapshot_contacts(m, d, ground_id)

    builder = SCENARIO_BUILDERS[fn_name]
    disturb_fn = builder(m, magnitude, direction=direction_deg)

    max_steps = int(MAX_EPISODE_TIME / dt)
    timing_steps = int(timing_phase_s / dt)
    stable_window_steps = int(STABLE_WINDOW / dt)

    log_time, log_gyro, log_acc, log_qpos, log_qvel = [], [], [], [], []
    fell, stable = False, False
    t_ground_contact = None
    stable_counter = 0
    peak_angvel, peak_acc = 0.0, 0.0
    step = 0

    while step < max_steps:
        d.ctrl[:] = stand_ctrl
        t_rel = (step - timing_steps) * dt
        if step >= timing_steps:
            disturb_fn(m, d, max(t_rel, 0.0))

        mujoco.mj_step(m, d)
        step += 1

        gyro = d.sensordata[gyro_adr:gyro_adr + 3].copy()
        acc = d.sensordata[acc_adr:acc_adr + 3].copy()
        peak_angvel = max(peak_angvel, float(np.linalg.norm(gyro)))
        peak_acc = max(peak_acc, float(np.linalg.norm(acc)))

        if step % log_every == 0:
            log_time.append(step * dt)
            log_gyro.append(gyro)
            log_acc.append(acc)
            log_qpos.append(d.qpos.copy())
            log_qvel.append(d.qvel.copy())

        current_contacts = snapshot_contacts(m, d, ground_id)
        new_bad_contacts = (current_contacts - baseline_contacts) - foot_ids
        if new_bad_contacts and not fell:
            fell = True
            t_ground_contact = step * dt - timing_phase_s
            break

        if step >= timing_steps:
            pelvis_z = d.qpos[2]
            stable_now = (pelvis_z > STABLE_PELVIS_Z_MIN and
                          float(np.linalg.norm(gyro)) < STABLE_ANGVEL_MAX)
            stable_counter = stable_counter + 1 if stable_now else 0
            if stable_counter >= stable_window_steps:
                stable = True
                break

    fall_direction = "none"
    if fell:
        dx, dy = d.qpos[0], d.qpos[1]
        angle = np.degrees(np.arctan2(dy, dx)) % 360
        if 45 <= angle < 135:
            fall_direction = "left"
        elif 135 <= angle < 225:
            fall_direction = "backward"
        elif 225 <= angle < 315:
            fall_direction = "right"
        else:
            fall_direction = "forward"

    label = TrialLabel(
        trial_id=trial_id, scenario_id=scen_id, scenario_category=category,
        scenario_fn=fn_name, magnitude_level=-1, magnitude_value=float(magnitude),
        timing_level=-1, timing_phase_s=float(timing_phase_s),
        direction_jitter_level=-1,
        direction_deg=float(direction_deg) if direction_deg is not None else -1.0,
        seed=seed, fell=fell, stable=stable, fall_direction=fall_direction,
        time_to_ground_contact=float(t_ground_contact) if t_ground_contact else -1.0,
        peak_pelvis_ang_vel=peak_angvel, peak_pelvis_lin_acc=peak_acc,
        episode_length_s=step * dt, recoverable_note="passive_pd_only",
    )

    npz_path = os.path.join(log_dir, f"{trial_id}.npz")
    np.savez_compressed(
        npz_path, time=np.array(log_time), gyro=np.array(log_gyro),
        acc=np.array(log_acc), qpos=np.array(log_qpos), qvel=np.array(log_qvel),
    )
    return label


def _worker(args):
    return run_trial(*args)


# ─────────────────────────────────────────────────────────────────
# TRIAL PLAN
# ─────────────────────────────────────────────────────────────────

def build_trial_plan(model_path, log_dir, base_seed=0):
    plan = []
    trial_counter = 0
    for scenario in SCENARIOS:
        scen_id, category, fn_name, nominal_dir = scenario
        mag_lo, mag_hi = MAGNITUDE_RANGES[fn_name]
        magnitudes = np.linspace(mag_lo, mag_hi, MAGNITUDE_LEVELS)
        timings = np.linspace(0.0, 0.8, TIMING_LEVELS)
        jitters = [-30, -15, 15, 30]

        for mi, mag in enumerate(magnitudes):
            for ti, timing in enumerate(timings):
                for di, jitter in enumerate(jitters):
                    direction = (nominal_dir + jitter) % 360 if nominal_dir is not None else None
                    seed = base_seed + trial_counter
                    trial_id = f"s{scen_id:02d}_m{mi}_t{ti}_d{di}_seed{seed}"
                    plan.append((model_path, scenario, mag, direction, timing, seed,
                                 trial_id, log_dir))
                    trial_counter += 1
    assert len(plan) == len(SCENARIOS) * N_VARIATIONS, \
        f"expected {len(SCENARIOS) * N_VARIATIONS}, got {len(plan)}"
    return plan


# ─────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────

def calibrate_magnitude(model_path, scenario, target_fall_rate=0.5, n_probe=12,
                         lo=None, hi=None, max_iters=8, seed_base=9000):
    scen_id, category, fn_name, nominal_dir = scenario
    ceiling = CALIBRATION_MAX_MAGNITUDE[fn_name]
    lo = lo if lo is not None else MAGNITUDE_RANGES[fn_name][0]
    hi = min(hi if hi is not None else MAGNITUDE_RANGES[fn_name][1] * 3, ceiling)

    def probe_fall_rate(mag):
        mag = min(mag, ceiling)
        results = []
        for i in range(n_probe):
            timing = np.linspace(0.0, 0.8, n_probe)[i]
            direction = nominal_dir if nominal_dir is not None else 0
            lab = run_trial(model_path, scenario, mag, direction, timing,
                             seed_base + i, f"calib_{scen_id}_{mag:.2f}_{i}", tempfile.gettempdir())
            results.append(lab.fell)
        return np.mean(results)

    rate_lo = probe_fall_rate(lo)
    rate_hi = probe_fall_rate(hi)
    expand_iters = 0
    while rate_hi < 0.8 and hi < ceiling and expand_iters < 5:
        hi = min(hi * 2, ceiling)
        rate_hi = probe_fall_rate(hi)
        expand_iters += 1

    if rate_hi < 0.5:
        print(f"  [WARN] scenario {scen_id} ({fn_name}): fall rate only reaches "
              f"{rate_hi*100:.0f}% even at the ceiling ({ceiling}). This "
              f"disturbance may still be physically too weak on your robot's "
              f"real mass/geometry — tell me and we'll strengthen the "
              f"mechanism itself, not just the number.")
        return lo, hi, rate_lo, rate_hi

    for _ in range(max_iters):
        mid = (lo + hi) / 2
        rate_mid = probe_fall_rate(mid)
        if rate_mid < target_fall_rate:
            lo = mid
        else:
            hi = mid

    crossover = (lo + hi) / 2
    calibrated_lo = max(0.0, crossover * 0.5)
    calibrated_hi = min(crossover * 1.5, ceiling)
    return calibrated_lo, calibrated_hi, rate_lo, rate_hi


def run_calibration(model_path):
    print(f"{'scenario':<10}{'fn_name':<20}{'new_lo':>10}{'new_hi':>10}")
    results = {}
    for scenario in SCENARIOS:
        scen_id, category, fn_name, nominal_dir = scenario
        calibrated_lo, calibrated_hi, rate_lo, rate_hi = calibrate_magnitude(model_path, scenario)
        results[fn_name] = (calibrated_lo, calibrated_hi)
        print(f"{scen_id:<10}{fn_name:<20}{calibrated_lo:>10.3f}{calibrated_hi:>10.3f}")

    print("\nSuggested MAGNITUDE_RANGES (paste in, replacing the whole dict):")
    for fn_name, (lo, hi) in results.items():
        print(f'    "{fn_name}": ({lo:.3f}, {hi:.3f}),')


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="g1_pendulum.xml")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap number of trials")
    ap.add_argument("--calibrate", action="store_true",
                     help="run binary-search magnitude calibration per scenario and exit")
    args = ap.parse_args()

    if args.calibrate:
        run_calibration(args.model)
        return

    os.makedirs(args.out, exist_ok=True)
    plan = build_trial_plan(args.model, args.out)
    if args.limit:
        plan = plan[:args.limit]

    print(f"Running {len(plan)} trials on {args.workers} workers...")
    t0 = time.time()
    with Pool(args.workers) as pool:
        labels = pool.map(_worker, plan)
    print(f"Done in {time.time() - t0:.1f}s")

    csv_path = os.path.join(args.out, "labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(labels[0]).keys()))
        writer.writeheader()
        for lab in labels:
            writer.writerow(asdict(lab))

    n_fell = sum(l.fell for l in labels)
    n_stable = sum(l.stable for l in labels)
    n_ambiguous = len(labels) - n_fell - n_stable
    print(f"Total trials: {len(labels)} | fell: {n_fell} | stable: {n_stable} | "
          f"ambiguous (timed out): {n_ambiguous}")
    print(f"Labels written to {csv_path}")
    print(f"Per-trial sensor arrays written to {args.out}/*.npz")


if __name__ == "__main__":
    main()
