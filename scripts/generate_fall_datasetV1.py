"""
generate_fall_dataset.py
=========================
Phase 1 dataset generation for AXON-R fall detection/classification.

Uses g1_pendulum.xml with NO active recovery policy — the position
actuators are held fixed at the 'stand' keyframe ctrl values for the
whole episode. This is a passive PD/impedance controller (kp=300 legs,
kp=800 pendulum), not zero control, so small disturbances naturally
recover ("stable") and large ones naturally overwhelm it ("fall").
This is what gives you a physically honest stable/fall split without
needing a trained recovery policy first (see design note in chat).

Episodes are VARIABLE LENGTH (not a fixed 2s window):
  - stop early once "fall" is confirmed (new non-foot geom contacts ground)
  - stop early once "stable" is confirmed (pelvis height/tilt/ang-vel stay
    inside a band for a continuous window)
  - hard cap at MAX_EPISODE_TIME as a safety net only

Run:
    python generate_fall_dataset.py --model g1_pendulum.xml --out dataset/ --workers 8
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from multiprocessing import Pool

import numpy as np
import mujoco


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

SETTLE_TIME = 0.5          # s, let the robot settle on the keyframe before disturbing
MAX_EPISODE_TIME = 4.0     # s, hard cap safety net (post-settle)
STABLE_WINDOW = 1.0        # s, continuous time required to confirm "stable"
LOG_HZ = 200                # Hz, resample sensor stream to a realistic IMU rate
STABLE_PELVIS_Z_MIN = 0.55  # m, below this pelvis height counts toward instability
STABLE_ANGVEL_MAX = 2.0     # rad/s, gyro magnitude above this counts toward instability

# 15 scenarios: (id, category, apply_fn_name, nominal_direction_deg_or_None)
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
    (12, "actuator",   "actuator_delay",   None),
    (13, "actuator",   "asymmetric_gain",  None),
    (14, "trip",       "trip",             None),
    (15, "trip",       "sudden_load",      None),
]

MAGNITUDE_LEVELS = 5   # per-scenario range calibrated below
TIMING_LEVELS = 5      # phase of gait cycle at which disturbance is applied
DIRECTION_JITTER_LEVELS = 4  # +/-15deg, +/-30deg style jitter around nominal direction
N_VARIATIONS = MAGNITUDE_LEVELS * TIMING_LEVELS * DIRECTION_JITTER_LEVELS  # = 100

# Per-scenario-category magnitude ranges (calibrate these against your own robot mass;
# these are starting points — see calibration note at bottom of file).
MAGNITUDE_RANGES = {
    "push":     (20.0, 100.0),    # Newtons, applied to pelvis for PUSH_DURATION
    "surface":  (0.05, 0.35),     # tilt radians / friction delta / drop height(m) depending on subtype
    "actuator": (0.2, 1.0),       # fault severity fraction (0=no fault, 1=full dropout)
    "trip":     (0.5, 3.0),       # trip: joint vel arrest strength; load: kg suddenly added
}

PUSH_DURATION = 0.10  # s, how long an external push force is applied


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
# One function per scenario type. Each returns a callable(model, data, t_rel, mag)
# that mutates data in-place for the current step. t_rel = time since disturbance onset.
# ─────────────────────────────────────────────────────────────────

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
    # Simulate a tilted support surface by rotating the *effective* gravity
    # vector in the sagittal (pitch) or frontal (roll) plane. This is a
    # common trick to fake ground tilt without remeshing the floor geom.
    g = 9.81
    if axis == "pitch":
        new_gravity = np.array([-g * np.sin(magnitude), 0.0, -g * np.cos(magnitude)])
    else:  # roll
        new_gravity = np.array([0.0, -g * np.sin(magnitude), -g * np.cos(magnitude)])

    def fn(m, d, t_rel):
        if t_rel < 0.02:  # apply once, sharply, then hold
            m.opt.gravity[:] = new_gravity
    return fn


def make_floor_drop_fn(drop_height):
    # Approximate a step-down by translating the ground plane down abruptly
    # under the robot. Simpler alternative: lower one foot's target height
    # is not directly controllable via position actuators here, so we shift
    # the ground geom pose instead.
    def fn(m, d, t_rel):
        if abs(t_rel - 0.0) < 1e-6:
            ground_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
            m.geom_pos[ground_gid, 2] -= drop_height
    return fn


def make_low_friction_fn(friction_delta):
    def fn(m, d, t_rel):
        if abs(t_rel - 0.0) < 1e-6:
            ground_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
            m.geom_friction[ground_gid, 0] = max(0.02, m.geom_friction[ground_gid, 0] - friction_delta)
    return fn


def make_actuator_fault_fn(actuator_name, severity):
    act_id = None  # resolved lazily inside fn using model, cached on first call
    state = {"id": None, "orig": None}

    def fn(m, d, t_rel):
        if state["id"] is None:
            state["id"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            state["orig"] = d.ctrl[state["id"]]
        if t_rel >= 0.0:
            d.ctrl[state["id"]] = state["orig"] * (1.0 - severity)
    return fn


def make_actuator_delay_fn(actuator_name, delay_steps):
    buf = {"queue": [], "id": None}

    def fn(m, d, t_rel):
        if buf["id"] is None:
            buf["id"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        buf["queue"].append(d.ctrl[buf["id"]])
        if len(buf["queue"]) > delay_steps:
            d.ctrl[buf["id"]] = buf["queue"].pop(0)
    return fn


def make_asymmetric_gain_fn(reduce_side, factor):
    # NOTE: MuJoCo position actuator gains (kp) are compiled into the model;
    # true online kp changes need actuator_gainprm mutation, which IS writable
    # on the mjModel copy each worker owns, so this is safe per-process.
    def fn(m, d, t_rel):
        if abs(t_rel - 0.0) < 1e-6:
            prefix = "left" if reduce_side == "left" else "right"
            for i in range(m.nu):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if name and name.startswith(prefix):
                    m.actuator_gainprm[i, 0] *= factor
                    m.actuator_biasprm[i, 1] *= factor
    return fn


def make_trip_fn(joint_name, arrest_strength):
    def fn(m, d, t_rel):
        if t_rel < 0.05:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            dof = m.jnt_dofadr[jid]
            d.qvel[dof] -= arrest_strength * np.sign(d.qvel[dof] + 1e-6)
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
    "floor_drop":       lambda m, mag, **kw: make_floor_drop_fn(mag),
    "low_friction":     lambda m, mag, **kw: make_low_friction_fn(mag),
    "actuator_fault":   lambda m, mag, **kw: make_actuator_fault_fn("right_ankle_pitch", min(mag, 1.0)),
    "actuator_delay":   lambda m, mag, **kw: make_actuator_delay_fn("left_knee", int(mag * 20)),
    "asymmetric_gain":  lambda m, mag, **kw: make_asymmetric_gain_fn("left", max(0.05, 1.0 - mag)),
    "trip":             lambda m, mag, **kw: make_trip_fn("left_knee_joint", mag),
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
    """Baseline non-foot contacts already touching ground at end of settle —
    needed so a naturally-grounded geom doesn't get mislabeled as a 'fall'."""
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
    rng = np.random.default_rng(seed)

    mujoco.mj_resetDataKeyframe(m, d, 0)
    stand_ctrl = m.key_ctrl[0].copy()
    d.ctrl[:] = stand_ctrl  # passive PD hold — no active recovery policy

    ground_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    foot_ids = get_foot_geom_ids(m)

    imu_gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu-pelvis-angular-velocity")
    imu_acc_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu-pelvis-linear-acceleration")
    gyro_adr = m.sensor_adr[imu_gyro_id]
    acc_adr = m.sensor_adr[imu_acc_id]

    dt = m.opt.timestep
    log_every = max(1, int(round(1.0 / (LOG_HZ * dt))))

    # ---- Settle phase (no logging, no disturbance) ----
    settle_steps = int(SETTLE_TIME / dt)
    for _ in range(settle_steps):
        d.ctrl[:] = stand_ctrl
        mujoco.mj_step(m, d)
    baseline_contacts = snapshot_contacts(m, d, ground_id)

    # ---- Build disturbance function for this scenario ----
    builder = SCENARIO_BUILDERS[fn_name]
    disturb_fn = builder(m, magnitude, direction=direction_deg)

    # ---- Rollout with variable-length termination ----
    max_steps = int(MAX_EPISODE_TIME / dt)
    timing_steps = int(timing_phase_s / dt)  # delay disturbance onset within episode
    stable_window_steps = int(STABLE_WINDOW / dt)

    log_time, log_gyro, log_acc, log_qpos, log_qvel = [], [], [], [], []
    fell, stable = False, False
    t_ground_contact = None
    stable_counter = 0
    peak_angvel, peak_acc = 0.0, 0.0
    step = 0

    while step < max_steps:
        d.ctrl[:] = stand_ctrl  # keep holding the stand pose throughout (passive baseline)
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

        # --- fall check: NEW non-foot / non-baseline contact with ground ---
        current_contacts = snapshot_contacts(m, d, ground_id)
        new_bad_contacts = (current_contacts - baseline_contacts) - foot_ids
        if new_bad_contacts and not fell:
            fell = True
            t_ground_contact = step * dt - timing_phase_s
            break  # stop episode as soon as fall is confirmed

        # --- stability check: only start counting after disturbance onset ---
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
        dx = d.qpos[0]
        dy = d.qpos[1]
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
        trial_id=trial_id,
        scenario_id=scen_id,
        scenario_category=category,
        scenario_fn=fn_name,
        magnitude_level=-1,  # filled by caller
        magnitude_value=float(magnitude),
        timing_level=-1,
        timing_phase_s=float(timing_phase_s),
        direction_jitter_level=-1,
        direction_deg=float(direction_deg) if direction_deg is not None else -1.0,
        seed=seed,
        fell=fell,
        stable=stable,
        fall_direction=fall_direction,
        time_to_ground_contact=float(t_ground_contact) if t_ground_contact else -1.0,
        peak_pelvis_ang_vel=peak_angvel,
        peak_pelvis_lin_acc=peak_acc,
        episode_length_s=step * dt,
        recoverable_note="passive_pd_only",
    )

    # ---- Save raw sensor + state arrays for this trial ----
    npz_path = os.path.join(log_dir, f"{trial_id}.npz")
    np.savez_compressed(
        npz_path,
        time=np.array(log_time),
        gyro=np.array(log_gyro),
        acc=np.array(log_acc),
        qpos=np.array(log_qpos),
        qvel=np.array(log_qvel),
    )
    return label


def _worker(args):
    return run_trial(*args)


# ─────────────────────────────────────────────────────────────────
# TRIAL PLAN — builds the full factorial 15 x 100 = 1500 trial list
# ─────────────────────────────────────────────────────────────────

def build_trial_plan(model_path, log_dir, base_seed=0):
    plan = []
    trial_counter = 0
    for scenario in SCENARIOS:
        scen_id, category, fn_name, nominal_dir = scenario
        mag_lo, mag_hi = MAGNITUDE_RANGES[category]
        magnitudes = np.linspace(mag_lo, mag_hi, MAGNITUDE_LEVELS)
        # timing phase: 5 points across an assumed ~1.0s double/single support cycle placeholder
        timings = np.linspace(0.0, 0.8, TIMING_LEVELS)
        jitters = [-30, -15, 15, 30]

        for mi, mag in enumerate(magnitudes):
            for ti, timing in enumerate(timings):
                for di, jitter in enumerate(jitters):
                    direction = None
                    if nominal_dir is not None:
                        direction = (nominal_dir + jitter) % 360
                    seed = base_seed + trial_counter
                    trial_id = f"s{scen_id:02d}_m{mi}_t{ti}_d{di}_seed{seed}"
                    plan.append((model_path, scenario, mag, direction, timing, seed,
                                 trial_id, log_dir))
                    trial_counter += 1
    assert len(plan) == len(SCENARIOS) * N_VARIATIONS, \
        f"expected {len(SCENARIOS) * N_VARIATIONS}, got {len(plan)}"
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="g1_pendulum.xml")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap number of trials")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    plan = build_trial_plan(args.model, args.out)
    if args.limit:
        plan = plan[:args.limit]

    print(f"Running {len(plan)} trials on {args.workers} workers...")
    t0 = time.time()
    with Pool(args.workers) as pool:
        labels = pool.map(_worker, plan)
    print(f"Done in {time.time() - t0:.1f}s")

    # Re-derive magnitude/timing/jitter level indices from trial_id for the CSV
    import csv
    csv_path = os.path.join(args.out, "labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(labels[0]).keys()))
        writer.writeheader()
        for lab in labels:
            writer.writerow(asdict(lab))

    n_fell = sum(l.fell for l in labels)
    print(f"Total trials: {len(labels)} | fell: {n_fell} | stable: {len(labels) - n_fell}")
    print(f"Labels written to {csv_path}")
    print(f"Per-trial sensor arrays written to {args.out}/*.npz")


if __name__ == "__main__":
    main()
