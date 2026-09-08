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
import tempfile
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

# Magnitude ranges keyed PER SCENARIO FUNCTION, not per category — different
# scenarios in the same category can use completely different physical units
# (radians vs meters vs kg vs a 0-1 severity fraction), and sharing one range
# across them silently corrupts calibration for whichever one doesn't match.
# These are starting points; --calibrate will refine them per scenario_fn.
MAGNITUDE_RANGES = {
      "push": (76.992, 230.977),
    "floor_tilt_pitch": (0.051, 0.153),
    "floor_tilt_roll": (0.105, 0.316),
    "floor_drop": (20.000, 400.000),
    "low_friction": (0.300, 0.780),
    "actuator_fault": (0.200, 1.000),
    "actuator_delay": (0.300, 1.000),
    "asymmetric_gain": (0.200, 1.000),
    "trip": (1.000, 20.000),
    "sudden_load": (9.595, 28.784),
}

# Hard search ceilings so --calibrate can't wander into physically nonsensical
# (tilt past ~90deg) or saturated (severity > 1.0) territory.
CALIBRATION_MAX_MAGNITUDE = {
    "push": 400.0,
    "floor_tilt_pitch": 1.0,
    "floor_tilt_roll": 1.0,
    "floor_drop": 400.0,
    "low_friction": 0.78,   # ground friction starts at 0.8; can't reduce below ~0.02
    "actuator_fault": 1.0,
    "actuator_delay": 1.0,
    "asymmetric_gain": 1.0,
    "trip": 20.0,
    "sudden_load": 30.0,
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
    # Clamp to a physically sane tilt range — past 90 deg the gravity-redirect
    # trick flips sign and stops meaning "tilt". Calibration must never probe
    # past this, or the search sees non-monotonic (nonsensical) results.
    magnitude = float(np.clip(magnitude, 0.0, 1.0))  # radians, ~57 deg hard ceiling
    g = 9.81
    if axis == "pitch":
        new_gravity = np.array([-g * np.sin(magnitude), 0.0, -g * np.cos(magnitude)])
    else:  # roll
        new_gravity = np.array([0.0, -g * np.sin(magnitude), -g * np.cos(magnitude)])

    def fn(m, d, t_rel):
        if t_rel < 0.02:  # apply once, sharply, then hold
            m.opt.gravity[:] = new_gravity
    return fn


def make_floor_drop_fn(foot_body_id, magnitude):
    # FIX: originally moved the ENTIRE ground plane down uniformly — both feet
    # lose and regain contact simultaneously and symmetrically, which can
    # never create the asymmetric torque needed to topple. A real "step down"
    # only affects ONE foot. Model it as a brief strong downward pull on one
    # foot body, simulating that foot suddenly finding no support underneath.
    def fn(m, d, t_rel):
        if t_rel < 0.15:
            d.xfrc_applied[foot_body_id, 2] = -magnitude
        else:
            d.xfrc_applied[foot_body_id, 2] = 0.0
    return fn


def make_low_friction_fn(friction_delta):
    # FIX: companion push was a fixed 15N regardless of severity — below the
    # push scenario's own ~40N+ threshold for any effect, so it never mattered
    # whether friction dropped or not. Scale the push with severity so a
    # bigger friction cut is paired with a stronger shear attempt.
    lateral_force = 30.0 + friction_delta * 80.0
    state = {"pelvis_id": None}

    def fn(m, d, t_rel):
        if state["pelvis_id"] is None:
            state["pelvis_id"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if abs(t_rel - 0.0) < 1e-6:
            ground_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
            m.geom_friction[ground_gid, 0] = max(0.02, m.geom_friction[ground_gid, 0] - friction_delta)
        if t_rel < PUSH_DURATION:
            d.xfrc_applied[state["pelvis_id"], 0] = lateral_force
        else:
            d.xfrc_applied[state["pelvis_id"], 0] = 0.0
    return fn


def make_actuator_fault_fn(severity):
    # FIX: originally scaled the commanded ctrl target toward 0 rad, but this
    # rig's standing keyframe already commands ~0 rad for most joints, making
    # that a near no-op regardless of severity. Real "actuator fault" should
    # mean the joint LOSES HOLDING TORQUE, so reduce stiffness (gain), same
    # mechanism as asymmetric_gain, but scaling how many of one leg's joints
    # are affected as well as how completely.
    joint_priority = ["right_ankle_pitch", "right_ankle_roll", "right_knee",
                       "right_hip_pitch", "right_hip_roll"]
    n_fail = max(1, int(round(severity * len(joint_priority))))
    failing = joint_priority[:n_fail]

    def fn(m, d, t_rel):
        if abs(t_rel - 0.0) < 1e-6:
            for name in failing:
                aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                factor = max(0.02, 1.0 - severity)
                m.actuator_gainprm[aid, 0] *= factor
                m.actuator_biasprm[aid, 1] *= factor
    return fn


def make_actuator_delay_fn(severity):
    # severity in [0,1] -> delay across up to 3 leg joints simultaneously,
    # not just one, so the effect can actually accumulate to a fall.
    joints = ["left_knee", "left_ankle_pitch", "left_hip_pitch"]
    delay_steps = max(1, int(severity * 60))  # up to ~120ms at 0.002s timestep
    state = {"ids": None, "queues": None}

    def fn(m, d, t_rel):
        if state["ids"] is None:
            state["ids"] = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in joints]
            state["queues"] = [[] for _ in joints]
        for i, aid in enumerate(state["ids"]):
            state["queues"][i].append(d.ctrl[aid])
            if len(state["queues"][i]) > delay_steps:
                d.ctrl[aid] = state["queues"][i].pop(0)
    return fn


def make_asymmetric_gain_fn(reduce_side, severity):
    # severity in [0,1]: 0 = no change, 1 = that leg's gains near zero.
    factor = max(0.02, 1.0 - severity)

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
    "floor_drop":       lambda m, mag, **kw: make_floor_drop_fn(
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"), mag),
    "low_friction":     lambda m, mag, **kw: make_low_friction_fn(mag),
    "actuator_fault":   lambda m, mag, **kw: make_actuator_fault_fn(np.clip(mag, 0.0, 1.0)),
    "actuator_delay":   lambda m, mag, **kw: make_actuator_delay_fn(np.clip(mag, 0.0, 1.0)),
    "asymmetric_gain":  lambda m, mag, **kw: make_asymmetric_gain_fn("left", np.clip(mag, 0.0, 1.0)),
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
        mag_lo, mag_hi = MAGNITUDE_RANGES[fn_name]
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


def calibrate_magnitude(model_path, scenario, target_fall_rate=0.5, n_probe=12,
                         lo=None, hi=None, max_iters=8, seed_base=9000):
    """
    Binary search on a single scalar magnitude for one scenario until the
    fall rate across n_probe trials (varied timing/seed only) lands near
    target_fall_rate. Returns a calibrated (lo, hi) range to use as the
    scenario's MAGNITUDE_RANGES entry — lo = magnitude where trials rarely
    fall, hi = magnitude where trials almost always fall.
    """
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
              f"{rate_hi*100:.0f}% even at the category ceiling ({ceiling}). "
              f"This disturbance may be physically too weak to topple the robot "
              f"as designed — consider strengthening the disturbance function "
              f"itself rather than raising the range further.")
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

    print("\nSuggested MAGNITUDE_RANGES (per scenario_fn — paste into the script,"
          " replacing the whole dict):")
    for fn_name, (lo, hi) in results.items():
        print(f'    "{fn_name}": ({lo:.3f}, {hi:.3f}),')


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

    # Re-derive magnitude/timing/jitter level indices from trial_id for the CSV
    import csv
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
          f"ambiguous (timed out, neither confirmed): {n_ambiguous}")
    print(f"Labels written to {csv_path}")
    print(f"Per-trial sensor arrays written to {args.out}/*.npz")


if __name__ == "__main__":
    main()
