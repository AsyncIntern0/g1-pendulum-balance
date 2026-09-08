"""
Generates a tiny fake dataset with the SAME labels.csv columns and .npz layout
as generate_fall_dataset_final.py, purely so train_tcn.py can be smoke-tested
end-to-end without the real MuJoCo dataset. Not part of the deliverable.
"""
import csv
import os
import numpy as np

SCENARIO_FNS = [
    "push", "push", "push", "push", "push", "push",
    "floor_tilt_pitch", "floor_tilt_roll", "floor_drop", "low_friction",
    "actuator_fault", "actuator_stuck", "asymmetric_gain", "trip", "sudden_load",
]
FIELDS = ["trial_id","scenario_id","scenario_category","scenario_fn","magnitude_level",
          "magnitude_value","timing_level","timing_phase_s","direction_jitter_level",
          "direction_deg","seed","fell","stable","fall_direction","time_to_ground_contact",
          "peak_pelvis_ang_vel","peak_pelvis_lin_acc","episode_length_s","recoverable_note"]

def make_trial(rng, scen_id, fn_name, fell, out_dir, trial_id):
    dt = 1.0 / 200.0
    timing_phase_s = float(rng.uniform(0.0, 0.8))
    if fell:
        length_s = timing_phase_s + rng.uniform(0.3, 1.2)
    else:
        length_s = timing_phase_s + 1.0 + rng.uniform(0.0, 0.5)
    n = max(8, int(length_s / dt))
    t = np.arange(n) * dt

    gyro = rng.normal(0, 0.05, size=(n, 3))
    acc = rng.normal(0, 0.2, size=(n, 3)) + np.array([0, 0, 9.81])
    qpos = np.zeros((n, 13)); qpos[:, 2] = 0.75
    qvel = rng.normal(0, 0.02, size=(n, 12))

    onset_idx = int(timing_phase_s / dt)
    t_ground_contact = -1.0
    if fell:
        ramp_len = n - onset_idx
        ramp = np.linspace(0, 1, ramp_len) ** 2
        gyro[onset_idx:, :] += ramp[:, None] * rng.normal(0, 3.0, size=(1, 3))
        acc[onset_idx:, :] += ramp[:, None] * rng.normal(0, 4.0, size=(1, 3))
        qpos[onset_idx:, 2] -= ramp * 0.5
        t_ground_contact = (n - 1) * dt - timing_phase_s
    else:
        # small transient that decays -> stable
        decay_len = min(n - onset_idx, int(0.3 / dt))
        if decay_len > 0:
            decay = np.exp(-np.linspace(0, 5, decay_len))
            gyro[onset_idx:onset_idx+decay_len, :] += decay[:, None] * rng.normal(0, 0.8, size=(1, 3))
            acc[onset_idx:onset_idx+decay_len, :] += decay[:, None] * rng.normal(0, 1.0, size=(1, 3))

    np.savez_compressed(os.path.join(out_dir, f"{trial_id}.npz"),
                         time=t, gyro=gyro, acc=acc, qpos=qpos, qvel=qvel)

    return dict(trial_id=trial_id, scenario_id=scen_id, scenario_category="synthetic",
                scenario_fn=fn_name, magnitude_level=0, magnitude_value=1.0,
                timing_level=0, timing_phase_s=timing_phase_s, direction_jitter_level=0,
                direction_deg=-1.0, seed=0, fell=fell, stable=(not fell),
                fall_direction="forward" if fell else "none",
                time_to_ground_contact=t_ground_contact, peak_pelvis_ang_vel=float(np.max(np.linalg.norm(gyro, axis=1))),
                peak_pelvis_lin_acc=float(np.max(np.linalg.norm(acc, axis=1))),
                episode_length_s=(n - 1) * dt, recoverable_note="synthetic")


def main(out_dir="/home/claude/synthetic_dataset", n_per_scenario=20, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    trial_counter = 0
    for scen_id, fn_name in enumerate(SCENARIO_FNS, start=1):
        # scenarios 11 (actuator_fault) and 13 (asymmetric_gain) never fall, matching real data
        base_fall_prob = 0.0 if fn_name in ("actuator_fault", "asymmetric_gain") else 0.5
        for i in range(n_per_scenario):
            fell = bool(rng.random() < base_fall_prob)
            trial_id = f"s{scen_id:02d}_syn{i}"
            rows.append(make_trial(rng, scen_id, fn_name, fell, out_dir, trial_id))
            trial_counter += 1

    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {trial_counter} synthetic trials to {out_dir}")


if __name__ == "__main__":
    main()
