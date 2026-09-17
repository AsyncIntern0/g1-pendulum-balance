"""
train_tcn.py
============
Phase 2 for AXON-R: TCN-based fall detection + fall-type classification,
trained on the Phase 1 MuJoCo dataset (labels.csv + per-trial .npz).

Two heads on a shared causal, dilated TCN trunk:
  - detection head  : binary, "is a fall imminent within FALL_LEAD_HORIZON_S?"
  - classification head : which of the 15 scenario_fn types, trained ONLY on
                            windows where the detection ground truth is positive
                            (ignore_index=-1 elsewhere)

=====================================================================
KEY DESIGN DECISIONS (read before changing behavior)
=====================================================================
1. Detection label is NOT "will this trial eventually fall". It's a
   per-window label: 1 iff the window's END time falls inside the last
   FALL_LEAD_HORIZON_S seconds before ground contact, for trials that fell.
   This is what makes "detection lead time" a well-defined, measurable
   quantity at eval time instead of an accuracy number with no operational
   meaning. Everything before the horizon (including the disturbance onset
   itself) is labeled negative on purpose -- the model's job is to fire
   inside the window that actually matters for a protective response, not
   to react to the disturbance in general.

2. Trial-level (not window-level) stratified 70/15/15 split by scenario_id.
   Splitting by window would leak highly-correlated frames from the same
   trial across splits and give an optimistic, meaningless val/test score.

3. Normalization stats (per-channel mean/std) are computed from the TRAIN
   split's trials only, then applied to val/test. This mirrors what you'd
   have to do on real hardware (you don't get to peek at test-time data
   before deploying).

4. Default input channels = gyro (3) + acc (3) only -- i.e. IMU-only. This
   is deliberate for Phase 5 (sim-to-real): the pelvis IMU is something the
   real G1 actually has. Use --include-joints to add joint qpos/qvel
   (assumes a standard MuJoCo free-joint base: qpos[0:7]=xyz+quat (dropped,
   not measurable on hardware without a motion-capture rig),
   qpos[7:]=joint angles (kept, these ARE measurable via encoders),
   qvel[0:6]=base twist (dropped), qvel[6:]=joint velocities (kept).
   If g1_pendulum.xml's qpos/qvel layout differs, fix JOINT_QPOS_OFFSET /
   JOINT_QVEL_OFFSET below before using --include-joints.

5. KNOWN GAP from Phase 1 stats: scenarios 11 (actuator_fault) and 13
   (asymmetric_gain) have a 0% fall rate at every magnitude tested. That
   means the classification head will NEVER see a positive training example
   for those two classes -- it structurally cannot learn to predict them,
   and the confusion matrix will show all-zero rows for classes 11 and 13.
   This isn't a Phase 2 bug; it's inherited from Phase 1 calibration
   ("raise max magnitude" in your own report). If you want all 15 classes
   representable, that needs a Phase 1 fix first. This script prints a
   loud warning at load time either way.

Usage:
    python train_tcn.py --data /path/to/dataset --out runs/phase2_v1
    python train_tcn.py --data /path/to/dataset --include-joints --epochs 40
"""
import argparse
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

LOG_HZ = 200
WINDOW_SIZE = 64                 # timesteps -> 320 ms @ 200 Hz
STRIDE = 8                       # timesteps between sampled window end-points -> 40 ms
FALL_LEAD_HORIZON_S = 0.4        # positive-label horizon before ground contact
TIME_JITTER_STEPS = 4            # +/- steps of random time-shift augmentation (train only)
NOISE_STD = 0.05                 # gaussian noise std, applied in NORMALIZED (z-scored) space
SCALE_JITTER = (0.9, 1.1)        # random amplitude scaling range (train only)

# Assumed MuJoCo free-joint layout for --include-joints. Verify against your
# g1_pendulum.xml if the joint-augmented run looks wrong.
JOINT_QPOS_OFFSET = 7
JOINT_QVEL_OFFSET = 6

SCENARIO_FN_ORDER = [
    "push", "floor_tilt_pitch", "floor_tilt_roll", "floor_drop", "low_friction",
    "actuator_fault", "actuator_stuck", "asymmetric_gain", "trip", "sudden_load",
]
NEVER_FALLS_WARNING_SCENARIOS = {"actuator_fault", "asymmetric_gain"}


# ─────────────────────────────────────────────────────────────────
# DATA LOADING, WINDOW INDEXING, SPLIT
# ─────────────────────────────────────────────────────────────────

@dataclass
class TrialArrays:
    trial_id: str
    scenario_id: int
    scenario_fn: str
    fell: bool
    stable: bool
    timing_phase_s: float
    t_impact_abs: float          # absolute time (matches npz 'time' array) of ground contact, or -1
    x: np.ndarray = field(repr=False)   # (T, C) float32, RAW (not yet normalized)
    t: np.ndarray = field(repr=False)   # (T,) float32 absolute time


def build_channel_matrix(npz, include_joints: bool) -> np.ndarray:
    gyro = npz["gyro"]           # (T, 3)
    acc = npz["acc"]             # (T, 3)
    chans = [gyro, acc]
    if include_joints:
        qpos = npz["qpos"][:, JOINT_QPOS_OFFSET:]
        qvel = npz["qvel"][:, JOINT_QVEL_OFFSET:]
        chans += [qpos, qvel]
    return np.concatenate(chans, axis=1).astype(np.float32)


def load_dataset(data_dir: str, include_joints: bool):
    labels_path = os.path.join(data_dir, "labels.csv")
    df = pd.read_csv(labels_path)

    n_ambiguous = int(((~df["fell"]) & (~df["stable"])).sum())
    if n_ambiguous:
        print(f"[WARN] {n_ambiguous} trials are neither fell nor stable "
              f"(timed out). Excluding them from training -- label is undefined.")
        df = df[df["fell"] | df["stable"]].reset_index(drop=True)

    never_fall = (df.groupby("scenario_fn")["fell"].mean() == 0)
    flagged = [s for s in never_fall[never_fall].index if s in NEVER_FALLS_WARNING_SCENARIOS]
    if flagged:
        print(f"[WARN] scenario(s) {flagged} have 0% fall rate in this dataset. "
              f"The classification head will never see a positive example for "
              f"{'these classes' if len(flagged) > 1 else 'this class'} and cannot "
              f"learn to predict {'them' if len(flagged) > 1 else 'it'}. This is "
              f"inherited from Phase 1 calibration, not a Phase 2 bug.")

    trials = []
    for row in df.itertuples():
        npz_path = os.path.join(data_dir, f"{row.trial_id}.npz")
        with np.load(npz_path) as npz:
            x = build_channel_matrix(npz, include_joints)
            t = npz["time"].astype(np.float32)
        t_impact_abs = row.timing_phase_s + row.time_to_ground_contact if row.fell else -1.0
        trials.append(TrialArrays(
            trial_id=row.trial_id, scenario_id=int(row.scenario_id),
            scenario_fn=row.scenario_fn, fell=bool(row.fell), stable=bool(row.stable),
            timing_phase_s=float(row.timing_phase_s), t_impact_abs=float(t_impact_abs),
            x=x, t=t,
        ))
    return trials, df


def stratified_trial_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=0):
    """Trial-level split, stratified by scenario_id. Returns 3 sets of trial_id."""
    from sklearn.model_selection import train_test_split
    trial_ids = df["trial_id"].values
    strata = df["scenario_id"].values

    train_val_ids, test_ids, train_val_strata, _ = train_test_split(
        trial_ids, strata, test_size=test_frac, stratify=strata, random_state=seed)

    val_relative = val_frac / (1.0 - test_frac)
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_relative, stratify=train_val_strata, random_state=seed)

    return set(train_ids), set(val_ids), set(test_ids)


def compute_norm_stats(trials, train_ids):
    all_x = np.concatenate([tr.x for tr in trials if tr.trial_id in train_ids], axis=0)
    mean = all_x.mean(axis=0)
    std = all_x.std(axis=0)
    std[std < 1e-6] = 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# WINDOW SAMPLE INDEX + LABELING
# ─────────────────────────────────────────────────────────────────

@dataclass
class WindowSample:
    trial_idx: int
    end_idx: int          # index into trial.t / trial.x of the window's last timestep
    detect_label: int     # 0/1
    class_label: int      # scenario_id if detect_label==1 else -1 (ignored in loss)


def label_window(trial: TrialArrays, end_idx: int) -> int:
    if not trial.fell:
        return 0
    t_end = trial.t[end_idx]
    return int((trial.t_impact_abs - FALL_LEAD_HORIZON_S) <= t_end < trial.t_impact_abs)


def build_window_index(trials, split_ids, stride=STRIDE):
    samples = []
    for trial_idx, trial in enumerate(trials):
        if trial.trial_id not in split_ids:
            continue
        T = trial.x.shape[0]
        if T < WINDOW_SIZE:
            continue
        for end_idx in range(WINDOW_SIZE - 1, T, stride):
            dlabel = label_window(trial, end_idx)
            clabel = trial.scenario_id if dlabel == 1 else -1
            samples.append(WindowSample(trial_idx, end_idx, dlabel, clabel))
    return samples


# ─────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────

class FallWindowDataset(Dataset):
    def __init__(self, trials, samples, mean, std, augment: bool):
        self.trials = trials
        self.samples = samples
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        trial = self.trials[s.trial_idx]
        end_idx = s.end_idx

        if self.augment and TIME_JITTER_STEPS > 0:
            lo = max(WINDOW_SIZE - 1, end_idx - TIME_JITTER_STEPS)
            hi = min(trial.x.shape[0] - 1, end_idx + TIME_JITTER_STEPS)
            end_idx = np.random.randint(lo, hi + 1)

        start_idx = end_idx - WINDOW_SIZE + 1
        window = trial.x[start_idx:end_idx + 1]                      # (WINDOW_SIZE, C)
        window = (window - self.mean) / self.std

        if self.augment:
            window = window + np.random.normal(0, NOISE_STD, window.shape).astype(np.float32)
            scale = np.random.uniform(*SCALE_JITTER)
            window = window * scale

        x = torch.from_numpy(window.T.astype(np.float32))            # (C, WINDOW_SIZE)
        return x, s.detect_label, s.class_label


# ─────────────────────────────────────────────────────────────────
# MODEL: causal dilated TCN, two heads
# ─────────────────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = F.pad(x, (self.pad, 0))
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.dropout(out)
        out = F.pad(out, (self.pad, 0))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class FallTCN(nn.Module):
    def __init__(self, in_channels, num_classes, channels=(32, 32, 64, 64, 64),
                 kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(prev_ch, ch, kernel_size, dilation=2 ** i, dropout=dropout))
            prev_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.detect_head = nn.Linear(prev_ch, 1)
        self.class_head = nn.Linear(prev_ch, num_classes)

    def forward(self, x):
        # x: (B, C_in, T)
        feats = self.tcn(x)              # (B, C_hidden, T)
        last = feats[:, :, -1]           # causal "now" summary -- no future leakage
        detect_logit = self.detect_head(last).squeeze(-1)   # (B,)
        class_logits = self.class_head(last)                # (B, num_classes)
        return detect_logit, class_logits

    def receptive_field(self, kernel_size=3, channels=(32, 32, 64, 64, 64)):
        rf = 1
        for i in range(len(channels)):
            rf += 2 * (kernel_size - 1) * (2 ** i)
        return rf


# ─────────────────────────────────────────────────────────────────
# METRICS (incl. detection lead time)
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def window_level_metrics(model, loader, device, threshold=0.5):
    model.eval()
    all_probs, all_dlabels, all_cpreds, all_clabels = [], [], [], []
    for x, dlabel, clabel in loader:
        x = x.to(device)
        dlogit, clogit = model(x)
        probs = torch.sigmoid(dlogit).cpu().numpy()
        all_probs.append(probs)
        all_dlabels.append(dlabel.numpy())
        all_cpreds.append(clogit.argmax(dim=1).cpu().numpy())
        all_clabels.append(clabel.numpy())
    probs = np.concatenate(all_probs)
    dlabels = np.concatenate(all_dlabels)
    cpreds = np.concatenate(all_cpreds)
    clabels = np.concatenate(all_clabels)

    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (dlabels == 1)).sum())
    tn = int(((preds == 0) & (dlabels == 0)).sum())
    fp = int(((preds == 1) & (dlabels == 0)).sum())
    fn = int(((preds == 0) & (dlabels == 1)).sum())

    detect_acc = (tp + tn) / max(1, len(dlabels))
    fpr = fp / max(1, fp + tn)
    recall = tp / max(1, tp + fn)

    pos_mask = clabels != -1
    class_acc = float((cpreds[pos_mask] == clabels[pos_mask]).mean()) if pos_mask.sum() > 0 else float("nan")

    return {
        "detect_acc": detect_acc, "fpr": fpr, "recall": recall,
        "class_acc": class_acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "probs": probs, "dlabels": dlabels,
    }


def find_operating_threshold(probs, labels, target_fpr=0.01):
    """Smallest threshold (best recall) that keeps window-level FPR <= target_fpr
    on a held-out (validation) set. Falls back to the threshold with the lowest
    achievable FPR if the target can't be met."""
    neg_mask = labels == 0
    pos_mask = labels == 1
    if neg_mask.sum() == 0 or pos_mask.sum() == 0:
        return 0.5
    thresholds = np.linspace(0.0, 1.0, 201)
    best_thresh, best_recall, best_fpr_at_target = 0.5, -1, None
    min_fpr, min_fpr_thresh = 1.0, 1.0
    for th in thresholds:
        fp = ((probs >= th) & neg_mask).sum()
        fpr = fp / neg_mask.sum()
        if fpr < min_fpr:
            min_fpr, min_fpr_thresh = fpr, th
        if fpr <= target_fpr:
            tp = ((probs >= th) & pos_mask).sum()
            recall = tp / pos_mask.sum()
            if recall > best_recall:
                best_recall, best_thresh = recall, th
    if best_recall < 0:
        print(f"[WARN] no threshold achieves target FPR {target_fpr}; "
              f"using min achievable FPR {min_fpr:.4f} at threshold {min_fpr_thresh:.3f}")
        return float(min_fpr_thresh)
    return float(best_thresh)


@torch.no_grad()
def lead_time_eval(model, trials, split_ids, mean, std, device, threshold, stride=1):
    """Sequential, whole-trial inference. For fall trials: first crossing of
    `threshold` -> lead time in ms before ground contact (or a miss). For
    stable trials: whether the model ever raises a false alarm (per-trial
    false-alarm rate, distinct from the window-level FPR above)."""
    model.eval()
    lead_times_ms = []
    n_fall_trials, n_detected = 0, 0
    n_stable_trials, n_false_alarms = 0, 0

    for trial in trials:
        if trial.trial_id not in split_ids:
            continue
        T = trial.x.shape[0]
        if T < WINDOW_SIZE:
            continue
        x_norm = (trial.x - mean) / std

        end_indices = np.arange(WINDOW_SIZE - 1, T, stride)
        windows = np.stack([x_norm[e - WINDOW_SIZE + 1:e + 1] for e in end_indices])  # (N, W, C)
        xb = torch.from_numpy(windows.transpose(0, 2, 1).astype(np.float32)).to(device)
        dlogit, _ = model(xb)
        probs = torch.sigmoid(dlogit).cpu().numpy()
        fired = probs >= threshold

        if trial.fell:
            n_fall_trials += 1
            fire_positions = np.where(fired)[0]
            # only count firings strictly before impact as a valid detection
            valid = [i for i in fire_positions if trial.t[end_indices[i]] < trial.t_impact_abs]
            if valid:
                n_detected += 1
                t_detect = trial.t[end_indices[valid[0]]]
                lead_times_ms.append((trial.t_impact_abs - t_detect) * 1000.0)
        elif trial.stable:
            n_stable_trials += 1
            if fired.any():
                n_false_alarms += 1

    return {
        "n_fall_trials": n_fall_trials,
        "detected_recall": n_detected / max(1, n_fall_trials),
        "mean_lead_ms": float(np.mean(lead_times_ms)) if lead_times_ms else float("nan"),
        "median_lead_ms": float(np.median(lead_times_ms)) if lead_times_ms else float("nan"),
        "n_stable_trials": n_stable_trials,
        "per_trial_false_alarm_rate": n_false_alarms / max(1, n_stable_trials),
    }


# ─────────────────────────────────────────────────────────────────
# TRAIN LOOP
# ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    trials, df = load_dataset(args.data, include_joints=args.include_joints)
    train_ids, val_ids, test_ids = stratified_trial_split(df, seed=args.seed)
    print(f"Trials: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    mean, std = compute_norm_stats(trials, train_ids)

    train_samples = build_window_index(trials, train_ids)
    val_samples = build_window_index(trials, val_ids)
    test_samples = build_window_index(trials, test_ids)
    n_pos = sum(s.detect_label for s in train_samples)
    n_neg = len(train_samples) - n_pos
    print(f"Train windows: {len(train_samples)} (pos={n_pos}, neg={n_neg}, "
          f"pos_rate={n_pos / max(1, len(train_samples)):.4f})")
    if n_pos == 0:
        raise RuntimeError("No positive (imminent-fall) windows in the training split. "
                            "Check FALL_LEAD_HORIZON_S / STRIDE against your data's typical "
                            "episode lengths, or check that fell=True trials exist in train_ids.")

    train_ds = FallWindowDataset(trials, train_samples, mean, std, augment=True)
    val_ds = FallWindowDataset(trials, val_samples, mean, std, augment=False)
    test_ds = FallWindowDataset(trials, test_samples, mean, std, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    in_channels = trials[0].x.shape[1]
    num_classes = int(df["scenario_id"].max()) + 1   # index by scenario_id directly (0 unused)
    model = FallTCN(in_channels=in_channels, num_classes=num_classes).to(device)
    print(f"Input channels: {in_channels} | receptive field: {model.receptive_field()} "
          f"timesteps ({model.receptive_field() / LOG_HZ * 1000:.0f} ms) vs window {WINDOW_SIZE} "
          f"({WINDOW_SIZE / LOG_HZ * 1000:.0f} ms)")

    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)
    detect_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    class_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    best_val_score, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for x, dlabel, clabel in train_loader:
            x = x.to(device)
            dlabel = dlabel.float().to(device)
            clabel = clabel.long().to(device)

            dlogit, clogit = model(x)
            loss_d = detect_loss_fn(dlogit, dlabel)
            # CrossEntropyLoss(ignore_index=-1) returns NaN (0/0) when a batch
            # happens to contain zero valid (non-ignored) targets -- routine given
            # the ~15% positive rate, so guard it explicitly rather than let a NaN
            # scalar pollute the logged loss every time it happens.
            has_valid_class_target = (clabel != -1).any()
            if has_valid_class_target:
                loss_c = class_loss_fn(clogit, clabel)
            else:
                loss_c = torch.zeros((), device=device)
            loss = loss_d + args.class_loss_weight * loss_c

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_ds)
        val_metrics = window_level_metrics(model, val_loader, device, threshold=0.5)
        # score used for checkpoint selection: recall at a fixed, tight FPR budget
        # matters more here than raw accuracy given the heavy class imbalance
        score = val_metrics["recall"] - 5.0 * val_metrics["fpr"]
        print(f"[epoch {epoch:03d}] loss={train_loss:.4f} "
              f"val_acc={val_metrics['detect_acc']:.3f} val_recall={val_metrics['recall']:.3f} "
              f"val_fpr={val_metrics['fpr']:.4f} val_class_acc={val_metrics['class_acc']:.3f} "
              f"({time.time() - t0:.1f}s)")

        if score > best_val_score:
            best_val_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.out, "model_best.pt"))

    # pick an operating threshold on VAL (never on test), then report test metrics at it
    val_metrics_for_thresh = window_level_metrics(model, val_loader, device, threshold=0.5)
    threshold = find_operating_threshold(
        val_metrics_for_thresh["probs"], val_metrics_for_thresh["dlabels"],
        target_fpr=args.target_fpr)
    print(f"\nOperating threshold selected on val set (target FPR <= {args.target_fpr}): {threshold:.3f}")

    test_metrics = window_level_metrics(model, test_loader, device, threshold=threshold)
    lead_metrics = lead_time_eval(model, trials, test_ids, mean, std, device, threshold)

    report = {
        "threshold": threshold,
        "window_level": {k: v for k, v in test_metrics.items() if k not in ("probs", "dlabels")},
        "lead_time": lead_metrics,
        "in_channels": in_channels,
        "num_train_trials": len(train_ids),
        "num_val_trials": len(val_ids),
        "num_test_trials": len(test_ids),
    }
    print("\n=== TEST SET REPORT ===")
    print(json.dumps(report, indent=2))
    with open(os.path.join(args.out, "test_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    np.savez(os.path.join(args.out, "norm_stats.npz"), mean=mean, std=std)
    return report


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir with labels.csv + *.npz")
    ap.add_argument("--out", default="runs/phase2", help="where to write checkpoints/report")
    ap.add_argument("--include-joints", action="store_true",
                     help="add joint qpos/qvel channels (see JOINT_*_OFFSET assumptions)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--class-loss-weight", type=float, default=0.5)
    ap.add_argument("--target-fpr", type=float, default=0.01,
                     help="window-level FPR budget used to pick the deployed threshold")
    ap.add_argument("--seed", type=int, default=0)
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
