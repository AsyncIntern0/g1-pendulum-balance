"""
phase3_pose_design.py
======================
AXON-R Phase 3 -- protective pose LOOKUP TABLE via biomechanical analysis in
MuJoCo, designed to plug directly into the mentor's Stage 3/4 spec:

    Stage 3 (elsewhere -- TCN regression head): Direction class, Cause class,
             continuous fall-velocity estimate (0-5 m/s).
    Stage 4 (THIS FILE's output + select_pose() below): rule-based lookup
             table + interpolation, keyed on (Direction, Cause, Velocity) ->
             target joint angles for every actuated joint. No inference at
             deploy time -- deterministic and auditable for the regulatory
             path, per the mentor's explicit requirement.

WHAT CHANGED FROM THE PREVIOUS VERSION (single-magnitude-per-scenario)
------------------------------------------------------------------------
The previous script optimized exactly one pose per scenario at a single
fixed magnitude (0.85 quantile of Phase 1's MAGNITUDE_RANGES) and had no
velocity axis anywhere -- every fall got the same pose regardless of how
fast it was happening, which does not satisfy "Output: Target joint angles
... Input: ... Velocity estimate" from the spec.

This version samples EACH scenario at several magnitude quantiles
(VELOCITY_QUANTILES below), runs a full CMA-ES optimization per bin, and
MEASURES the actual resulting fall velocity for each bin (base linear
velocity magnitude, ||qvel[0:3]||, sampled at the instant the pose is
triggered -- see run_protected_trial). Those (velocity, pose) pairs become
control points; select_pose() at the bottom of this file does the lookup +
linear interpolation Stage 4 needs, and has NO mujoco/cma dependency, so
Phase 4 (or a real-time controller) can import just that function.

WHAT THIS SCRIPT STILL ASSUMES ABOUT THE MODEL (read this first)
------------------------------------------------------------------
g1_pendulum.xml has NO arms and NO head. The entire upper body is a 2-axis
reaction-mass pendulum (a 5kg "bob" 0.3m above the pelvis, collision-disabled
by default). There is therefore no way to measure a "hand" impact, and no
real arm-catch protective pose is physically representable on this rig.
This pipeline uses two impact proxies instead of the three the spec assumed:
  - PELVIS : the existing pelvis collision geom (already contype/conaffinity
             enabled in the XML) -> stands in for hip/torso impact.
  - HEAD   : the pendulum bob, with contact ENABLED AT RUNTIME (in Python,
             not by editing the XML) -> stands in for head/upper-body impact.
No XML edits are needed or shipped; g1_pendulum.xml is used as-is.

TWO OPEN QUESTIONS FOR YOUR MENTOR -- flag these explicitly, don't guess:
  1. The spec says "Output: Target joint angles for all 6 actuated joints."
     This rig has 14 actuated joints (2 pendulum + 12 leg). Confirm whether
     "6" refers to a different/simplified joint set the mentor has in mind,
     or whether the spec text predates the current rig. This script outputs
     angles for every actuator on the loaded model (model.nu), whatever that
     number is -- it does not silently truncate or pad to 6.
  2. Phase 1 scenarios carry both a `category` (e.g. "push", "trip",
     "actuator_fault") and a `fn_name` (a more specific mechanism name).
     It is not yet confirmed which granularity the Phase 2 TCN's "Cause
     class" output actually corresponds to. This script's lookup table is
     keyed on BOTH so Phase 4 can match on whichever one the TCN emits --
     but this doubles-up ambiguity should be resolved with your mentor
     before Phase 5, not carried silently into hardware.

PIPELINE
--------
1. For each Phase-1 scenario and each velocity quantile in
   VELOCITY_QUANTILES, run the UNPROTECTED baseline at that quantile's
   magnitude to find natural time-to-ground-contact (skip the bin if the
   robot never falls in the observation window at that magnitude -- some
   low-severity magnitudes legitimately never produce a fall, which is fine;
   there's nothing to protect against in that case).
2. Optimize a joint-angle pose per (scenario, bin) with CMA-ES (gradient-free
   -- contact/impact events are discontinuous and non-differentiable) that
   minimizes weighted peak impact force on the pelvis+head proxies when
   commanded at (t_impact - LEAD_TIME_S) via a minimum-jerk trajectory (NOT
   a direct step command -- see minimum_jerk_trajectory and the reference
   doc's section 8.2) over POSE_TRANSITION_BUDGET_S, subject to joint
   limits and a tracking-error penalty (see TRACKING_ERROR_TOL_RAD).
   Records the measured base velocity at trigger time for that bin.
3. Cross-evaluate every scenario's MOST SEVERE (highest-velocity) pose
   against every OTHER scenario's most severe conditions, to measure
   robustness to Phase 2's ~25% fall-type misclassification rate.
4. Greedily merge scenarios whose most-severe poses are interchangeable with
   little performance loss, compressing raw per-scenario poses down into a
   final library of TARGET_LIBRARY_MIN-TARGET_LIBRARY_MAX poses. Each merged
   pose keeps ALL of its representative scenario's velocity bins (not just
   the reference one used for the merge decision).
5. Write pose_library.json: each pose entry carries a `velocity_control_points`
   list (sorted ascending by measured velocity) plus a flat `lookup_keys`
   table mapping every covered (category / fn_name, direction) to its
   pose_id, so Phase 4 has an O(1) lookup surface.
6. select_pose() (bottom of file, pure Python + numpy, no mujoco/cma) does
   the actual Stage-4 rule-based lookup + linear interpolation between the
   two bracketing velocity control points -- this is what Phase 4 and any
   real-time controller should import and call.

Run:
    python phase3_pose_design.py --model g1_pendulum.xml --out phase3_out/
    python phase3_pose_design.py --model g1_pendulum.xml --quick   # smoke test
    python phase3_pose_design.py --selftest                        # no mujoco needed

Requires: mujoco, numpy, cma  (pip install mujoco cma numpy)
Must be run from a directory where `generate_fall_dataset_final.py` (Phase 1)
is importable -- this script reuses its scenario mechanisms directly so
Phase 3 falls are generated by the exact same physics as Phase 1/2.
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field, asdict

import numpy as np

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

LEAD_TIME_S = 0.30              # conservative trigger-to-impact budget to design
                                 # against (worst realistic case). Phase 2 achieved
                                 # median 396ms / mean 380ms -- see confirm_margin().
POSE_TRANSITION_BUDGET_S = 0.30 # mentor's spec AND the reference doc's own
                                 # T_transition: pose must land within 300ms. This
                                 # is now literally the minimum-jerk trajectory's
                                 # duration T (see minimum_jerk_trajectory below),
                                 # not just a deadline checked after the fact --
                                 # the commanded path reaches pose_ctrl at exactly
                                 # t=T by construction, every time.

W_HEAD = 2.0                    # weight on pendulum-bob (head-proxy) peak force
W_PELVIS = 1.0                  # weight on pelvis peak force
W_OTHER = 0.5                   # weight on any other non-foot body hitting the
                                 # ground (knee, hip-roll, etc). Lower than
                                 # pelvis/head since absorbing a fall through
                                 # the hip/thigh is genuinely less dangerous --
                                 # but NOT zero, so a pose can't escape scoring
                                 # entirely just by landing somewhere untracked.
TRACKING_ERROR_TOL_RAD = 0.15   # How far the REAL joints may lag the commanded
                                 # minimum-jerk path before it counts as a
                                 # meaningful tracking failure rather than
                                 # ordinary servo lag under load (~8.6 deg --
                                 # looser than the old fixed-pose arrival
                                 # tolerance of 0.05 rad, deliberately: a MOVING
                                 # target under an active disturbance will
                                 # legitimately lag some; this tolerance is for
                                 # catching genuinely large deviations, i.e. the
                                 # "enormous joint torques" case the reference
                                 # doc's 8.2 warns about, not small expected lag).
TRACKING_ERROR_PENALTY_PER_RAD = 3000.0   # N-equivalent penalty per radian of peak
                                 # tracking error beyond TRACKING_ERROR_TOL_RAD.
                                 # Applied to EVERY triggered trial, recovered or
                                 # not -- unlike the ground-impact force terms
                                 # (correctly 0 when there's no contact), a large
                                 # tracking error is itself a real, separate
                                 # injury channel (actuator torque spikes /
                                 # secondary joint injury from failing to follow
                                 # the planned smooth path) that can happen
                                 # regardless of whether the fall is ultimately
                                 # avoided -- see score_pose.
FALL_OCCURRENCE_PENALTY = 1500.0           # Flat cost added whenever a trial falls at
                                            # all, on top of the weighted force terms.
                                            # This is what makes "prevent the fall"
                                            # dominate over "reduce force a bit among
                                            # falls that still happen" -- without it,
                                            # CMA-ES only ever sees force-magnitude
                                            # gradients, so a pose that trims a couple
                                            # hundred N off every trial can look just as
                                            # good as one that fully prevents some of
                                            # them. Set well below a catastrophic
                                            # tracking-error penalty so a wildly
                                            # untrackable-AND-consequential fall still
                                            # scores worse than a well-tracked one, and
                                            # well above typical single-trial force costs
                                            # (~500-2000 in early runs) so that going
                                            # from "falls with moderate force" to "falls
                                            # with near-zero force" is NOT enough to beat
                                            # "doesn't fall" -- prevention is meant to
                                            # dominate, not just add another force term.
                                            # Tune this directly if your team wants a
                                            # different prevention-vs-impact trade-off.
                                            # DEFINED BUT NO LONGER APPLIED IN SCORING.
                                            # A prior revision added this to the score,
                                            # then found it was a mistake: it mixes an
                                            # arbitrary, unitless constant into an
                                            # otherwise physically meaningful (Newton)
                                            # score, and it isn't what the reference doc
                                            # asks for -- the doc's own optimizer
                                            # (section 8.3) scores ONLY mean peak contact
                                            # force, nothing else. It also wasn't
                                            # necessary: a fully prevented fall already
                                            # scores 0 on every force term, which is
                                            # already the best any peak-force sum can
                                            # produce, so CMA-ES was already maximally
                                            # incentivized to find recovery wherever it's
                                            # reachable without this term's help. See
                                            # score_pose for the corrected objective;
                                            # this constant is kept, unused, only so the
                                            # reasoning for NOT using it stays on record.

# Phase 1 stops a trial at the FIRST new ground contact (correct for
# detection labeling). For impact-force measurement that first contact is
# often a knee/hip touching down before the pelvis or head-proxy ever does
# -- so we deliberately keep simulating a short window past that first
# contact to actually capture the pelvis/head impact, instead of reading a
# false 0 N. Empirically (bob-height trace on push/forward): the pendulum
# bob doesn't reach the ground until ~1.3s post-disturbance, so 0.45s was
# far too short -- raised to 1.5s. This makes every trial ~3x more
# expensive; if a full run is too slow, profile per-scenario before cutting
# this back down rather than reverting it blindly.
POST_FALL_OBSERVATION_S = 1.5

# Conditions sampled per (scenario, velocity bin) during optimization (kept
# small for speed; widen for a final run). timing/jitter values are a subset
# of Phase 1's own grid so the sim is exercised the same way Phase 1
# characterized it.
COND_TIMINGS = [0.0, 0.4]
COND_JITTERS = [-15, 15]

# --- Velocity-bin design (new) --------------------------------------------
# Each scenario is sampled at these magnitude quantiles (fraction of the way
# from Phase 1's MAGNITUDE_RANGES lo->hi for that mechanism). Each quantile
# that produces a natural fall becomes one control point in that scenario's
# velocity -> pose curve, tagged with the MEASURED base-velocity at trigger
# time (not the quantile itself -- the quantile is just how we reach a given
# severity; the velocity actually observed is what Stage 4 keys off of).
# 3 bins roughly triples the per-scenario optimization cost vs. a single
# magnitude; drop to 2 (e.g. [0.6, 0.85]) if a full run is too slow.
VELOCITY_QUANTILES = [0.5, 0.7, 0.85]

# Which bin is used for the cross-scenario merge decision (step 3/4 above).
# "max" = each scenario's highest-velocity (most severe) successfully
# optimized bin -- conservative, matches the existing worst-case design
# philosophy (LEAD_TIME_S is already the worst-case budget, not the median).
MERGE_REFERENCE = "max"   # one of: "max", "min"

MERGE_TOLERANCE = 0.15   # allow <=15% score degradation when reusing one
                          # scenario's pose for another, to permit a merge
TARGET_LIBRARY_MIN = 12
TARGET_LIBRARY_MAX = 15


# ─────────────────────────────────────────────────────────────────
# MINIMUM-JERK TRAJECTORY (reference doc section 8.2)
# ─────────────────────────────────────────────────────────────────

def minimum_jerk_trajectory(t, T, q_start, q_target):
    """Smooth position trajectory from q_start to q_target over duration T
    (Flash & Hogan 1985, "The coordination of arm movements", J. Neurosci.
    5(7):1688-1703 -- as specified in the reference doc's section 8.2).

    Commanding a position actuator straight to q_target in one step makes it
    compute (target-current)*kp instantly -- a torque spike that is a real
    secondary-injury risk on hardware, which is exactly why the doc calls
    that pattern out by name ("data.ctrl = target_pose # DO NOT DO THIS").
    This is the replacement: run_protected_trial calls this every step from
    the trigger instant onward and commands its output instead of pose_ctrl
    directly.

    t: time since trigger (s). T: transition duration (POSE_TRANSITION_
    BUDGET_S). q_start/q_target: (n_act,) arrays. tau is clamped to [0,1],
    so this is safe to call unconditionally for the whole post-trigger
    window, including steps at or after t>=T (it just holds q_target
    exactly from then on) -- no separate "transition finished" branch
    needed anywhere that calls it."""
    tau = min(t / T, 1.0) if T > 0 else 1.0
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    return q_start + (q_target - q_start) * s


# ─────────────────────────────────────────────────────────────────
# MODEL INSTRUMENTATION (done at runtime, not via XML edits)
# ─────────────────────────────────────────────────────────────────

def load_instrumented_model(model_path, passive_pendulum=False):
    """Load g1_pendulum.xml unmodified, then enable contact on the pendulum
    bob geom(s) in Python so it can register ground impact as a head proxy.

    FAILS LOUDLY if "pelvis" or "pendulum_bob" don't resolve to a real body
    id. A previous run silently proceeded when a body name didn't match
    (mj_name2id returns -1, not an exception), which meant contact never got
    enabled AND, separately, any force that *did* land on that body would
    get mis-bucketed into the "other" channel in impact_body_ids() /
    contact_peak_forces() instead of "head" -- a silent, hard-to-diagnose
    failure mode that produced confusing intermittent-looking data. Fail
    fast instead.

    `passive_pendulum` (default False -- NOT silently changed on you):
    when True, zeroes the pendulum's actuator_gainprm/biasprm so its two
    actuators output exactly 0 N*m always, leaving the torso governed only
    by gravity + native joint damping (damping=2.0/joint) + its +-45deg
    range limit, instead of the XML's kp=800 position-hold. Default False
    because generate_fall_dataset_final.py deliberately defines the
    kp=800-hold-at-'stand' behavior as THE natural/unprotected baseline
    ("passive PD/impedance hold ... not zero control ... honest,
    non-fabricated" -- its own docstring) -- that's an already-reviewed
    Phase 1 decision, not something to override by default from Phase 3.
    Pass True explicitly to A/B it against that baseline; see the
    'actuator_scope' note in main()'s JSON output for why you might want
    to."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(model_path)
    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    bob_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pendulum_bob")
    if pelvis_id < 0 or bob_id < 0:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        raise RuntimeError(
            "load_instrumented_model: could not resolve required body name(s) "
            f"-- pelvis_id={pelvis_id}, pendulum_bob_id={bob_id}. Body names in "
            f"this model: {names}. Fix the name lookup above before trusting "
            "any downstream force attribution (a -1 id silently mis-buckets "
            "forces into the 'other' channel instead of raising)."
        )
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] == bob_id:
            m.geom_contype[gid] = 1
            m.geom_conaffinity[gid] = 1

    if passive_pendulum:
        # See actuator_group_ids() for the full history. v6 pinned
        # pose_ctrl's pendulum entries at 'stand' to stop CMA-ES from
        # steering the mass around, but pinning the CTRL TARGET does not
        # remove the actuator's TORQUE -- an 800-kp servo rigidly fighting
        # to hold 0 rad through a fall makes the torso behave like a stiff
        # mast bolted to the pelvis, not a floppy trunk. That's the new
        # failure v6 uncovered: forces that used to vanish (old exploit)
        # are now dumped entirely onto the head-proxy contact (measured
        # 3-6.7kN in pose_library_v6.json, vs ~0N pelvis/other in the same
        # trials) -- consistent with a rigid rod slamming down independent
        # of leg posture, not a person falling.
        #
        # Runtime-zeroing gainprm/biasprm (NOT an XML edit -- same
        # "instrument in Python" convention as the contact-enable above)
        # makes both pendulum actuators apply exactly 0 N*m regardless of
        # ctrl. The pendulum then swings under gravity, its own native
        # joint damping (damping=2.0/joint, untouched) and its hard
        # +-45deg range limit ONLY.
        #
        # MUST happen before the caller's snapshot_model_state(): both
        # actuator_gainprm and actuator_biasprm are already restored from
        # that snapshot before every trial (Phase 1's actuator_fault/
        # asymmetric_gain mutate them in place), so zeroing here -- before
        # load_instrumented_model returns -- makes "passive pendulum" the
        # restored baseline for every trial in BOTH conditions. This
        # changes what "unprotected" means too: a rigid-mast torso was
        # never a real "no protection" baseline either, so an A/B force
        # comparison built on it would compare two flavors of the same
        # artifact rather than a real fall vs. a protected one.
        pend_ids, _ = actuator_group_ids(m)
        m.actuator_gainprm[pend_ids, :] = 0.0
        m.actuator_biasprm[pend_ids, :] = 0.0
    return m


# Several Phase 1 disturbance mechanisms mutate the MODEL itself, not just
# per-episode data: actuator_fault/asymmetric_gain multiply
# actuator_gainprm/biasprm IN PLACE every time they fire; floor_tilt_* sets
# opt.gravity; low_friction sets geom_friction. Phase 1 got away with this
# because it loads a brand-new MjModel per trial. Phase 3 reuses ONE model
# instance across thousands of trials for speed, so without an explicit
# reset these mutations leak and compound across every later trial and
# scenario. Snapshot the mutable fields once, restore them before every trial.
_MUTABLE_MODEL_FIELDS = ["actuator_gainprm", "actuator_biasprm", "geom_friction",
                          "geom_contype", "geom_conaffinity"]


def snapshot_model_state(model):
    snap = {name: getattr(model, name).copy() for name in _MUTABLE_MODEL_FIELDS}
    snap["gravity"] = model.opt.gravity.copy()
    return snap


def restore_model_state(model, snapshot):
    for name in _MUTABLE_MODEL_FIELDS:
        getattr(model, name)[:] = snapshot[name]
    model.opt.gravity[:] = snapshot["gravity"]


def impact_body_ids(model):
    import mujoco
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    bob_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pendulum_bob")
    if pelvis_id < 0 or bob_id < 0:
        raise RuntimeError(
            f"impact_body_ids: unresolved body id(s) pelvis={pelvis_id} "
            f"head={bob_id} -- see load_instrumented_model for why this must "
            "raise rather than silently mis-bucket forces."
        )
    return {"pelvis": pelvis_id, "head": bob_id}


def contact_peak_forces(model, data, ground_id, body_ids, foot_geom_ids):
    """Peak resultant contact force this step, per tracked body (pelvis,
    head), counting ONLY contacts against the ground geom (environmental
    impact) -- contacts between the robot's own links are ignored so
    folded/self-contact doesn't get mistaken for an impact.

    Also tracks `other`: peak force from any NON-FOOT body contacting the
    ground that isn't pelvis or head (e.g. a knee or hip-roll link). Normal
    standing foot-ground contact is excluded via `foot_geom_ids`. This
    channel exists specifically so a pose that dumps the fall onto an
    untracked body part reads as a mitigated-but-real impact, not a free 0
    -- see the reward-hacking case found in actuator_fault."""
    import mujoco
    peaks = {k: 0.0 for k in body_ids}
    peaks["other"] = 0.0
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = con.geom1, con.geom2
        if ground_id not in (g1, g2):
            continue
        other_g = g2 if g1 == ground_id else g1
        if other_g in foot_geom_ids:
            continue
        other_body = model.geom_bodyid[other_g]
        force6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, c, force6)
        mag = float(np.linalg.norm(force6[:3]))
        matched = False
        for name, bid in body_ids.items():
            if other_body == bid:
                if mag > peaks[name]:
                    peaks[name] = mag
                matched = True
        if not matched and mag > peaks["other"]:
            peaks["other"] = mag
    return peaks


# ─────────────────────────────────────────────────────────────────
# MAGNITUDE / VELOCITY-BIN HELPERS
# ─────────────────────────────────────────────────────────────────

def fall_magnitude(fn_name, quantile, p1):
    """Magnitude at the given quantile of Phase 1's calibrated range for this
    mechanism. `quantile` is explicit now (previously hardcoded 0.85) so the
    same helper can generate every velocity bin, not just one severity."""
    lo, hi = p1.MAGNITUDE_RANGES[fn_name]
    return lo + quantile * (hi - lo)


def build_conditions(scenario, magnitude):
    """`magnitude` is now passed in explicitly (previously computed inside
    this function via a single hardcoded quantile) so the same scenario can
    be exercised at several severities for the velocity-bin design."""
    scen_id, category, fn_name, nominal_dir = scenario
    conds = []
    for timing in COND_TIMINGS:
        for jitter in COND_JITTERS:
            direction = ((nominal_dir + jitter) % 360) if nominal_dir is not None else None
            conds.append((magnitude, direction, timing))
    return conds


# ─────────────────────────────────────────────────────────────────
# PROTECTED ROLLOUT (the thing we optimize)
# ─────────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    peak_force_pelvis: float
    peak_force_head: float
    peak_force_other: float           # any non-foot, non-pelvis, non-head body
                                       # hitting the ground -- e.g. a knee/hip
    peak_tracking_error_rad: float    # max over all joints and all steps during the
                                       # transition window [trigger, trigger+POSE_
                                       # TRANSITION_BUDGET_S] of |actual qpos - the
                                       # COMMANDED minimum-jerk trajectory position at
                                       # that instant| (not vs. the final target -- vs.
                                       # where the smooth path itself says the joint
                                       # should be right now). Large values mean the
                                       # disturbance is overpowering the controller's
                                       # ability to follow the planned smooth path,
                                       # which is itself an injury-risk channel per the
                                       # reference doc's minimum-jerk rationale
                                       # (uncontrolled joint deviation -> torque spikes),
                                       # independent of whether the fall is prevented.
                                       # 0.0 if never triggered.
    final_tracking_error_rad: float   # |actual qpos - pose_ctrl| at the end of the
                                       # transition window (how close the REAL robot,
                                       # not the reference trajectory, ended up to the
                                       # true target once the smooth ramp finished) --
                                       # reporting/diagnostic only, not scored directly.
                                       # -1.0 if never triggered.
    time_to_impact_s: float           # -1.0 if the UNPROTECTED baseline never
                                       # fell within the observation window --
                                       # in that case the pose is never
                                       # triggered at all (see below)
    fell: bool
    no_natural_fall: bool             # True if baseline never falls unprotected
                                       # -> pose was not applied; excluded from
                                       # meaningful optimization signal
    triggered: bool                   # True iff the pose was actually commanded at
                                       # some point during THIS trial. False covers
                                       # TWO distinct cases that must both be excluded
                                       # from scoring, not penalized: (1) no_natural_fall
                                       # (trigger_t_rel = inf, pose never meant to fire),
                                       # and (2) a subtler one -- _find_time_to_impact
                                       # (used only to time the trigger) has no stability
                                       # check, so it can find a fall further out in time
                                       # than run_protected_trial's OWN pre-trigger loop
                                       # ever reaches, because that loop's stability
                                       # early-exit (same STABLE_WINDOW logic Phase 1
                                       # uses) can fire first during a slow topple that
                                       # transiently looks stable. The trigger point
                                       # calculated from pass 1 is then never reached in
                                       # pass 2, so the pose is never tested under that
                                       # condition at all. See score_pose -- this was
                                       # previously mis-scored as a failed transition.
    recovered: bool                   # True if the pose fired (no_natural_fall is
                                       # False) but the robot never actually fell
                                       # (fell stayed False) -- a full save, not
                                       # just a softened impact. Zero injury risk
                                       # by definition (no ground contact ever
                                       # registered, so every force term is
                                       # naturally 0) -- but the tracking-error
                                       # penalty still applies regardless of this
                                       # flag (it's a separate, joint/actuator-level
                                       # injury channel that can occur even when
                                       # the fall itself is avoided). Reported per
                                       # pose as recovery_rate; does not otherwise
                                       # affect scoring -- see score_pose.
    velocity_at_trigger_mps: float    # ||base linear velocity|| (qvel[0:3]) at
                                       # the instant the pose is first commanded
                                       # -- the measured proxy for Phase 2's
                                       # Stage-3 velocity-regression output.
                                       # -1.0 if the pose never triggered.


def run_protected_trial(model, scenario, magnitude, direction_deg, timing_phase_s,
                         pose_ctrl, trigger_lead_s, impact_ids, snapshot, p1,contact_tracker=None):
    """Re-run the same disturbance mechanism as Phase 1, but from
    (t_impact_estimate - trigger_lead_s) onward, command `pose_ctrl` instead
    of holding `stand`. `model` must already be mj.MjModel instrumented via
    load_instrumented_model(); a fresh MjData is created per trial. `snapshot`
    (from snapshot_model_state) is restored first so mutations left over from
    a PREVIOUS trial can't leak in."""
    import mujoco
    restore_model_state(model, snapshot)
    scen_id, category, fn_name, nominal_dir = scenario
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    stand_ctrl = model.key_ctrl[0].copy()
    d.ctrl[:] = stand_ctrl

    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    foot_ids = p1.get_foot_geom_ids(model)

    dt = model.opt.timestep
    settle_steps = int(p1.SETTLE_TIME / dt)
    for _ in range(settle_steps):
        d.ctrl[:] = stand_ctrl
        mujoco.mj_step(model, d)
    baseline_contacts = p1.snapshot_contacts(model, d, ground_id)

    # First pass (no pose) to find this condition's natural time-to-impact,
    # so we know when to trigger. Restore pristine state again right after,
    # before the real stepping loop below runs the SAME mechanism for real.
    t_impact = _find_time_to_impact(model, scenario, magnitude, direction_deg,
                                     timing_phase_s, snapshot, p1)
    restore_model_state(model, snapshot)
    # If the UNPROTECTED baseline never falls within the observation window,
    # there is no real fall to protect against for this condition -- send
    # the trigger to "never" so the pose is NOT applied at all, rather than
    # falling back to an immediate trigger (which previously let an
    # aggressive pose self-collapse onto an untracked body part for a free
    # 0 score -- see peak_force_other / no_natural_fall history).
    no_natural_fall = t_impact is None
    trigger_t_rel = max(0.0, t_impact - trigger_lead_s) if not no_natural_fall else float("inf")

    builder = p1.SCENARIO_BUILDERS[fn_name]
    disturb_fn = builder(model, magnitude, direction=direction_deg)

    max_steps = int(p1.MAX_EPISODE_TIME / dt)
    timing_steps = int(timing_phase_s / dt)
    stable_window_steps = int(p1.STABLE_WINDOW / dt)

    peak = {"pelvis": 0.0, "head": 0.0, "other": 0.0}
    fell, stable = False, False
    stable_counter = 0
    step = 0
    pose_ctrl = np.asarray(pose_ctrl)
    n_act = pose_ctrl.shape[0]
    peak_tracking_error_rad = 0.0
    final_tracking_error_rad = -1.0
    q_start = None
    trigger_fired = False
    trigger_step = None
    fell_step = None
    velocity_at_trigger_mps = -1.0

    while step < max_steps:
        t_rel = (step - timing_steps) * dt
        if step >= timing_steps and max(t_rel, 0.0) >= trigger_t_rel:
            if not trigger_fired:
                trigger_fired = True
                trigger_step = step
                # Base linear velocity at the moment the pose is commanded --
                # the measured proxy for the Stage-3 velocity estimate this
                # pose's control point is filed under.
                velocity_at_trigger_mps = float(np.linalg.norm(d.qvel[0:3]))
                # ACTUAL current joint angles, not the commanded 'stand'
                # target -- per the reference doc's own
                # q_start = get_current_joint_angles(data). The disturbance
                # may already have pushed real qpos away from whatever was
                # commanded before trigger, and the minimum-jerk trajectory
                # must start from where the robot actually is.
                q_start = d.qpos[7:7 + n_act].copy()
            # Minimum-jerk trajectory (Flash & Hogan 1985), per the reference
            # doc's 8.2: NEVER command the target angle directly in one step
            # -- that produces an instantaneous (target-current)*kp torque
            # spike, a real secondary-injury risk on hardware and exactly the
            # anti-pattern the doc calls out by name. self-clamps at t>=T
            # (tau capped at 1.0), so this is safe to call unconditionally
            # for the entire post-trigger window with no separate branching
            # for "transition finished" needed.
            t_since_trigger = (step - trigger_step) * dt
            q_cmd = minimum_jerk_trajectory(t_since_trigger, POSE_TRANSITION_BUDGET_S,
                                             q_start, pose_ctrl)
            d.ctrl[:n_act] = q_cmd
        else:
            d.ctrl[:] = stand_ctrl

        if step >= timing_steps:
            disturb_fn(model, d, max(t_rel, 0.0))  # may override some ctrl entries

        mujoco.mj_step(model, d)
        step += 1

        forces = contact_peak_forces(model, d, ground_id, impact_ids, foot_ids)
        if contact_tracker is not None:
            contact_tracker.update(d)
        peak["pelvis"] = max(peak["pelvis"], forces["pelvis"])
        peak["head"] = max(peak["head"], forces["head"])
        peak["other"] = max(peak["other"], forces["other"])

        if trigger_fired:
            t_since_trigger = (step - trigger_step) * dt
            if t_since_trigger <= POSE_TRANSITION_BUDGET_S + 1e-9:
                # Tracking error vs. the COMMANDED trajectory at this instant
                # (not vs. the final target) -- this is what the doc's
                # "enormous joint torques" risk actually looks like: the real
                # joints falling far behind the smooth reference path because
                # the disturbance is overpowering the controller, not the
                # ordinary/expected small following lag of a well-behaved
                # position servo. Both peak (worst moment) and final (state
                # at the nominal end of the transition) are tracked; only
                # peak is scored -- see score_pose -- final is diagnostic.
                q_cmd_now = minimum_jerk_trajectory(t_since_trigger, POSE_TRANSITION_BUDGET_S,
                                                     q_start, pose_ctrl)
                cur_err = float(np.max(np.abs(d.qpos[7:7 + n_act] - q_cmd_now)))
                peak_tracking_error_rad = max(peak_tracking_error_rad, cur_err)
                final_tracking_error_rad = float(np.max(np.abs(d.qpos[7:7 + n_act] - pose_ctrl)))

        current_contacts = p1.snapshot_contacts(model, d, ground_id)
        new_bad_contacts = (current_contacts - baseline_contacts) - foot_ids
        if new_bad_contacts and not fell:
            fell = True
            fell_step = step
        if fell and (step - fell_step) * dt >= POST_FALL_OBSERVATION_S:
            break

        if step >= timing_steps:
            pelvis_z = d.qpos[2]
            gyro_norm = np.linalg.norm(d.qvel[3:6])
            stable_now = (pelvis_z > p1.STABLE_PELVIS_Z_MIN and gyro_norm < p1.STABLE_ANGVEL_MAX)
            stable_counter = stable_counter + 1 if stable_now else 0
            if stable_counter >= stable_window_steps:
                stable = True
                break

    recovered = trigger_fired and (not fell) and (not no_natural_fall)

    return TrialResult(
        peak_force_pelvis=peak["pelvis"], peak_force_head=peak["head"],
        peak_force_other=peak["other"],
        peak_tracking_error_rad=peak_tracking_error_rad,
        final_tracking_error_rad=final_tracking_error_rad,
        time_to_impact_s=t_impact if t_impact is not None else -1.0, fell=fell,
        no_natural_fall=no_natural_fall, triggered=trigger_fired, recovered=recovered,
        velocity_at_trigger_mps=velocity_at_trigger_mps,
    )


def _find_time_to_impact(model, scenario, magnitude, direction_deg, timing_phase_s, snapshot, p1):
    """One quick unprotected pass (fresh MjData, same model) to get this
    condition's natural time_to_ground_contact, used only to time the trigger."""
    import mujoco
    restore_model_state(model, snapshot)
    scen_id, category, fn_name, nominal_dir = scenario
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    stand_ctrl = model.key_ctrl[0].copy()
    d.ctrl[:] = stand_ctrl
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    foot_ids = p1.get_foot_geom_ids(model)
    dt = model.opt.timestep
    settle_steps = int(p1.SETTLE_TIME / dt)
    for _ in range(settle_steps):
        d.ctrl[:] = stand_ctrl
        mujoco.mj_step(model, d)
    baseline_contacts = p1.snapshot_contacts(model, d, ground_id)
    builder = p1.SCENARIO_BUILDERS[fn_name]
    disturb_fn = builder(model, magnitude, direction=direction_deg)
    max_steps = int(p1.MAX_EPISODE_TIME / dt)
    timing_steps = int(timing_phase_s / dt)
    step = 0
    while step < max_steps:
        d.ctrl[:] = stand_ctrl
        t_rel = (step - timing_steps) * dt
        if step >= timing_steps:
            disturb_fn(model, d, max(t_rel, 0.0))
        mujoco.mj_step(model, d)
        step += 1
        current_contacts = p1.snapshot_contacts(model, d, ground_id)
        new_bad = (current_contacts - baseline_contacts) - foot_ids
        if new_bad:
            return step * dt - timing_phase_s
    return None  # never fell in this pass -- caller should raise magnitude


# ─────────────────────────────────────────────────────────────────
# MULTIPROCESSING (CPU-parallel; there is no GPU path for plain MuJoCo).
# Every candidate pose within a single CMA-ES generation is scored
# completely independently, so we hand popsize candidates to a worker pool.
# Each worker loads its OWN model instance (via _worker_init) rather than
# pickling a shared MjModel -- also naturally immune to the model-mutation
# contamination bug, since separate processes can't share mutable state.
# ─────────────────────────────────────────────────────────────────

_worker_model = None
_worker_snapshot = None
_worker_p1 = None


def _worker_init(model_path, p1_module_name, passive_pendulum=False):
    global _worker_model, _worker_snapshot, _worker_p1
    import importlib
    _worker_p1 = importlib.import_module(p1_module_name)
    _worker_model = load_instrumented_model(model_path, passive_pendulum=passive_pendulum)
    _worker_snapshot = snapshot_model_state(_worker_model)


def _worker_score_task(task):
    scenario, pose_ctrl, conditions, lead_time_s = task
    impact_ids = impact_body_ids(_worker_model)
    return score_pose(_worker_model, scenario, pose_ctrl, conditions, impact_ids,
                       lead_time_s, _worker_snapshot, _worker_p1)


# ─────────────────────────────────────────────────────────────────
# OPTIMIZATION (CMA-ES, gradient-free)
# ─────────────────────────────────────────────────────────────────

def actuator_bounds(model):
    lo = model.actuator_ctrlrange[:, 0].copy()
    hi = model.actuator_ctrlrange[:, 1].copy()
    return lo, hi


def actuator_group_ids(model):
    """Split model.nu actuators into 'pendulum' (2: pendulum_roll,
    pendulum_pitch) and 'leg' (the remaining 12) by NAME, not by hardcoded
    index -- fail loudly if either pendulum actuator can't be resolved,
    same philosophy as load_instrumented_model.

    WHY THIS EXISTS (read before removing): g1_pendulum.xml gives the
    pendulum its own 2 position actuators with a deliberately high kp
    ("COM shifting" -- see the XML's own comments) because the rig started
    life as a self-balancing-robot reaction-mass model. But the system
    AXON-R actually ships on is a LOWER-LIMB-ONLY exoskeleton for paralyzed
    rehab patients: the hardware has no actuator on the wearer's own
    torso/head at all. Letting CMA-ES freely command those 2 actuators
    (as every version through v5 did -- see 'stand = model.key_ctrl[0]'
    being the full 14-dim search vector in optimize_pose_for_bin) means the
    optimizer can "solve" head/pelvis impact by swinging the pendulum mass
    out of the way -- exactly the way a person would lift their own head to
    dodge a fall. A paralyzed wearer cannot do that, and the exoskeleton has
    no way to do it for them. Confirmed against real MuJoCo-viewer footage:
    protected trials were visibly repositioning the pendulum bob rather
    than changing leg posture. This is a control-authority modeling error,
    not a scoring-weight problem -- fixing W_HEAD/W_OTHER cannot fix it,
    because the exploit is in which actuators the optimizer is allowed to
    touch, not in how their effects are weighted.
    """
    import mujoco
    names = ["pendulum_roll", "pendulum_pitch"]
    pend_ids = []
    for n in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if aid < 0:
            all_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                         for i in range(model.nu)]
            raise RuntimeError(
                f"actuator_group_ids: could not resolve actuator '{n}'. "
                f"Actuator names in this model: {all_names}. Fix the name "
                "lookup before trusting the leg/pendulum split below -- a "
                "silent miss here would let a pendulum actuator slip back "
                "into the 'leg' (CMA-ES-controlled) group unnoticed."
            )
        pend_ids.append(aid)
    leg_ids = [i for i in range(model.nu) if i not in pend_ids]
    return np.array(pend_ids, dtype=int), np.array(leg_ids, dtype=int)


def score_pose(model, scenario, pose_ctrl, conditions, impact_ids, lead_time_s, snapshot, p1):
    total = 0.0
    counted = 0
    diagnostics = []
    for mag, direction, timing in conditions:
        r = run_protected_trial(model, scenario, mag, direction, timing,
                                 pose_ctrl, lead_time_s, impact_ids, snapshot, p1)
        diagnostics.append(r)
        if r.no_natural_fall or not r.triggered:
            # Excluded from the average rather than defaulted to 0 -- see the
            # reward-hacking history in the module docstring for the
            # no_natural_fall case. `not r.triggered` (with no_natural_fall
            # False) is the second, subtler case: _find_time_to_impact found
            # a fall further out in time than run_protected_trial's own
            # pre-trigger stability check ever let this trial run to -- the
            # pose was never actually commanded, so there is nothing to
            # score here, and it must not be treated as a failed transition
            # (see TrialResult.triggered for the full explanation).
            continue
        s = (W_HEAD * r.peak_force_head + W_PELVIS * r.peak_force_pelvis
             + W_OTHER * r.peak_force_other)
        # Tracking-error penalty: a SEPARATE injury channel from ground
        # impact (joint/actuator torque spikes from failing to follow the
        # planned minimum-jerk path under the disturbance's load) that can
        # occur whether or not the fall itself is prevented -- applies to
        # every triggered trial, recovered included, unlike the
        # ground-impact force terms above (which are correctly 0 exactly
        # when no contact happened). This term IS in the reference doc's
        # spirit (section 8.2's whole point is avoiding injury from a bad
        # transition), unlike a fall/no-fall penalty -- see below.
        tracking_overrun = max(0.0, r.peak_tracking_error_rad - TRACKING_ERROR_TOL_RAD)
        s += TRACKING_ERROR_PENALTY_PER_RAD * tracking_overrun
        # Deliberately NO fall-occurrence penalty here. The reference doc's
        # own optimizer (section 8.3) scores ONLY mean peak contact force --
        # "Evaluate peak contact force for given pose parameters. Lower is
        # better." Every principle in the doc (lower CoM, align impact zone,
        # minimum-jerk transition) is about reducing injury from a fall that
        # is already happening, not about preventing the fall outright. A
        # prior version of this scorer added a flat FALL_OCCURRENCE_PENALTY
        # on top of these force terms to push CMA-ES toward preventing falls
        # -- which is not what the doc asks for, and mixes an arbitrary,
        # unitless constant into an otherwise physically meaningful
        # Newton-valued score, making scores from different runs impossible
        # to compare honestly. It also wasn't necessary: a fully prevented
        # fall already scores 0 here (every force term is exactly 0 when no
        # contact occurs), which is already the best any peak-force sum can
        # produce -- CMA-ES is already maximally incentivized to find full
        # recovery whenever one is reachable, with no extra term required.
        # recovery_rate (below) is still computed and reported per pose --
        # it's valuable information for judging hardware readiness -- it
        # just does not drive the score.
        total += s
        counted += 1
    if counted == 0:
        # Nothing for this pose search to optimize at this magnitude --
        # surface loudly (NaN) rather than a deceptively perfect 0.
        return float("nan"), diagnostics,None
    recovered_count = sum(1 for r in diagnostics if r.recovered)
    recovery_rate = recovered_count / counted
    return total / counted, diagnostics, recovery_rate


RESTART_SIGMA_GROWTH = 1.7   # each successive restart's sigma0 multiplied by this --
                              # a cheap IPOP-style heuristic: if earlier, tighter
                              # restarts converge to a failing local optimum, later
                              # restarts search more broadly rather than repeating
                              # the same narrow search around a new random point.
RESTART_START_SPREAD_BASE = 0.5   # stddev (rad) of the random perturbation from
                                  # 'stand' used to pick restart>0's starting pose;
                                  # grows with restart index (see below)


def optimize_pose_for_bin(model, scenario, magnitude, lead_time_s, popsize, maxiter,
                           snapshot, p1, seed=0, pool=None, n_restarts=1):
    """Optimize one pose for one (scenario, magnitude) velocity bin.
    Returns a dict including the MEASURED velocity_mps for this bin (mean
    ||base linear velocity|| at trigger time across the best pose's
    successful conditions) -- this becomes one control point in that
    scenario's velocity -> pose interpolation curve.

    `pool`: an optional multiprocessing.Pool. When given, every candidate in
    a CMA-ES generation is scored in parallel across worker processes.

    `n_restarts`: run this many INDEPENDENT CMA-ES searches and keep the
    global best, rather than a single search from 'stand'. This exists
    because a single local search can converge cleanly to a real local
    force-minimum that still falls every time, without ever sampling into a
    qualitatively different, possibly fall-preventing region of pose space
    (e.g. a wide crouch reachable only by a path that looks worse than
    'stand' along the way, not by descending the force gradient smoothly).
    Restart 0 always starts at 'stand' (matches prior single-run behavior
    exactly when n_restarts=1); restarts >0 start from a randomized
    perturbation of 'stand' with growing spread and sigma0, so later
    restarts explore more broadly if earlier ones plateau on the same kind
    of failing optimum. This multiplies wall-clock cost by roughly
    n_restarts (restarts run sequentially; each restart's own population is
    still parallelized across `pool` as before)."""
    import cma
    scen_id, category, fn_name, nominal_dir = scenario
    lo_full, hi_full = actuator_bounds(model)
    stand = model.key_ctrl[0].copy()
    pend_ids, leg_ids = actuator_group_ids(model)
    lo, hi = lo_full[leg_ids], hi_full[leg_ids]

    def expand(leg_x):
        """leg_x: (len(leg_ids),) CMA-ES candidate -> full (model.nu,) pose_ctrl
        with the pendulum actuators PINNED at their 'stand' target. This is
        the only place a leg-only candidate becomes a full pose_ctrl -- see
        actuator_group_ids for why the pendulum must never be part of the
        search space. Pinning at 'stand' (rather than e.g. 0) keeps the
        pendulum at its designed neutral/balance point instead of an
        arbitrary one; it does NOT make the pendulum torque-free/passive
        (the XML's high-kp position actuator will still resist being
        pushed off that target by the disturbance) -- that would require an
        XML-level gain change, flagged separately, not a search-space fix."""
        full = stand.copy()
        full[leg_ids] = leg_x
        return full

    conditions = build_conditions(scenario, magnitude)
    impact_ids = impact_body_ids(model)

    # Pre-check: if the UNPROTECTED baseline never falls for ANY tested
    # condition at this magnitude, no candidate pose can ever produce a real
    # score. Check up front with the cheap 'stand' pose rather than burning
    # the whole CMA-ES budget on a search that cannot possibly succeed.
    probe_score, probe_diag, probe_recovery = score_pose(model, scenario, stand, conditions, impact_ids,
                                                          lead_time_s, snapshot, p1)
    if probe_score != probe_score:  # NaN check without importing math
        return {
            "scenario_id": scen_id, "scenario_fn": fn_name, "category": category,
            "magnitude_used": magnitude, "pose": stand.tolist(), "score": float("nan"),
            "velocity_mps": None, "valid": False, "recovery_rate": None,
            "winning_restart": None, "diagnostics": [asdict(d) for d in probe_diag],
        }

    # Track (score, diag, recovery_rate, restart_idx) for the best candidate
    # across ALL restarts. -1 = the 'stand' probe itself never beaten by any
    # restart (should be rare, but possible on a very easy bin). Selection is
    # purely by `score` (pure weighted peak force + tracking-error penalty,
    # matching the reference doc's own objective -- see score_pose).
    # recovery_rate is a derived reporting metric only: it is NOT a scoring
    # term, but a fully prevented fall already scores 0 (every force term is
    # exactly 0 with no contact), so CMA-ES is already maximally incentivized
    # to find recovery wherever the search reaches it, with no extra term
    # needed to push it there.
    best_score, best_diag, best_x, best_recovery, best_restart = (
        probe_score, probe_diag, stand, probe_recovery, -1)
    rng = np.random.default_rng(seed)
    stand_leg = stand[leg_ids]

    for restart_idx in range(n_restarts):
        if restart_idx == 0:
            x0, sigma0 = stand_leg.copy(), 0.3
        else:
            spread = RESTART_START_SPREAD_BASE * (1.0 + 0.6 * restart_idx)
            x0 = np.clip(stand_leg + rng.normal(0.0, spread, size=stand_leg.shape), lo, hi)
            sigma0 = 0.3 * (RESTART_SIGMA_GROWTH ** restart_idx)

        es = cma.CMAEvolutionStrategy(
            x0, sigma0,
            {"bounds": [lo.tolist(), hi.tolist()], "popsize": popsize,
             "maxiter": maxiter, "seed": seed + restart_idx, "verbose": -9},
        )
        while not es.stop():
            candidates = es.ask()
            full_candidates = [expand(np.array(x)) for x in candidates]
            if pool is not None:
                tasks = [(scenario, fx, conditions, lead_time_s) for fx in full_candidates]
                results = pool.map(_worker_score_task, tasks)
            else:
                results = [score_pose(model, scenario, fx, conditions, impact_ids,
                                       lead_time_s, snapshot, p1) for fx in full_candidates]
            fitnesses = []
            for x, fx, (s, diag, recovery_rate) in zip(candidates, full_candidates, results):
                # NaN candidates must not be handed to CMA-ES as fitness --
                # replace with a large-but-finite penalty so the search steers
                # away from them instead of crashing or being silently ignored.
                fitnesses.append(s if s == s else 1e6)
                if s == s and s < best_score:
                    # best_x stores the FULL (model.nu,) pose -- everything
                    # downstream (JSON output, cross_evaluate, merge_library,
                    # select_pose) expects a full pose_ctrl vector, not the
                    # leg-only search vector.
                    best_score, best_diag, best_x, best_recovery, best_restart = (
                        s, diag, fx, recovery_rate, restart_idx)
            es.tell(candidates, fitnesses)

    valid_vels = [dd.velocity_at_trigger_mps for dd in best_diag
                  if not dd.no_natural_fall and dd.velocity_at_trigger_mps >= 0]
    velocity_mps = float(np.mean(valid_vels)) if valid_vels else 0.0

    return {
        "scenario_id": scen_id, "scenario_fn": fn_name, "category": category,
        "magnitude_used": magnitude, "pose": best_x.tolist(), "score": float(best_score),
        "velocity_mps": velocity_mps, "valid": True, "recovery_rate": best_recovery,
        "winning_restart": best_restart, "diagnostics": [asdict(d) for d in best_diag],
    }


def run_scenario_all_bins(model, scenario, quantiles, lead_time_s, popsize, maxiter,
                           snapshot, p1, pool=None, n_restarts=1):
    """Optimize every velocity bin for one scenario. Returns bins sorted
    ascending by MEASURED velocity (not by quantile order -- stochastic
    contact dynamics mean quantile order and measured-velocity order can
    occasionally disagree; sorting by the measured value keeps the
    interpolation curve well-defined regardless)."""
    scen_id, category, fn_name, nominal_dir = scenario
    bins = []
    for q in quantiles:
        magnitude = fall_magnitude(fn_name, q, p1)
        res = optimize_pose_for_bin(model, scenario, magnitude, lead_time_s, popsize,
                                     maxiter, snapshot, p1, seed=int(q * 1000), pool=pool,
                                     n_restarts=n_restarts)
        res["quantile"] = q
        bins.append(res)
    valid_bins = [b for b in bins if b["valid"]]
    valid_bins.sort(key=lambda b: b["velocity_mps"])
    invalid_quantiles = [b["quantile"] for b in bins if not b["valid"]]
    return valid_bins, invalid_quantiles


# ─────────────────────────────────────────────────────────────────
# CROSS-EVALUATION + LIBRARY MERGING
# (operates on each scenario's REFERENCE bin only -- see MERGE_REFERENCE --
#  to keep this stage's cost independent of the number of velocity bins)
# ─────────────────────────────────────────────────────────────────

def cross_evaluate(model, scenarios, reference_results, lead_time_s, snapshot, p1, pool=None):
    """score[i][j] = mean score when scenario_i's REFERENCE pose is applied
    to scenario_j's own reference-magnitude conditions. Diagonal = each
    scenario's own reference optimum."""
    impact_ids = impact_body_ids(model)
    n = len(scenarios)
    cells = [(i, j) for j in range(n) for i in range(n)]
    conds_by_j = {j: build_conditions(scenarios[j], reference_results[j]["magnitude_used"])
                  for j in range(n)}

    if pool is not None:
        tasks = [(scenarios[j], np.array(reference_results[i]["pose"]), conds_by_j[j], lead_time_s)
                 for i, j in cells]
        results = pool.map(_worker_score_task, tasks)
    else:
        results = [score_pose(model, scenarios[j], np.array(reference_results[i]["pose"]),
                               conds_by_j[j], impact_ids, lead_time_s, snapshot, p1)
                   for i, j in cells]

    matrix = np.zeros((n, n))
    for (i, j), (s, _, _rec) in zip(cells, results):
        matrix[i, j] = s
    return matrix


def merge_library(scenarios, reference_results, cross_matrix):
    """Greedily merge scenario i into scenario j (drop i's own pose, cover
    it with j's) whenever the cross-degradation is within MERGE_TOLERANCE of
    each scenario's own optimum, stopping once the library is within
    [TARGET_LIBRARY_MIN, TARGET_LIBRARY_MAX] poses. Unchanged from the
    single-magnitude version except it operates on reference_results
    (each scenario's chosen reference bin) instead of a single fixed pose."""
    n = len(scenarios)
    own = np.diag(cross_matrix)
    groups = {i: {i} for i in range(n)}
    representative = {i: i for i in range(n)}

    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            degradation = (cross_matrix[i, j] - own[j]) / max(own[j], 1e-6)
            pairs.append((degradation, i, j))
    pairs.sort(key=lambda t: t[0])

    for degradation, i, j in pairs:
        if len(groups) <= TARGET_LIBRARY_MIN:
            break
        if degradation > MERGE_TOLERANCE:
            break
        pi, pj = representative[i], representative[j]
        if pi == pj:
            continue
        ok = all((cross_matrix[i, k] - own[k]) / max(own[k], 1e-6) <= MERGE_TOLERANCE
                 for k in groups[pj] | {j})
        if not ok:
            continue
        groups[pi] |= groups.pop(pj)
        for k in groups[pi]:
            representative[k] = pi

    return groups, representative


# ─────────────────────────────────────────────────────────────────
# MARGIN CONFIRMATION (mentor's explicit ask)
# ─────────────────────────────────────────────────────────────────

def confirm_margin(phase2_median_lead_ms=396.0, phase2_mean_lead_ms=380.0):
    budget_ms = POSE_TRANSITION_BUDGET_S * 1000.0
    margin_median = phase2_median_lead_ms - budget_ms
    margin_mean = phase2_mean_lead_ms - budget_ms
    print(f"[margin check] pose transition budget = {budget_ms:.0f} ms")
    print(f"  vs Phase 2 median lead time {phase2_median_lead_ms:.0f} ms "
          f"-> {margin_median:+.0f} ms margin for compute/comms overhead")
    print(f"  vs Phase 2 mean lead time   {phase2_mean_lead_ms:.0f} ms "
          f"-> {margin_mean:+.0f} ms margin")
    if margin_mean < 0:
        print("  [WARN] even the MEAN lead time does not clear the transition "
              "budget -- tighten the pose search's transition penalty or "
              "revisit debouncing latency before Phase 5.")
    return margin_median, margin_mean


def _direction_label(scenario):
    scen_id, category, fn_name, nominal_dir = scenario
    if nominal_dir is None:
        return category  # non-directional mechanisms (surface/actuator/trip)
    table = {0: "forward", 180: "backward", 90: "left", 270: "right",
             45: "forward-left", 315: "forward-right"}
    return table.get(nominal_dir, f"{nominal_dir}deg")


# ─────────────────────────────────────────────────────────────────
# STAGE 4: LOOKUP + INTERPOLATION -- pure Python/numpy, NO mujoco/cma import.
# This is what Phase 4 (or a real-time controller) should actually import:
#     from phase3_pose_design import select_pose
# ─────────────────────────────────────────────────────────────────

def build_lookup_index(library):
    """(cause, direction) -> pose_id, for both granularities of `cause`
    (scenario_fn and category -- see the open-question note in the module
    docstring about which one the Phase 2 TCN actually emits)."""
    idx = {}
    for row in library["lookup_keys"]:
        idx.setdefault((row["scenario_fn"], row["direction"]), row["pose_id"])
        idx.setdefault((row["category"], row["direction"]), row["pose_id"])
    pose_by_id = {p["pose_id"]: p for p in library["poses"]}
    return idx, pose_by_id


def _interpolate_control_points(control_points, velocity_mps):
    """Linear interpolation between the two control points bracketing
    `velocity_mps`; clamps at the ends rather than extrapolating. Returns
    (pose_array, out_of_range: bool)."""
    pts = sorted(control_points, key=lambda c: c["velocity_mps"])
    vs = [c["velocity_mps"] for c in pts]
    if velocity_mps <= vs[0]:
        return np.array(pts[0]["pose"], dtype=float), velocity_mps < vs[0]
    if velocity_mps >= vs[-1]:
        return np.array(pts[-1]["pose"], dtype=float), velocity_mps > vs[-1]
    for k in range(len(pts) - 1):
        v_lo, v_hi = vs[k], vs[k + 1]
        if v_lo <= velocity_mps <= v_hi:
            p_lo = np.array(pts[k]["pose"], dtype=float)
            p_hi = np.array(pts[k + 1]["pose"], dtype=float)
            if v_hi - v_lo < 1e-9:
                return p_lo, False
            t = (velocity_mps - v_lo) / (v_hi - v_lo)
            return p_lo + t * (p_hi - p_lo), False
    return np.array(pts[-1]["pose"], dtype=float), True  # unreachable in practice


def select_pose(library, cause, direction, velocity_mps):
    """Stage 4: rule-based lookup + interpolation. NO ML inference, fully
    deterministic and auditable, per the mentor's regulatory requirement.

    Args:
        library: the loaded pose_library.json dict.
        cause: either a scenario_fn (e.g. "push_forward") or a category
               (e.g. "push") -- whichever granularity the Phase 2 TCN's
               Cause-class output emits (see module docstring open question).
        direction: a direction label matching _direction_label()'s output
                   space (e.g. "forward", "backward", "left", "trip").
        velocity_mps: Stage 3's continuous fall-velocity estimate (0-5 m/s).

    Returns None if no pose covers `cause` at all (should not happen once
    the library is complete and correct -- surfacing None rather than
    guessing is deliberate for the regulatory audit trail). Otherwise a dict
    with the resulting joint angles and metadata about how the match/
    interpolation was performed.
    """
    idx, pose_by_id = build_lookup_index(library)
    pose_id = idx.get((cause, direction))
    matched_direction = direction
    exact_direction_match = pose_id is not None
    if pose_id is None:
        # Fall back: same cause, any covered direction. Better to hand back
        # a pose designed for the right cause at the wrong direction than no
        # pose at all -- but flag it clearly so this fallback path is
        # auditable, not silent.
        candidates = [r for r in library["lookup_keys"]
                      if r["scenario_fn"] == cause or r["category"] == cause]
        if not candidates:
            return None
        pose_id = candidates[0]["pose_id"]
        matched_direction = candidates[0]["direction"]

    entry = pose_by_id[pose_id]
    pose_vec, out_of_range = _interpolate_control_points(entry["velocity_control_points"], velocity_mps)
    vs = [c["velocity_mps"] for c in entry["velocity_control_points"]]

    return {
        "pose_id": pose_id,
        "matched_cause_exact": pose_id is not None and exact_direction_match,
        "matched_direction": matched_direction,
        "requested_direction": direction,
        "requested_velocity_mps": velocity_mps,
        "velocity_range_covered_mps": [min(vs), max(vs)],
        "velocity_out_of_range": out_of_range,
        "joint_angles": pose_vec.tolist(),
    }


def _selftest():
    """Pure-Python sanity check of select_pose()/interpolation -- no mujoco,
    no cma, no model file needed. Run with `python phase3_pose_design.py
    --selftest` any time you touch the lookup/interpolation logic."""
    fake_library = {
        "poses": [{
            "pose_id": 0,
            "velocity_control_points": [
                {"velocity_mps": 1.0, "pose": [0.0, 0.0]},
                {"velocity_mps": 3.0, "pose": [2.0, 4.0]},
            ],
        }],
        "lookup_keys": [
            {"scenario_fn": "push_forward", "category": "push", "direction": "forward", "pose_id": 0},
        ],
    }
    checks = []

    r = select_pose(fake_library, "push_forward", "forward", 2.0)
    checks.append(("midpoint interpolation", np.allclose(r["joint_angles"], [1.0, 2.0])))

    r = select_pose(fake_library, "push_forward", "forward", 0.0)
    checks.append(("clamp below range", np.allclose(r["joint_angles"], [0.0, 0.0]) and r["velocity_out_of_range"]))

    r = select_pose(fake_library, "push_forward", "forward", 5.0)
    checks.append(("clamp above range", np.allclose(r["joint_angles"], [2.0, 4.0]) and r["velocity_out_of_range"]))

    r = select_pose(fake_library, "push", "forward", 2.0)
    checks.append(("category-level lookup", np.allclose(r["joint_angles"], [1.0, 2.0])))

    r = select_pose(fake_library, "push_forward", "backward", 2.0)
    checks.append(("direction fallback", r is not None and r["matched_direction"] == "forward"
                   and not r["matched_cause_exact"]))

    r = select_pose(fake_library, "trip", "forward", 2.0)
    checks.append(("unknown cause returns None", r is None))

    all_ok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("SELFTEST", "PASSED" if all_ok else "FAILED")
    return all_ok


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml")
    ap.add_argument("--out", default="phase3_out_v7",
                     help="Directory where pose_library.json is written.")
    ap.add_argument("--passive-pendulum", action="store_true",
                     help="Zero the pendulum's actuator gain/bias (see "
                          "load_instrumented_model) so the torso is governed "
                          "by gravity/damping/ROM-limit only instead of the "
                          "XML's kp=800 position-hold. Off by default: it "
                          "changes the UNPROTECTED baseline too, deviating "
                          "from generate_fall_dataset_final.py's own defined "
                          "'passive PD/impedance hold' baseline -- turn on "
                          "deliberately to A/B against that baseline, not "
                          "as a default.")
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--restarts", type=int, default=1,
                     help="Independent CMA-ES restarts per (scenario, velocity bin), "
                          "keeping the global best. Restart 0 always starts at 'stand'; "
                          "restarts >0 start from a randomized, increasingly wide "
                          "perturbation of 'stand' with growing sigma0 (IPOP-style), "
                          "so the search can find fall-preventing poses that aren't "
                          "reachable by descending the force gradient smoothly from "
                          "'stand'. Multiplies wall-clock cost by roughly this factor "
                          "-- each restart still parallelizes its own population "
                          "across --workers as before.")
    ap.add_argument("--lead-time", type=float, default=LEAD_TIME_S)
    ap.add_argument("--velocity-quantiles", type=str,
                     default=",".join(str(q) for q in VELOCITY_QUANTILES),
                     help="Comma-separated magnitude quantiles sampled per scenario "
                          "to build each pose's velocity->joint-angle curve.")
    ap.add_argument("--quick", action="store_true",
                     help="tiny popsize/maxiter/1 scenario/1 velocity bin, for a smoke test")
    ap.add_argument("--workers", type=int, default=1,
                     help="CPU worker processes for parallel pose evaluation "
                          "(there is no GPU path for plain MuJoCo). 1 = serial. "
                          "Each worker loads its own model copy.")
    ap.add_argument("--p1-module", default="generate_fall_dataset_final",
                     help="Importable module name for the Phase-1 scenario "
                          "implementation (falls back to generate_fall_dataset "
                          "if this one is not importable).")
    ap.add_argument("--selftest", action="store_true",
                     help="Run the pure-Python select_pose()/interpolation "
                          "unit tests and exit. No mujoco/cma/model required.")
    args = ap.parse_args()

    if args.selftest:
        ok = _selftest()
        sys.exit(0 if ok else 1)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import importlib
        p1 = importlib.import_module(args.p1_module)
    except ImportError:
        import generate_fall_dataset as p1
        args.p1_module = "generate_fall_dataset"

    required_p1 = ("SCENARIOS", "MAGNITUDE_RANGES", "SCENARIO_BUILDERS",
                   "run_trial", "get_foot_geom_ids", "snapshot_contacts")
    missing_p1 = [name for name in required_p1 if not hasattr(p1, name)]
    if missing_p1:
        raise RuntimeError(
            "Phase-1 module is missing required interfaces: " + ", ".join(missing_p1)
        )

    os.makedirs(args.out, exist_ok=True)
    confirm_margin()

    quantiles = [float(q) for q in args.velocity_quantiles.split(",") if q.strip()]
    model = load_instrumented_model(args.model, passive_pendulum=args.passive_pendulum)
    snapshot = snapshot_model_state(model)
    scenarios = p1.SCENARIOS if not args.quick else p1.SCENARIOS[:1]
    if args.quick:
        quantiles = quantiles[-1:]
    popsize = 4 if args.quick else args.popsize
    maxiter = 2 if args.quick else args.maxiter
    n_restarts = 1 if args.quick else args.restarts

    pool = None
    if args.workers > 1:
        model_abspath = os.path.abspath(args.model)
        pool = mp.Pool(processes=args.workers, initializer=_worker_init,
                        initargs=(model_abspath, args.p1_module, args.passive_pendulum))
        print(f"Started a pool of {args.workers} worker processes (CPU-parallel).")

    print(f"\nOptimizing {len(scenarios)} scenario(s) x {len(quantiles)} velocity bin(s) "
          f"(popsize={popsize}, maxiter={maxiter}, restarts={n_restarts}, "
          f"lead_time={args.lead_time}s)...")

    bins_by_scenario = {}   # scenario index -> list of valid bin result dicts
    excluded = []
    try:
        for idx, scenario in enumerate(scenarios):
            t0 = time.time()
            valid_bins, invalid_quantiles = run_scenario_all_bins(
                model, scenario, quantiles, args.lead_time, popsize, maxiter, snapshot, p1,
                pool=pool, n_restarts=n_restarts)
            scen_id, category, fn_name, nominal_dir = scenario
            if not valid_bins:
                print(f"  scenario {scen_id:>2} ({fn_name:<18}) [WARN] no quantile in "
                      f"{quantiles} produced a natural fall -- excluded. Recalibrate "
                      f"MAGNITUDE_RANGES/quantiles for this scenario before considering "
                      f"Phase 3 complete for it. [{time.time()-t0:.1f}s]")
                excluded.append({"scenario_id": scen_id, "scenario_fn": fn_name,
                                  "quantiles_tried": quantiles})
                continue
            bins_by_scenario[idx] = valid_bins
            vel_str = ", ".join(f"{b['velocity_mps']:.2f}m/s->{b['score']:.0f}"
                                 f"(rec {b['recovery_rate']*100:.0f}%)" for b in valid_bins)
            warn = f" [WARN: quantiles {invalid_quantiles} never fell]" if invalid_quantiles else ""
            print(f"  scenario {scen_id:>2} ({fn_name:<18}) bins=[{vel_str}]{warn} "
                  f"[{time.time()-t0:.1f}s]")

        valid_indices = sorted(bins_by_scenario.keys())
        scenarios_v = [scenarios[i] for i in valid_indices]
        # Reference bin per scenario for cross-eval/merge (see MERGE_REFERENCE).
        reference_results = []
        for i in valid_indices:
            bins = bins_by_scenario[i]
            reference_results.append(bins[-1] if MERGE_REFERENCE == "max" else bins[0])

        if len(scenarios_v) > 1:
            print("\nCross-evaluating pose robustness across scenarios (reference bin only)...")
            cross_matrix = cross_evaluate(model, scenarios_v, reference_results, args.lead_time,
                                           snapshot, p1, pool=pool)
            groups, representative = merge_library(scenarios_v, reference_results, cross_matrix)
            print(f"Merged {len(scenarios_v)} scenario-specific poses into {len(groups)} library poses.")
        elif len(scenarios_v) == 1:
            groups = {0: {0}}
        else:
            groups = {}
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    library = []
    lookup_keys = []
    for pose_id, covered in groups.items():
        rep_scenario_local_idx = pose_id  # representative's index within scenarios_v/bins_by_scenario keys
        rep_global_idx = valid_indices[rep_scenario_local_idx]
        rep_bins = bins_by_scenario[rep_global_idx]
        covered_scenarios = [scenarios_v[k] for k in covered]
        directions = sorted({_direction_label(s) for s in covered_scenarios})
        velocity_control_points = [
            {
                "velocity_mps": b["velocity_mps"],
                "magnitude_used": b["magnitude_used"],
                "quantile": b["quantile"],
                "pose": b["pose"],
                "score": b["score"],
                "recovery_rate": b["recovery_rate"],
                "winning_restart": b["winning_restart"],
                "diagnostics": b["diagnostics"],
            }
            for b in rep_bins
        ]
        library.append({
            "pose_id": pose_id,
            "covers_scenario_ids": [scenarios_v[k][0] for k in covered],
            "covers_scenario_fns": sorted({scenarios_v[k][2] for k in covered}),
            "covers_categories": sorted({scenarios_v[k][1] for k in covered}),
            "fall_directions": directions,
            "reference_bin_score": reference_results[pose_id]["score"],
            "reference_bin_velocity_mps": reference_results[pose_id]["velocity_mps"],
            "reference_bin_recovery_rate": reference_results[pose_id]["recovery_rate"],
            "velocity_control_points": velocity_control_points,
        })
        for k in covered:
            s = scenarios_v[k]
            for d_label in [_direction_label(s)]:
                lookup_keys.append({
                    "scenario_id": s[0], "category": s[1], "scenario_fn": s[2],
                    "direction": d_label, "pose_id": pose_id,
                })

    out_path = os.path.join(args.out, "pose_library_v7.json")
    with open(out_path, "w") as f:
        json.dump({
            "lead_time_s_used_for_design": args.lead_time,
            "pose_transition_budget_s": POSE_TRANSITION_BUDGET_S,
            "velocity_quantiles_sampled": quantiles,
            "scoring_weights": {
                "w_head": W_HEAD, "w_pelvis": W_PELVIS, "w_other": W_OTHER,
                "tracking_error_tol_rad": TRACKING_ERROR_TOL_RAD,
                "tracking_error_penalty_per_rad": TRACKING_ERROR_PENALTY_PER_RAD,
                "note": (
                    "Score = weighted peak contact force (w_head/w_pelvis/w_other) "
                    "plus a tracking-error penalty, matching the reference doc's "
                    "own optimizer (section 8.3: 'evaluate peak contact force ... "
                    "lower is better') plus its section 8.2 minimum-jerk-trajectory "
                    "safety requirement. There is deliberately NO separate reward "
                    "for preventing the fall outright: every reference-doc "
                    "principle (lower CoM, align impact zone, minimum-jerk "
                    "transition) is about reducing injury from a fall that is "
                    "already happening, not about avoiding the fall itself -- and "
                    "a fully prevented fall already scores 0 on every force term "
                    "(no contact occurs), which is already the best any peak-force "
                    "sum can produce, so this objective already rewards prevention "
                    "as strongly as physically possible without a separate term. "
                    "tracking_error_penalty_per_rad IS a real, separate injury "
                    "channel from ground impact (joint/actuator torque spikes from "
                    "failing to follow the planned smooth path under load) -- "
                    "applied to every triggered trial, recovered included, since "
                    "that risk is independent of whether the fall itself is "
                    "avoided. Each control point's recovery_rate (below) is "
                    "reported for hardware go/no-go review -- the fraction of "
                    "tested conditions where the pose prevented the fall entirely "
                    "-- but is informational only and does not affect score."
                ),
            },
            "velocity_estimate_source": (
                "base linear velocity magnitude (||qvel[0:3]||) measured at the "
                "instant the pose is triggered -- proxy for the Phase-2 TCN's "
                "Stage-3 velocity-regression output. Verify qvel[0:3] really is "
                "the free-joint base linear velocity against g1_pendulum.xml's "
                "actual joint ordering before trusting this in Phase 4/5."
            ),
            "actuator_scope": (
                "CMA-ES optimizes ONLY the 12 leg actuators (pose_ctrl's "
                "pendulum_roll/pendulum_pitch entries are pinned at their "
                "'stand' keyframe value for every trial -- see "
                "actuator_group_ids). This rig's real-world target is a "
                "LOWER-LIMB-ONLY exoskeleton for paralyzed rehab patients: "
                "the hardware has no actuator on the wearer's own "
                "torso/head, so letting the optimizer command the pendulum "
                "(as every pose in pose_library_v5.json and earlier does) "
                "let it 'solve' head/pelvis impact by swinging that mass "
                "out of the way -- something no real wearer or exoskeleton "
                "can do. Confirmed visually in the MuJoCo viewer before "
                "this fix. Pinning removes the exploit but is NOT full "
                "passivity: the pendulum's high-kp position actuator still "
                "resists being pushed off 'stand' by the disturbance, "
                "acting like a stiff neck-brace rather than a truly floppy "
                "unsupported torso. A closer-to-real passive response would "
                "need an XML-level actuator-gain change -- flag with your "
                "mentor before making it, since it touches the shared model "
                "file every phase depends on. Every pose in this file was "
                "(re-)optimized under the leg-only scope; anything compared "
                "against pose_library_v5.json or earlier must be re-run, "
                "not diffed directly, since the search space itself changed. "
                "UPDATE (v7): pinning ctrl alone was not enough -- the "
                "pendulum's own kp=800 position servo still rigidly held it "
                "at 0 rad through the whole fall, so v6 turned it into a "
                "stiff mast that dumped 3-6.7kN spikes onto the head-proxy "
                "contact with pelvis/other near 0 in many bins. "
                "load_instrumented_model() now accepts passive_pendulum= "
                "(CLI: --passive-pendulum), which zeroes the pendulum's "
                "actuator_gainprm/biasprm at load time so both pendulum "
                "actuators output exactly 0 N*m always -- torso governed "
                "purely by gravity, its own joint damping (damping=2.0/ "
                "joint), and its +-45deg range limit. Left OFF by default: "
                "it also changes the UNPROTECTED baseline, deviating from "
                "generate_fall_dataset_final.py's own defined 'passive PD/ "
                "impedance hold ... not zero control ... honest, "
                "non-fabricated' baseline. This run used "
                f"passive_pendulum={args.passive_pendulum}. Do not compare "
                "runs with different passive_pendulum settings as if they "
                "were the same experiment -- pose_library_v6.json's "
                "unprotected numbers used the rigid-mast torso and are not "
                "a valid 'no protection' reference for a passive_pendulum="
                "True run, or vice versa."
            ),
            "impact_proxies": {
                "pelvis": "pelvis collision geom (existing)",
                "head": "pendulum bob, contact enabled at runtime",
                "hands": "NOT MODELED -- rig has no arms",
                "other": "any other non-foot body (e.g. knee/hip) -- weighted "
                         "lower than pelvis/head, but not free",
            },
            "open_questions_for_mentor": [
                "Spec says 6 actuated joints; this rig has model.nu actuators "
                "(14 on the real g1_pendulum.xml: 2 pendulum + 12 leg). This "
                "library outputs angles for every actuator on the loaded "
                "model -- confirm which joint set Stage 4 actually expects.",
                "Lookup table is keyed on BOTH scenario_fn and category for "
                "'cause' because it isn't yet confirmed which granularity the "
                "Phase 2 TCN's Cause-class output uses. Resolve before Phase 5.",
            ],
            "excluded_scenarios_needing_recalibration": excluded,
            "n_poses": len(library),
            "poses": library,
            "lookup_keys": lookup_keys,
        }, f, indent=2)
    print(f"\nWrote {len(library)} poses ({len(lookup_keys)} lookup rows) to {out_path}")


if __name__ == "__main__":
    main()
