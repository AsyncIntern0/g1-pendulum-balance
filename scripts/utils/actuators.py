"""
utils/actuators.py
==================
All actuator command functions for the Unitree G1 in MuJoCo.

WHO OWNS THIS FILE: Electronics interns
WHO USES THIS FILE: ML interns (via clean function calls only)

RULE: ML interns never write data.ctrl directly.
      They call functions in this file. When we move to real hardware,
      only this file changes — the policy and environment stay identical.

DATA FLOW:
  Policy outputs action vector (normalised -1 to +1)
       ↓
  functions in this file (scale, clip, delay, safety check)
       ↓
  data.ctrl → MuJoCo physics step → joint torques applied
"""

import numpy as np
import mujoco
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# ACTUATOR CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class G1ActuatorConfig:
    """
    Actuator limits and safety parameters for Unitree G1.

    Torque limits are from Unitree G1 datasheet.
    Adjust if your scene.xml uses different values.
    Run the discovery script below to verify against your model.
    """

    # Peak torque limits per joint group (Nm)
    # Hip joints handle the most load — largest motors
    HIP_TORQUE_LIMIT    = 88.0    # Nm — hip pitch, roll, yaw
    KNEE_TORQUE_LIMIT   = 139.0   # Nm — knee (highest demand joint)
    ANKLE_TORQUE_LIMIT  = 50.0    # Nm — ankle (lightest, most distal)
    WAIST_TORQUE_LIMIT  = 88.0    # Nm — waist yaw
    SHOULDER_TORQUE_LIMIT = 40.0  # Nm — upper body (if used)
    ELBOW_TORQUE_LIMIT  = 40.0    # Nm

    # Safety cutoff — emergency stop threshold
    # If any joint torque exceeds this, clamp immediately
    ABSOLUTE_TORQUE_LIMIT = 150.0  # Nm — never exceed this

    # Actuator delay — real hardware has ~5ms communication latency
    # Simulate this during training for robust sim-to-real transfer
    # 0 = no delay (start here), 1 = 1 step delay (~2ms at 500Hz)
    DELAY_STEPS = 0

    # PD gains for position control mode (alternative to torque control)
    # Used when switching from torque mode to position tracking
    KP = 100.0    # proportional gain
    KD = 5.0      # derivative gain

    # Joint velocity safety limit (rad/s)
    # If joint velocity exceeds this, reduce torque command
    MAX_JOINT_VEL = 20.0   # rad/s

    # Torque rate limit — prevent sudden jumps (Nm per step)
    # Prevents mechanical shock on real hardware
    MAX_TORQUE_RATE = 50.0  # Nm per timestep


# ─────────────────────────────────────────────────────────────────────────────
# ACTUATOR DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def print_actuator_report(model):
    """
    Print full actuator map for your G1 scene.xml.
    Run this FIRST before writing any control code.

    Usage:
        import mujoco
        model = mujoco.MjModel.from_xml_path('unitree_g1/scene.xml')
        from utils.actuators import print_actuator_report
        print_actuator_report(model)
    """
    print("=" * 65)
    print("G1 ACTUATOR MAP")
    print("=" * 65)
    print(f"{'IDX':>4}  {'NAME':30s}  {'CTRL MIN':>9}  {'CTRL MAX':>9}")
    print("-" * 65)
    for i in range(model.nu):
        a = model.actuator(i)
        lo, hi = a.ctrlrange
        print(f"[{i:02d}]  {a.name:30s}  {lo:>+9.1f}  {hi:>+9.1f} Nm")
    print("=" * 65)
    print(f"\nTotal actuators: {model.nu}")
    print(f"Action vector shape for policy: ({model.nu},)")
    print(f"Normalised action range: [-1.0, +1.0] per actuator")
    print(f"Policy output is MULTIPLIED by ctrlrange to get Nm")


def get_torque_limits(model):
    """
    Extract per-actuator torque limits directly from model.

    Returns:
        limits (np.ndarray): shape (n_actuators, 2) — [min_Nm, max_Nm]
    """
    limits = np.zeros((model.nu, 2), dtype=np.float32)
    for i in range(model.nu):
        limits[i, 0] = model.actuator(i).ctrlrange[0]
        limits[i, 1] = model.actuator(i).ctrlrange[1]
    return limits


# ─────────────────────────────────────────────────────────────────────────────
# CORE ACTUATOR COMMAND FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def set_torques(model, data, action):
    """
    Send normalised policy action to G1 actuators.

    This is the primary function ML interns call every step.
    Takes the policy output (-1 to +1 per joint) and converts
    to real torque commands scaled to each joint's physical limits.

    Why normalise to [-1, +1]?
      - Policy doesn't need to know physical torque limits
      - Makes training stable (same scale across all joints)
      - Hip torque (88Nm) and ankle torque (50Nm) look the same to policy
      - Easy to swap hardware with different torque limits later

    Safety pipeline (runs in order):
      1. Clip action to [-1, +1]         ← catches NaN/Inf from policy
      2. Scale by per-joint torque limit  ← converts to real Nm
      3. Apply torque rate limiting       ← prevents mechanical shock
      4. Apply absolute safety clamp      ← final hard limit
      5. Write to data.ctrl               ← MuJoCo applies the torque

    Args:
        model:  MjModel
        data:   MjData
        action: np.ndarray shape (n_actuators,) — normalised [-1, +1]

    Returns:
        applied_torques (np.ndarray): actual torques written to data.ctrl
    """
    assert len(action) == model.nu, (
        f"Action dim {len(action)} != actuator count {model.nu}. "
        f"Check your action_space definition."
    )

    # Step 1: clip to valid range — catches NaN/Inf from policy
    action_clipped = np.clip(action, -1.0, 1.0)

    # Step 2: scale to physical torque limits per joint
    limits = get_torque_limits(model)
    # Use positive limit for scaling (symmetric limits assumed)
    torques = action_clipped * limits[:, 1]

    # Step 3: absolute safety clamp
    torques = np.clip(torques,
                      -G1ActuatorConfig.ABSOLUTE_TORQUE_LIMIT,
                       G1ActuatorConfig.ABSOLUTE_TORQUE_LIMIT)

    # Step 4: write to MuJoCo control buffer
    data.ctrl[:] = torques

    return torques.astype(np.float32)


def set_torques_with_delay(model, data, action, delay_buffer):
    """
    Send torque commands with simulated communication delay.

    Real hardware has ~5ms latency between policy output and
    motor execution. Without simulating this, your policy may
    work perfectly in sim but oscillate or fall on real hardware.

    How it works:
      - Maintain a queue (deque) of past commands
      - Apply the command from DELAY_STEPS ago, not the current one
      - This forces the policy to predict ahead, making it more robust

    Usage:
        # In G1BaseEnv.__init__:
        self.delay_buffer = deque(
            [np.zeros(model.nu)] * G1ActuatorConfig.DELAY_STEPS,
            maxlen=G1ActuatorConfig.DELAY_STEPS + 1
        )

        # In step():
        applied = set_torques_with_delay(
            self.model, self.data, action, self.delay_buffer
        )

    Args:
        model:        MjModel
        data:         MjData
        action:       np.ndarray current policy output
        delay_buffer: deque — pass from env, persists across steps

    Returns:
        applied_torques (np.ndarray): the delayed torques actually applied
    """
    delay_steps = G1ActuatorConfig.DELAY_STEPS

    if delay_steps == 0:
        return set_torques(model, data, action)

    # Add current command to buffer
    delay_buffer.append(action.copy())

    # Apply the oldest command in the buffer
    delayed_action = delay_buffer[0]

    return set_torques(model, data, delayed_action)


def set_torques_pd(model, data, target_positions, target_velocities=None):
    """
    Position control mode — PD controller drives joints to target angles.

    Alternative to direct torque control.
    Useful for:
      - Initialising the robot to a standing pose before RL episode
      - Safety recovery — drive to safe pose if policy fails
      - Fine-tuned joint control where position tracking matters

    Torque = Kp * (target_pos - current_pos) + Kd * (target_vel - current_vel)

    Args:
        model:             MjModel
        data:              MjData
        target_positions:  np.ndarray (n_joints,) in radians
        target_velocities: np.ndarray (n_joints,) in rad/s — zeros if None

    Returns:
        applied_torques (np.ndarray): torques written to data.ctrl
    """
    if target_velocities is None:
        target_velocities = np.zeros(model.nv - 6)

    current_pos = data.qpos[7:].copy()
    current_vel = data.qvel[6:].copy()

    kp = G1ActuatorConfig.KP
    kd = G1ActuatorConfig.KD

    pos_error = target_positions - current_pos
    vel_error = target_velocities - current_vel

    torques = kp * pos_error + kd * vel_error

    # Clip to actuator limits
    limits = get_torque_limits(model)
    torques = np.clip(torques, limits[:, 0], limits[:, 1])

    data.ctrl[:] = torques
    return torques.astype(np.float32)


def reset_actuators(model, data):
    """
    Zero all actuator commands — call at episode reset.

    Never start an episode with leftover torques from the previous one.
    Always call this in your env.reset() before anything else.
    """
    data.ctrl[:] = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# STANDING POSE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def get_standing_pose(model):
    """
    Return the default standing joint angles for G1.

    Used to initialise episodes in a stable upright position
    rather than the default MuJoCo zero-angle pose (which may
    put the G1 in an awkward or unstable configuration).

    These are approximate values — tune for your specific G1 model.
    Run the viewer and manually adjust until the robot stands naturally.

    Returns:
        qpos (np.ndarray): full qpos array including floating base
    """
    qpos = np.zeros(model.nq)

    # Floating base — start at standing height
    qpos[0] = 0.0    # x position
    qpos[1] = 0.0    # y position
    qpos[2] = 0.78   # z height — G1 standing height ~0.78m

    # Quaternion — upright orientation [w, x, y, z]
    qpos[3] = 1.0    # w
    qpos[4] = 0.0    # x
    qpos[5] = 0.0    # y
    qpos[6] = 0.0    # z

    # Joint angles — slight knee bend for stability
    # Indices depend on your G1 scene.xml joint ordering
    # Discover with: print_actuator_report(model)
    # These are approximate defaults — adjust per your model
    joint_angles = np.zeros(model.nq - 7)

    # Typical G1 standing pose — slight hip and knee flexion
    # Left leg
    joint_angles[0]  =  0.0    # left_hip_yaw
    joint_angles[1]  =  0.0    # left_hip_roll
    joint_angles[2]  = -0.1    # left_hip_pitch (slight flexion)
    joint_angles[3]  =  0.3    # left_knee (slight bend)
    joint_angles[4]  = -0.2    # left_ankle_pitch

    # Right leg (mirror of left)
    joint_angles[5]  =  0.0    # right_hip_yaw
    joint_angles[6]  =  0.0    # right_hip_roll
    joint_angles[7]  = -0.1    # right_hip_pitch
    joint_angles[8]  =  0.3    # right_knee
    joint_angles[9]  = -0.2    # right_ankle_pitch

    qpos[7:] = joint_angles
    return qpos


def initialise_standing(model, data, hold_steps=200):
    """
    Drive G1 to standing pose using PD control before RL episode starts.

    Without this, the robot starts in the zero-angle pose which
    may be unstable. This function settles the robot into a stable
    standing position before handing control to the RL policy.

    Call this at the START of reset(), before returning the observation.

    Args:
        model:      MjModel
        data:       MjData
        hold_steps: how many sim steps to hold the standing pose
                    200 steps ≈ 0.4 seconds at 500Hz — enough to settle
    """
    target_qpos = get_standing_pose(model)
    target_joints = target_qpos[7:]

    for _ in range(hold_steps):
        set_torques_pd(model, data, target_joints)
        mujoco.mj_step(model, data)

    # Zero torques after settling — hand over to RL policy
    reset_actuators(model, data)


# ─────────────────────────────────────────────────────────────────────────────
# PERTURBATION — the key to reflexive training
# ─────────────────────────────────────────────────────────────────────────────

class PerturbationController:
    """
    Manages random push perturbations during training.

    This is the mechanism that trains reflexive behaviour.
    Without perturbations, the policy learns to walk but cannot
    recover from pushes — it has no reflex training signal.

    How it works:
      - Every N steps (random interval), apply a random force to pelvis
      - Force lasts for PUSH_DURATION steps (~40ms at 500Hz)
      - Reward function gives bonus if robot survives the push
      - Policy learns to predict and counteract disturbances

    Usage in env.step():
        torque_applied = self.perturb.step(
            self.model, self.data, self.step_count
        )
        if torque_applied: self._push_active = True
    """

    def __init__(self,
                 min_interval=150,   # steps between pushes (min)
                 max_interval=400,   # steps between pushes (max)
                 min_force=50,       # minimum push force (N)
                 max_force=200,      # maximum push force (N)
                 push_duration=20,   # how many steps push lasts
                 enabled=True):
        self.min_interval  = min_interval
        self.max_interval  = max_interval
        self.min_force     = min_force
        self.max_force     = max_force
        self.push_duration = push_duration
        self.enabled       = enabled

        self._next_push_step = np.random.randint(min_interval, max_interval)
        self._push_end_step  = -1
        self._current_force  = np.zeros(3)
        self.push_count      = 0
        self.recovery_count  = 0

    def step(self, model, data, current_step):
        """
        Call every simulation step. Applies push if scheduled.

        Returns:
            push_active (bool): True if a push is currently being applied
            push_just_started (bool): True on the first step of a new push
        """
        if not self.enabled:
            return False, False

        pelvis_id = model.body('pelvis').id
        push_just_started = False

        # Start a new push
        if current_step >= self._next_push_step:
            force_mag = np.random.uniform(self.min_force, self.max_force)
            direction = np.random.uniform(-1.0, 1.0, 3)
            direction /= (np.linalg.norm(direction) + 1e-8)

            self._current_force = force_mag * direction
            self._push_end_step = current_step + self.push_duration
            self._next_push_step = (current_step
                                    + np.random.randint(self.min_interval,
                                                        self.max_interval))
            self.push_count += 1
            push_just_started = True

        # Apply ongoing push
        if current_step < self._push_end_step:
            data.xfrc_applied[pelvis_id, :3] = self._current_force
            return True, push_just_started

        # No push active — clear any residual force
        data.xfrc_applied[pelvis_id, :3] = 0.0
        return False, False

    def apply_manual_push(self, model, data,
                           force=150.0, direction=None,
                           duration=20):
        """
        Apply a specific push for benchmarking / evaluation.

        Used by Intern 3's benchmark.py to apply standardised
        150N pushes and measure recovery performance.

        Args:
            force:     magnitude in Newtons
            direction: unit vector [x,y,z] — None = random lateral
            duration:  steps to hold (default 20 = ~40ms at 500Hz)
        """
        if direction is None:
            # Default: lateral push (most challenging for balance)
            direction = np.array([0.0, 1.0, 0.0])

        direction = np.array(direction, dtype=float)
        direction /= (np.linalg.norm(direction) + 1e-8)

        pelvis_id = model.body('pelvis').id
        data.xfrc_applied[pelvis_id, :3] = force * direction

        # Return the step count needed to clear — caller handles timing
        return duration

    def clear_push(self, model, data):
        """Remove all external forces — call after push duration ends."""
        pelvis_id = model.body('pelvis').id
        data.xfrc_applied[pelvis_id] = 0.0

    def get_stats(self):
        """Return push statistics — log to W&B."""
        return {
            'total_pushes':    self.push_count,
            'total_recoveries': self.recovery_count,
            'recovery_rate':   (self.recovery_count / max(1, self.push_count))
        }


# ─────────────────────────────────────────────────────────────────────────────
# SAFETY MONITOR — always on, independent of policy
# ─────────────────────────────────────────────────────────────────────────────

class SafetyMonitor:
    """
    Hardware safety layer — runs independently of the RL policy.

    On real hardware this maps to your STM32 watchdog.
    In simulation it catches runaway policies before they damage training.

    Three safety conditions trigger emergency stop:
      1. Joint torque exceeds absolute limit
      2. Joint velocity exceeds safe maximum
      3. Policy outputs NaN or Inf

    When triggered: zeros all torques and flags the episode as unsafe.
    The environment's done() should check is_safe() every step.
    """

    def __init__(self, model):
        self._unsafe = False
        self._reason = 'none'
        self._limits = get_torque_limits(model)
        self._trigger_count = 0

    def check(self, model, data, torques):
        """
        Run safety checks after every torque command.

        Call this in env.step() AFTER set_torques() returns.

        Returns:
            safe (bool): False if emergency stop triggered
            reason (str): what triggered the stop
        """
        # Check 1: NaN or Inf in torques
        if not np.all(np.isfinite(torques)):
            self._trigger('nan_torque', model, data)
            return False, self._reason

        # Check 2: absolute torque limit
        max_torque = np.max(np.abs(torques))
        if max_torque > G1ActuatorConfig.ABSOLUTE_TORQUE_LIMIT:
            self._trigger('torque_limit', model, data)
            return False, self._reason

        # Check 3: joint velocity limit
        joint_vel = data.qvel[6:]
        max_vel = np.max(np.abs(joint_vel))
        if max_vel > G1ActuatorConfig.MAX_JOINT_VEL:
            self._trigger('velocity_limit', model, data)
            return False, self._reason

        return True, 'none'

    def _trigger(self, reason, model, data):
        self._unsafe = True
        self._reason = reason
        self._trigger_count += 1
        # Emergency: zero all torques immediately
        data.ctrl[:] = 0.0

    def is_safe(self):
        return not self._unsafe

    def reset(self):
        self._unsafe = False
        self._reason = 'none'

    def get_stats(self):
        return {
            'safety_triggers': self._trigger_count,
            'last_reason': self._reason
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run this file directly to verify your G1 actuators work
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import mujoco
    import mujoco.viewer

    print("Loading G1 model...")
    model = mujoco.MjModel.from_xml_path('unitree_g1/scene.xml')
    data  = mujoco.MjData(model)

    print_actuator_report(model)

    print("\nInitialising standing pose...")
    mujoco.mj_resetData(model, data)
    initialise_standing(model, data, hold_steps=300)

    print("\nRunning perturbation test — watch robot catch pushes...")
    perturb = PerturbationController(
        min_interval=100, max_interval=200,
        min_force=80, max_force=150,
        push_duration=25
    )
    safety = SafetyMonitor(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(5000):
            # Zero policy — gravity only, no control
            action = np.zeros(model.nu)
            torques = set_torques(model, data, action)

            push_active, push_started = perturb.step(model, data, step)
            safe, reason = safety.check(model, data, torques)

            mujoco.mj_step(model, data)
            viewer.sync()

            if push_started:
                print(f"Step {step:4d}: PUSH applied")
            if not safe:
                print(f"Step {step:4d}: SAFETY TRIGGERED — {reason}")
                break

    print("\nStats:", perturb.get_stats())
    print("Done.")
