"""
utils/sensors.py — complete version for Windows
"""
import numpy as np
import mujoco


class G1SensorConfig:
    PELVIS_ACCEL_IDX  = 0
    PELVIS_GYRO_IDX   = 1
    JOINT_POS_START   = 7
    JOINT_VEL_START   = 6
    ACCEL_NOISE_STD   = 0.02
    GYRO_NOISE_STD    = 0.003
    JOINT_POS_NOISE   = 0.001
    JOINT_VEL_NOISE   = 0.01
    ACTUATOR_DELAY_STEPS = 0


def _sensor_adr(model, sensor_idx):
    adr = 0
    for i in range(sensor_idx):
        adr += model.sensor(i).dim
    return adr


def read_imu_pelvis(model, data, add_noise=False):
    """Read trunk IMU — accelerometer + gyroscope (6 floats)."""
    try:
        # Try by name first
        accel_start = _sensor_adr(model, G1SensorConfig.PELVIS_ACCEL_IDX)
        gyro_start  = _sensor_adr(model, G1SensorConfig.PELVIS_GYRO_IDX)
        accel = data.sensordata[accel_start:accel_start+3].copy()
        gyro  = data.sensordata[gyro_start:gyro_start+3].copy()
    except Exception:
        accel = np.zeros(3)
        gyro  = np.zeros(3)

    if add_noise:
        accel += np.random.normal(0, G1SensorConfig.ACCEL_NOISE_STD, 3)
        gyro  += np.random.normal(0, G1SensorConfig.GYRO_NOISE_STD,  3)

    return np.concatenate([accel, gyro]).astype(np.float32)


def read_joint_positions(model, data, add_noise=False):
    """Read all joint angles in radians."""
    start     = G1SensorConfig.JOINT_POS_START
    joint_pos = data.qpos[start:].copy()
    if add_noise:
        joint_pos += np.random.normal(0, G1SensorConfig.JOINT_POS_NOISE,
                                       len(joint_pos))
    return joint_pos.astype(np.float32)


def read_joint_velocities(model, data, add_noise=False):
    """Read all joint velocities in rad/s."""
    start     = G1SensorConfig.JOINT_VEL_START
    joint_vel = data.qvel[start:].copy()
    if add_noise:
        joint_vel += np.random.normal(0, G1SensorConfig.JOINT_VEL_NOISE,
                                       len(joint_vel))
    return joint_vel.astype(np.float32)


def read_foot_contacts(model, data):
    """Binary foot contact detection — [left, right]."""
    left  = 0.0
    right = 0.0
    for i in range(data.ncon):
        contact    = data.contact[i]
        geom1_name = model.geom(contact.geom1).name.lower()
        geom2_name = model.geom(contact.geom2).name.lower()
        if 'left'  in geom1_name or 'left'  in geom2_name:
            left  = 1.0
        if 'right' in geom1_name or 'right' in geom2_name:
            right = 1.0
    return np.array([left, right], dtype=np.float32)


def read_trunk_tilt(model, data):
    """Trunk lean angle from vertical in degrees."""
    try:
        pelvis_id = model.body('pelvis').id
        R         = data.xmat[pelvis_id].reshape(3, 3)
        cos_tilt  = float(np.clip(R[2, 2], -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_tilt)))
    except Exception:
        return 0.0


def read_base_state(model, data):
    """Base position, orientation and velocity."""
    return {
        'pos':    data.qpos[0:3].copy().astype(np.float32),
        'quat':   data.qpos[3:7].copy().astype(np.float32),
        'linvel': data.qvel[0:3].copy().astype(np.float32),
        'angvel': data.qvel[3:6].copy().astype(np.float32),
        'height': float(data.qpos[2]),
    }


def read_foot_contact_forces(model, data):
    """Foot contact normal forces in Newtons."""
    left  = 0.0
    right = 0.0
    for i in range(data.ncon):
        contact    = data.contact[i]
        geom1_name = model.geom(contact.geom1).name.lower()
        geom2_name = model.geom(contact.geom2).name.lower()
        force_vec  = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force_vec)
        normal = abs(force_vec[0])
        if 'left'  in geom1_name or 'left'  in geom2_name:
            left  += normal
        if 'right' in geom1_name or 'right' in geom2_name:
            right += normal
    return np.array([left, right], dtype=np.float32)


def build_student_observation(model, data, add_noise=False):
    """
    Student policy observation — hardware sensors only.
    Shape: 6 + n_joints + n_joints + 2
    """
    imu      = read_imu_pelvis(model, data, add_noise)
    jnt_pos  = read_joint_positions(model, data, add_noise)
    jnt_vel  = read_joint_velocities(model, data, add_noise)
    contacts = read_foot_contacts(model, data)
    return np.concatenate([imu, jnt_pos, jnt_vel, contacts]).astype(np.float32)


def build_teacher_observation(model, data, add_noise=False):
    """
    Teacher policy observation — includes privileged sim info.
    Shape: student_obs + 3 + 3 + 1 + 2 + 3
    """
    student  = build_student_observation(model, data, add_noise)
    base     = read_base_state(model, data)
    linvel   = base['linvel']
    angvel   = base['angvel']
    tilt     = np.array([read_trunk_tilt(model, data)], dtype=np.float32)
    forces   = read_foot_contact_forces(model, data)

    try:
        pelvis_id = model.body('pelvis').id
        ext_force = data.xfrc_applied[pelvis_id, :3].copy().astype(np.float32)
    except Exception:
        ext_force = np.zeros(3, dtype=np.float32)

    return np.concatenate([student, linvel, angvel, tilt,
                           forces, ext_force]).astype(np.float32)


def get_observation_dims(model):
    """Return observation dimensions for student and teacher."""
    n_joints = model.nq - 7
    student  = 6 + n_joints + n_joints + 2
    teacher  = student + 3 + 3 + 1 + 2 + 3
    return {'student': student, 'teacher': teacher, 'n_joints': n_joints}


def is_fallen(model, data,
              tilt_threshold_deg=45.0,
              height_threshold=0.4):
    """Check if robot has fallen."""
    height = float(data.qpos[2])
    tilt   = read_trunk_tilt(model, data)
    if height < height_threshold:
        return True, 'height'
    if tilt > tilt_threshold_deg:
        return True, 'tilt'
    return False, 'none'


def print_sensor_report(model, data):
    """Print full sensor diagnostic."""
    dims    = get_observation_dims(model)
    imu     = read_imu_pelvis(model, data)
    jnt_pos = read_joint_positions(model, data)
    tilt    = read_trunk_tilt(model, data)
    contact = read_foot_contacts(model, data)
    base    = read_base_state(model, data)

    print("=" * 55)
    print("G1 SENSOR REPORT")
    print("=" * 55)
    print(f"Joints:         {dims['n_joints']}")
    print(f"Student obs:    {dims['student']} floats")
    print(f"Teacher obs:    {dims['teacher']} floats")
    print(f"Trunk accel:    [{imu[0]:+.3f} {imu[1]:+.3f} {imu[2]:+.3f}] m/s2")
    print(f"Trunk gyro:     [{imu[3]:+.3f} {imu[4]:+.3f} {imu[5]:+.3f}] rad/s")
    print(f"Trunk tilt:     {tilt:.1f} degrees")
    print(f"Height:         {base['height']:.3f} m")
    print(f"Foot contacts:  L={contact[0]:.0f}  R={contact[1]:.0f}")
    fallen, reason = is_fallen(model, data)
    print(f"Status:         {'FALLEN ('+reason+')' if fallen else 'UPRIGHT'}")
    print("=" * 55)


if __name__ == '__main__':
    print("Testing sensors.py...")
    model = mujoco.MjModel.from_xml_path('unitree_g1/scene.xml')
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.78
    mujoco.mj_forward(model, data)

    print_sensor_report(model, data)

    obs_s = build_student_observation(model, data)
    obs_t = build_teacher_observation(model, data)
    print(f"\nbuild_student_observation: shape={obs_s.shape}  dtype={obs_s.dtype}")
    print(f"build_teacher_observation: shape={obs_t.shape}  dtype={obs_t.dtype}")
    print("\nsensors.py OK")
