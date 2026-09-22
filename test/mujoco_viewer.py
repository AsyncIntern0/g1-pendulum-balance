import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path(
    r"C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml"
)

data = mujoco.MjData(model)

# Reset to standing keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Left hip pitch joint
hip_pitch_joint_id = 9
qadr = model.jnt_qposadr[hip_pitch_joint_id]

# Apply +40 degrees
data.qpos[qadr] += np.radians(40)

mujoco.mj_forward(model, data)

print("Applied +40° to left_hip_pitch_joint")
print("Joint ID:", hip_pitch_joint_id)
print("Current angle:", np.degrees(data.qpos[qadr]))

# Open viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
print("qpos:", np.degrees(data.qpos[qadr]))