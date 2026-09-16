import mujoco
import numpy as np
import old_files.phase3_pose_design as p3
import generate_fall_dataset_final as p1

model = p3.load_instrumented_model(r'C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml')
snapshot = p3.snapshot_model_state(model)
scenario = p1.SCENARIOS[10]  # actuator_fault
mag = p3.fall_magnitude(scenario[2])
impact_ids = p3.impact_body_ids(model)

# Use the pose the optimizer actually picked for this scenario/merge group --
# paste in raw_results[10]["pose"] from your run, or just test the 'stand'
# pose first to see if the SAME thing happens with no intervention at all:
pose = model.key_ctrl[0].copy()

p3.restore_model_state(model, snapshot)
d = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, d, 0)
stand_ctrl = model.key_ctrl[0].copy()
d.ctrl[:] = stand_ctrl
ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
foot_ids = p1.get_foot_geom_ids(model)
dt = model.opt.timestep
for _ in range(int(p1.SETTLE_TIME / dt)):
    d.ctrl[:] = stand_ctrl
    mujoco.mj_step(model, d)
baseline_contacts = p1.snapshot_contacts(model, d, ground_id)

builder = p1.SCENARIO_BUILDERS[scenario[2]]
disturb_fn = builder(model, mag, direction=None)

pelvis_heights = []
for step in range(int(4.0 / dt)):
    d.ctrl[:] = pose  # command the corrective pose immediately, matching trigger_t_rel=0
    disturb_fn(model, d, step * dt)
    mujoco.mj_step(model, d)
    pelvis_heights.append(d.qpos[2])
    current = p1.snapshot_contacts(model, d, ground_id)
    new_bad = (current - baseline_contacts) - foot_ids
    if new_bad:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) for g in new_bad]
        print(f"t={step*dt:.3f}s first new contact: {names}, pelvis_z={d.qpos[2]:.3f}")
        break

print("pelvis height 0.5s later:", pelvis_heights[-1] if len(pelvis_heights) < int(4.0/dt) else "ran full 4s")
# keep stepping a bit further to see if it recovers or stays down
for _ in range(250):  # +0.5s more
    d.ctrl[:] = pose
    disturb_fn(model, d, len(pelvis_heights) * dt)
    mujoco.mj_step(model, d)
    pelvis_heights.append(d.qpos[2])
print("pelvis height trace (last 10):", [round(h, 3) for h in pelvis_heights[-10:]])