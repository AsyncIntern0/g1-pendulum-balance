import mujoco
import numpy as np
import phase3_pose_design as p3
import generate_fall_dataset_final as p1

model = p3.load_instrumented_model(r'C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml')  # <-- run this on your machine with the real model
snapshot = p3.snapshot_model_state(model)
scenario = p1.SCENARIOS[10]  # actuator_fault
mag = p3.fall_magnitude(scenario[2])

pose = np.array([-0.5119414919605179, -0.2718817933867951, -0.43890624222365404,
                  0.42510066184448325, -0.6673941762162667, 0.00731639355977211,
                  -0.3783869870824381, -0.06568097984541985, 1.7633757280086306,
                  0.5621762676867682, -0.4612704040343787, 2.0697239680253823,
                  0.17244088442369288, 0.34460622491597])

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

# Also try WITHOUT the fault at all, just commanding this pose cold, to
# isolate whether the pose alone (no disturbance) causes ground contact.
found = False
for step in range(int(1.0 / dt)):  # only need ~1s to see if it touches down
    d.ctrl[:] = pose
    disturb_fn(model, d, step * dt)   # comment this line out to test pose alone, no fault
    mujoco.mj_step(model, d)
    current = p1.snapshot_contacts(model, d, ground_id)
    new_bad = (current - baseline_contacts) - foot_ids
    if new_bad:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) for g in new_bad]
        print(f"t={step*dt:.3f}s first new contact: {names}, pelvis_z={d.qpos[2]:.3f}")
        found = True
        break
if not found:
    print("no new contact in 1.0s with this pose + actuator_fault")
print("final pelvis_z:", d.qpos[2])
