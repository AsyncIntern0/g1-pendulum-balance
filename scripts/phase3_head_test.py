#import phase3_pose_design as p3
#import generate_fall_dataset_final as p1

#model = p3.load_instrumented_model(r'C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml')
#scenario = p1.SCENARIOS[0]
#mag = p3.fall_magnitude(scenario[2])
#impact_ids = p3.impact_body_ids(model)

#stand = model.key_ctrl[0].copy()
#r = p3.run_protected_trial(model, scenario, mag, 0, 0.0, stand, 0.3, impact_ids)
#print(r)

"""
import numpy as np
import mujoco
import phase3_pose_design as p3
import generate_fall_dataset_final as p1

model = p3.load_instrumented_model(r'C:/Users/Asyncronix/Downloads/Asyncronix_Intern/g1-pendulum-balance/unitree_g1/g1_pendulum.xml')
scenario = p1.SCENARIOS[0]
mag = p3.fall_magnitude(scenario[2])

d = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, d, 0)
stand_ctrl = model.key_ctrl[0].copy()
d.ctrl[:] = stand_ctrl
bob_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pendulum_bob")

for _ in range(int(p1.SETTLE_TIME / model.opt.timestep)):
    d.ctrl[:] = stand_ctrl
    mujoco.mj_step(model, d)

builder = p1.SCENARIO_BUILDERS[scenario[2]]
disturb_fn = builder(model, mag, direction=0)

heights, times = [], []
for step in range(int(2.0 / model.opt.timestep)):  # run a full 2s, ignore fall/stable stopping
    d.ctrl[:] = stand_ctrl
    disturb_fn(model, d, step * model.opt.timestep)
    mujoco.mj_step(model, d)
    heights.append(d.xpos[bob_id, 2])
    times.append(step * model.opt.timestep)

heights = np.array(heights)
print("min bob height:", heights.min(), "at t =", times[int(heights.argmin())])
print("bob height at t=0.45s post-disturbance:", heights[int(0.45/model.opt.timestep)])
print("bob height at end (t=2.0s):", heights[-1]) """


# import phase3_pose_design as p3
# import generate_fall_dataset_final as p1

# model = p3.load_instrumented_model(r'C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml')

# for sid in [3, 4]:
#     scenario = p1.SCENARIOS[sid - 1]
#     mag = p3.fall_magnitude(scenario[2])
#     t = p3._find_time_to_impact(model, scenario, mag, scenario[3], 0.0)
#     print(f"scenario {sid} (dir={scenario[3]}): mag={mag:.1f}N -> t_impact={t}")

#     # also check whether it CAN fall at all, at the calibration ceiling
#     ceiling = p1.CALIBRATION_MAX_MAGNITUDE["push"]
#     t_ceiling = p3._find_time_to_impact(model, scenario, ceiling, scenario[3], 0.0)
#     print(f"  at ceiling ({ceiling}N) -> t_impact={t_ceiling}")

import mujoco
import phase3_pose_design as p3
import generate_fall_dataset_final as p1

model = p3.load_instrumented_model(r'C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml')
scenario = p1.SCENARIOS[11]  # actuator_stuck
mag = p3.fall_magnitude(scenario[2])

d = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, d, 0)
stand_ctrl = model.key_ctrl[0].copy()
d.ctrl[:] = stand_ctrl
ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
foot_ids = p1.get_foot_geom_ids(model)

for _ in range(int(p1.SETTLE_TIME / model.opt.timestep)):
    d.ctrl[:] = stand_ctrl
    mujoco.mj_step(model, d)
baseline_contacts = p1.snapshot_contacts(model, d, ground_id)
print("baseline contact geoms:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in baseline_contacts])
print("excluded 'foot' geoms:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in foot_ids])

for step in range(200):  # 1.0s, no disturbance at all -- just hold 'stand'
    d.ctrl[:] = stand_ctrl
    mujoco.mj_step(model, d)
    current = p1.snapshot_contacts(model, d, ground_id)
    new_bad = (current - baseline_contacts) - foot_ids
    if new_bad:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in new_bad]
        print(f"t={step*model.opt.timestep:.3f}s -- NEW CONTACT: {names}")
        break
else:
    print("no spurious contact in 1.0s of pure standing")