import mujoco
import numpy as np
import phase3_pose_design as p3
import generate_fall_dataset_final as p1

model = p3.load_instrumented_model(r'C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml')  # run on your machine
snapshot = p3.snapshot_model_state(model)
scenario = p1.SCENARIOS[0]  # push forward
mag = p3.fall_magnitude(scenario[2])

pose = np.array([-0.7849049094681303, 0.6318596211979245, 0.3968818081802506,
                  -0.15137016520788127, -0.5827137269844445, 0.30380665754175357,
                  -0.327160835934178, -0.3468999448456891, 0.6620915029969643,
                  -0.518204608331951, 0.273617178617432, 0.750431991476791,
                  0.274997452650491, -0.29725087795566696])

# Find the ACTUAL sphere geom (the mass), not the body origin/pivot.
bob_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pendulum_bob")
sphere_gid = None
for gid in range(model.ngeom):
    if model.geom_bodyid[gid] == bob_id and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
        sphere_gid = gid
print("sphere geom id:", sphere_gid, "local pos:", model.geom_pos[sphere_gid], "radius:", model.geom_size[sphere_gid][0])

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

t_impact = p3._find_time_to_impact(model, scenario, mag, scenario[3], 0.0, snapshot)
p3.restore_model_state(model, snapshot)
print("baseline t_impact:", t_impact)

trigger_t_rel = max(0.0, t_impact - 0.3)
builder = p1.SCENARIO_BUILDERS[scenario[2]]
disturb_fn = builder(model, mag, direction=scenario[3])

fell_step = None
sphere_heights, pivot_heights, times = [], [], []
for step in range(int(4.0 / dt)):
    t_rel = step * dt
    if t_rel >= trigger_t_rel:
        d.ctrl[:] = pose
    else:
        d.ctrl[:] = stand_ctrl
    disturb_fn(model, d, t_rel)
    mujoco.mj_step(model, d)
    sphere_heights.append(d.geom_xpos[sphere_gid, 2])   # <-- the ACTUAL mass
    pivot_heights.append(d.xpos[bob_id, 2])              # <-- what the old script measured
    times.append(t_rel)
    current = p1.snapshot_contacts(model, d, ground_id)
    new_bad = (current - baseline_contacts) - foot_ids
    if new_bad and fell_step is None:
        fell_step = step
        print(f"first fall contact at t={t_rel:.3f}s, sphere_z={d.geom_xpos[sphere_gid,2]:.3f}")
    if fell_step is not None and (step - fell_step) * dt >= 1.5:
        break

sphere_heights = np.array(sphere_heights)
pivot_heights = np.array(pivot_heights)
print(f"SPHERE (real mass) min height: {sphere_heights.min():.3f} at t={times[int(sphere_heights.argmin())]:.3f}s")
print(f"PIVOT (old, wrong measurement) min height: {pivot_heights.min():.3f} at t={times[int(pivot_heights.argmin())]:.3f}s")
print("sphere height every 0.2s:", [round(h,3) for h in sphere_heights[::int(0.2/dt)]])
