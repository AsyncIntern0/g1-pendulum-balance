import mujoco
import numpy as np
import phase3_pose_design as p3
import generate_fall_dataset_final as p1

model = p3.load_instrumented_model(r'C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml') # run on your machine
# DEBUG: Check collision configuration


# ===== TEST 1: CHECK COMPILED GEOM PROPERTIES =====
print("\n========== TEST 1 ==========")

print("GROUND")
print("type =", model.geom_type[0])
print("body =", model.geom_bodyid[0])
print("pos  =", model.geom_pos[0])
print("size =", model.geom_size[0])

print("\nBOB SPHERE (geom 6)")
print("type =", model.geom_type[6])
print("body =", model.geom_bodyid[6])
print("pos  =", model.geom_pos[6])
print("size =", model.geom_size[6])
print("contype =", model.geom_contype[6])
print("conaffinity =", model.geom_conaffinity[6])
print("rbound =", model.geom_rbound[6])

print("============================\n")

data = mujoco.MjData(model)
print("\n=== COLLISION CONFIGURATION ===")

ground_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
)

print(
    f"GROUND: geom_id={ground_id}, "
    f"contype={model.geom_contype[ground_id]}, "
    f"conaffinity={model.geom_conaffinity[ground_id]}"
)

bob_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "pendulum_bob"
)

for gid in range(model.ngeom):
    if model.geom_bodyid[gid] == bob_id:
        geom_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, gid
        )

        print(
            f"BOB: geom_id={gid}, name={geom_name}, "
            f"contype={model.geom_contype[gid]}, "
            f"conaffinity={model.geom_conaffinity[gid]}, "
            f"size={model.geom_size[gid]}"
        )

print("===============================\n")
snapshot = p3.snapshot_model_state(model)
scenario = p1.SCENARIOS[0]  # push forward
mag = p3.fall_magnitude(scenario[2])

pose = np.array([-0.7849049094681303, 0.6318596211979245, 0.3968818081802506,
                  -0.15137016520788127, -0.5827137269844445, 0.30380665754175357,
                  -0.327160835934178, -0.3468999448456891, 0.6620915029969643,
                  -0.518204608331951, 0.273617178617432, 0.750431991476791,
                  0.274997452650491, -0.29725087795566696])

bob_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pendulum_bob")

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

trigger_t_rel = max(0.0, t_impact - 0.3)
builder = p1.SCENARIO_BUILDERS[scenario[2]]
disturb_fn = builder(model, mag, direction=scenario[3])

fell_step = None
for step in range(int(4.0 / dt)):
    t_rel = step * dt
    if t_rel >= trigger_t_rel:
        d.ctrl[:] = pose
    else:
        d.ctrl[:] = stand_ctrl
    disturb_fn(model, d, t_rel)
    mujoco.mj_step(model, d)
    bob_geom_id = 6

    ground_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
    )

    for c in range(d.ncon):
        con = d.contact[c]

        if bob_geom_id in (con.geom1, con.geom2):
            other_geom = con.geom2 if con.geom1 == bob_geom_id else con.geom1

            other_body = model.geom_bodyid[other_geom]

            other_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                other_body
            )

            print(
                f"BOB CONTACT: t={t_rel:.3f}, "
                f"other_body={other_name}, "
                f"other_geom={other_geom}"
            )
    # DEBUG: monitor bob position and contact count
    if step % 50 == 0:
        bob_geom_id = 6

        bob_pos = d.geom_xpos[bob_geom_id].copy()

        print(
            f"t={t_rel:.3f}s "
            f"bob_z={bob_pos[2]:.3f} "
            f"ncon={d.ncon}"
        )

    for c in range(d.ncon):
        con = d.contact[c]

        g1 = con.geom1
        g2 = con.geom2

        b1 = model.geom_bodyid[g1]
        b2 = model.geom_bodyid[g2]

        g1_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, g1
        )
        g2_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, g2
        )

        b1_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, b1
        )
        b2_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, b2
        )

        force6 = np.zeros(6)
        mujoco.mj_contactForce(model, d, c, force6)

        force = float(np.linalg.norm(force6[:3]))

        if 0.8 <= t_rel <= 1.5:
            print(
                f"t={t_rel:.3f} "
                f"contact: "
                f"geom1='{g1_name}' body1='{b1_name}' "
                f"<-> "
                f"geom2='{g2_name}' body2='{b2_name}' "
                f"force={force:.1f}N"
            )

    current = p1.snapshot_contacts(model, d, ground_id)
    new_bad = (current - baseline_contacts) - foot_ids
    if new_bad and fell_step is None:
        fell_step = step
    if fell_step is not None and (step - fell_step) * dt >= 1.5:
        break
