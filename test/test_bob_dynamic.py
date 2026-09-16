import mujoco
import numpy as np

XML_PATH = r"C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml"


print("\n========== TEST 9: BOB DYNAMIC / CONTACT STATE ==========")

# ---------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# ---------------------------------------------------------
# 2. Find important IDs
# ---------------------------------------------------------
ground_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "ground"
)

bob_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "pendulum_bob"
)

# If the bob geom itself has no name, find it from the bob body
bob_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "pendulum_bob"
)

print("Ground geom ID :", ground_id)
print("Bob body ID    :", bob_body_id)

# Find geoms belonging to pendulum_bob body
bob_geoms = np.where(model.geom_bodyid == bob_body_id)[0]

print("Bob body geoms :", bob_geoms)

for gid in bob_geoms:
    print(
        f"\nGeom {gid}:"
        f"\n  type       = {model.geom_type[gid]}"
        f"\n  contype    = {model.geom_contype[gid]}"
        f"\n  conaffinity= {model.geom_conaffinity[gid]}"
        f"\n  body       = {model.geom_bodyid[gid]}"
    )

# ---------------------------------------------------------
# 3. Show body / joint information
# ---------------------------------------------------------
print("\n========== BODY / JOINT INFO ==========")

print("Bob body:")
print("  body id     :", bob_body_id)
print("  parent body :", model.body_parentid[bob_body_id])
print("  mass        :", model.body_mass[bob_body_id])
print("  dof address :", model.body_dofadr[bob_body_id])
print("  jnt address :", model.body_jntadr[bob_body_id])

if model.body_jntadr[bob_body_id] >= 0:
    jid = model.body_jntadr[bob_body_id]

    print("  joint id    :", jid)
    print("  joint type  :", model.jnt_type[jid])
    print("  joint axis  :", model.jnt_axis[jid])
    print("  qpos adr    :", model.jnt_qposadr[jid])
    print("  dof adr     :", model.jnt_dofadr[jid])

# ---------------------------------------------------------
# 4. Enable collision
# ---------------------------------------------------------
for gid in bob_geoms:
    model.geom_contype[gid] = 1
    model.geom_conaffinity[gid] = 1

mujoco.mj_setConst(model, data)

# ---------------------------------------------------------
# 5. Put robot into a known pose
# ---------------------------------------------------------
data.qpos[:] = model.key_qpos[0]
data.qvel[:] = 0

# Move entire robot downward so bob penetrates ground
data.qpos[2] -= 1.10

mujoco.mj_forward(model, data)

# ---------------------------------------------------------
# 6. Print bob position
# ---------------------------------------------------------
print("\n========== BOB POSITION ==========")

for gid in bob_geoms:
    print(
        f"Geom {gid} world position:",
        data.geom_xpos[gid]
    )

print("Bob body world position:", data.xpos[bob_body_id])

# ---------------------------------------------------------
# 7. Geometry distance
# ---------------------------------------------------------
print("\n========== GEOMETRY DISTANCE ==========")

fromto = np.zeros(6)

# Use the sphere geom (usually geom 6)
sphere_id = None

for gid in bob_geoms:
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
        sphere_id = gid
        break

if sphere_id is None:
    print("ERROR: No sphere geom found!")
    raise SystemExit

distance = mujoco.mj_geomDistance(
    model,
    data,
    ground_id,
    sphere_id,
    1.0,
    fromto
)

print("Sphere geom ID :", sphere_id)
print("Distance       :", distance)
print("From-to        :", fromto)

# ---------------------------------------------------------
# 8. Generate contacts
# ---------------------------------------------------------
print("\n========== CONTACT GENERATION ==========")

mujoco.mj_collision(model, data)

print("Number of contacts:", data.ncon)

found = False

for i in range(data.ncon):

    c = data.contact[i]

    g1 = c.geom1
    g2 = c.geom2

    if (
        (g1 == ground_id and g2 == sphere_id)
        or
        (g2 == ground_id and g1 == sphere_id)
    ):

        found = True

        print("\n>>> GROUND <-> BOB CONTACT FOUND <<<")
        print("Contact index :", i)
        print("geom1         :", g1)
        print("geom2         :", g2)
        print("distance      :", c.dist)
        print("position      :", c.pos)
        print("normal        :", c.frame[:3])

if not found:
    print("\n>>> NO GROUND <-> BOB CONTACT <<<")

# ---------------------------------------------------------
# 9. Check all contacts involving bob body
# ---------------------------------------------------------
print("\n========== ALL BOB CONTACTS ==========")

bob_contact_count = 0

for i in range(data.ncon):

    c = data.contact[i]

    g1_body = model.geom_bodyid[c.geom1]
    g2_body = model.geom_bodyid[c.geom2]

    if g1_body == bob_body_id or g2_body == bob_body_id:

        bob_contact_count += 1

        print(
            f"Contact {i}: "
            f"geom {c.geom1} <-> geom {c.geom2}, "
            f"dist={c.dist:.6f}"
        )

print("\nTotal contacts involving bob body:", bob_contact_count)

print("\n========== TEST 9 COMPLETE ==========\n")