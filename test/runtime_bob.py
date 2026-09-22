import mujoco
import numpy as np

XML_PATH = r"C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

ground_id = 0
bob_id = 6

print("Initial collision settings:")
print("Ground:", model.geom_contype[ground_id], model.geom_conaffinity[ground_id])
print("Bob   :", model.geom_contype[bob_id], model.geom_conaffinity[bob_id])


# ---------------------------------------------------------
# STEP 1: Enable bob collision at runtime
# ---------------------------------------------------------

model.geom_contype[bob_id] = 1
model.geom_conaffinity[bob_id] = 1

print("\nAfter changing bob collision settings:")
print("Ground:", model.geom_contype[ground_id], model.geom_conaffinity[ground_id])
print("Bob   :", model.geom_contype[bob_id], model.geom_conaffinity[bob_id])


# ---------------------------------------------------------
# STEP 2: Refresh MuJoCo model constants
# ---------------------------------------------------------

mujoco.mj_setConst(model,data)

print("\nCalled mj_setConst(model)")


# ---------------------------------------------------------
# STEP 3: Put robot into stand pose
# ---------------------------------------------------------

mujoco.mj_resetDataKeyframe(model, data, 0)


# ---------------------------------------------------------
# STEP 4: Move entire robot DOWN
# ---------------------------------------------------------

floating_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "floating_base_joint"
)

qadr = model.jnt_qposadr[floating_joint]

original_z = data.qpos[qadr + 2]

# Move down by 1.10 m
data.qpos[qadr + 2] = original_z - 1.10


mujoco.mj_forward(model, data)
# ---------------------------------------------------------
# TEST 8: DIRECT GEOMETRIC DISTANCE
# ---------------------------------------------------------

print("\n========== TEST 8: GEOM DISTANCE ==========")

ground_id = 0
bob_id = 6

fromto = np.zeros(6)

distance = mujoco.mj_geomDistance(
    model,
    data,
    ground_id,
    bob_id,
    1.0,
    fromto
)

print("Distance     :", distance)
print("Bob position :", data.geom_xpos[bob_id])
print("Bob radius   :", model.geom_size[bob_id][0])

print("\nFrom-to:")
print(fromto)

if distance < 0:
    print("\n>>> GEOMETRY SAYS: BOB IS PENETRATING GROUND")
elif distance == 0:
    print("\n>>> GEOMETRY SAYS: BOB IS TOUCHING GROUND")
else:
    print("\n>>> GEOMETRY SAYS: BOB IS SEPARATED FROM GROUND")

print("\n==========================================")
# ---------------------------------------------------------
# TEST 4: Check ALL contacts involving bob sphere (geom 6)
# ---------------------------------------------------------

# print("\n========== TEST 4: BOB CONTACTS ==========")

# bob_id = 6

# print("Bob geom ID:", bob_id)
# print("Bob position:", data.geom_xpos[bob_id])
# print("Total contacts:", data.ncon)

# found_bob_contact = False

# for i in range(data.ncon):

#     c = data.contact[i]

#     # Check whether geom 6 is involved
#     if c.geom1 == bob_id or c.geom2 == bob_id:

#         found_bob_contact = True

#         other_geom = c.geom2 if c.geom1 == bob_id else c.geom1

#         other_name = mujoco.mj_id2name(
#             model,
#             mujoco.mjtObj.mjOBJ_GEOM,
#             other_geom
#         )

#         bob_name = mujoco.mj_id2name(
#             model,
#             mujoco.mjtObj.mjOBJ_GEOM,
#             bob_id
#         )

#         print(
#             f"\nContact {i}:"
#         )
#         print("  Bob geom       :", bob_id, bob_name)
#         print("  Other geom     :", other_geom, other_name)
#         print("  Other body     :", model.geom_bodyid[other_geom])

#         force = np.zeros(6)
#         mujoco.mj_contactForce(model, data, i, force)

#         print("  Contact force  :", np.linalg.norm(force[:3]), "N")


# if not found_bob_contact:
#     print("\n>>> GEOM 6 HAS NO CONTACT WITH ANY GEOM <<<")
# else:
#     print("\n>>> GEOM 6 IS PARTICIPATING IN CONTACTS <<<")

# print("\n===========================================")
# print("TEST 4 COMPLETE")
# print("===========================================")
# print("\n========== TEST 5: BOB GEOM DETAILS ==========")

# bob = 6

# print("Geom ID          :", bob)
# print("Geom type        :", model.geom_type[bob])
# print("Geom body ID     :", model.geom_bodyid[bob])
# print("Body name        :",
#       mujoco.mj_id2name(
#           model,
#           mujoco.mjtObj.mjOBJ_BODY,
#           model.geom_bodyid[bob]
#       ))

# print("Geom position     :", model.geom_pos[bob])
# print("Geom size         :", model.geom_size[bob])
# print("Geom contype      :", model.geom_contype[bob])
# print("Geom conaffinity  :", model.geom_conaffinity[bob])
# print("Geom group        :", model.geom_group[bob])
# print("Geom priority     :", model.geom_priority[bob])
# print("Geom condim       :", model.geom_condim[bob])

# print("\nBody information:")

# body = model.geom_bodyid[bob]

# print("Body ID           :", body)
# print("Parent body ID    :", model.body_parentid[body])
# print("Parent body name  :",
#       mujoco.mj_id2name(
#           model,
#           mujoco.mjtObj.mjOBJ_BODY,
#           model.body_parentid[body]
#       ))

# print("Number of joints  :", model.body_jntnum[body])
# print("Number of DOFs    :", model.body_dofnum[body])
# print("Number of geoms   :", model.body_geomnum[body])

# print("\nWorld position of bob:")
# print(data.geom_xpos[bob])

# print("\n==============================================")
print("\n========== COLLISION FILTER TEST ==========")

g1 = 0
g2 = 6

print("Ground geom:", g1)
print("Bob geom   :", g2)

print("\nGround:")
print("  contype     =", model.geom_contype[g1])
print("  conaffinity =", model.geom_conaffinity[g1])
print("  bodyid      =", model.geom_bodyid[g1])

print("\nBob:")
print("  contype     =", model.geom_contype[g2])
print("  conaffinity =", model.geom_conaffinity[g2])
print("  bodyid      =", model.geom_bodyid[g2])

# MuJoCo collision filtering rule:
#
# Pair can collide if:
#
# (contype1 & conaffinity2) != 0
# OR
# (contype2 & conaffinity1) != 0

rule1 = (
    model.geom_contype[g1] &
    model.geom_conaffinity[g2]
)

rule2 = (
    model.geom_contype[g2] &
    model.geom_conaffinity[g1]
)

print("\nFiltering calculation:")
print("ground.contype & bob.conaffinity =", rule1)
print("bob.contype & ground.conaffinity =", rule2)

if rule1 != 0 or rule2 != 0:
    print("\n>>> FILTER ALLOWS COLLISION")
else:
    print("\n>>> FILTER BLOCKS COLLISION")

print("==========================================")


# ---------------------------------------------------------
# STEP 5: Check bob position
# ---------------------------------------------------------

bob_pos = data.geom_xpos[bob_id]

print("\nBob position:")
print(bob_pos)

print("Bob radius:", model.geom_size[bob_id][0])

print("\nNumber of contacts:", data.ncon)


# ---------------------------------------------------------
# STEP 6: Look specifically for ground <-> bob
# ---------------------------------------------------------

found = False

for i in range(data.ncon):

    c = data.contact[i]

    print(
        f"Contact {i}: "
        f"geom1={c.geom1}, "
        f"geom2={c.geom2}"
    )

    if (c.geom1 == ground_id and c.geom2 == bob_id) or \
       (c.geom1 == bob_id and c.geom2 == ground_id):

        found = True

        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)

        print("\n================================")
        print("GROUND <-> BOB CONTACT FOUND!")
        print("Contact force:", np.linalg.norm(force[:3]), "N")
        print("================================")


if not found:
    print("\n================================")
    print("NO GROUND <-> BOB CONTACT")
    print("================================")