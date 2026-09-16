# import mujoco
# import numpy as np

# XML_PATH = r"C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml"

# print("\n========== TEST 10: FORCED BOB COLLISION ==========")

# # ---------------------------------------------------------
# # 1. Load
# # ---------------------------------------------------------
# model = mujoco.MjModel.from_xml_path(XML_PATH)
# data = mujoco.MjData(model)

# ground_id = mujoco.mj_name2id(
#     model,
#     mujoco.mjtObj.mjOBJ_GEOM,
#     "ground"
# )

# bob_body_id = mujoco.mj_name2id(
#     model,
#     mujoco.mjtObj.mjOBJ_BODY,
#     "pendulum_bob"
# )

# bob_geoms = np.where(model.geom_bodyid == bob_body_id)[0]

# sphere_id = None

# for gid in bob_geoms:
#     if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
#         sphere_id = gid

# print("Ground ID :", ground_id)
# print("Bob body  :", bob_body_id)
# print("Bob geoms :", bob_geoms)
# print("Sphere ID :", sphere_id)

# # ---------------------------------------------------------
# # 2. BEFORE modification
# # ---------------------------------------------------------
# print("\n========== BEFORE COLLISION ENABLE ==========")

# print(
#     "Sphere contype    :",
#     model.geom_contype[sphere_id]
# )

# print(
#     "Sphere conaffinity:",
#     model.geom_conaffinity[sphere_id]
# )

# # ---------------------------------------------------------
# # 3. ENABLE COLLISION
# # ---------------------------------------------------------
# for gid in bob_geoms:
#     model.geom_contype[gid] = 1
#     model.geom_conaffinity[gid] = 1

# # ---------------------------------------------------------
# # 4. Verify AFTER modification
# # ---------------------------------------------------------
# print("\n========== AFTER COLLISION ENABLE ==========")

# print(
#     "Sphere contype    :",
#     model.geom_contype[sphere_id]
# )

# print(
#     "Sphere conaffinity:",
#     model.geom_conaffinity[sphere_id]
# )

# print(
#     "Ground contype    :",
#     model.geom_contype[ground_id]
# )

# print(
#     "Ground conaffinity:",
#     model.geom_conaffinity[ground_id]
# )

# # ---------------------------------------------------------
# # 5. Reset model state
# # ---------------------------------------------------------
# data.qpos[:] = model.key_qpos[0]
# data.qvel[:] = 0

# # Move whole robot down
# data.qpos[2] -= 1.10

# # ---------------------------------------------------------
# # 6. Forward kinematics
# # ---------------------------------------------------------
# mujoco.mj_forward(model, data)

# print("\n========== POSITION ==========")

# print(
#     "Sphere world position:",
#     data.geom_xpos[sphere_id]
# )

# print(
#     "Sphere radius:",
#     model.geom_size[sphere_id][0]
# )

# # ---------------------------------------------------------
# # 7. Check geometry distance
# # ---------------------------------------------------------
# fromto = np.zeros(6)

# distance = mujoco.mj_geomDistance(
#     model,
#     data,
#     ground_id,
#     sphere_id,
#     1.0,
#     fromto
# )

# print("\n========== GEOMETRY DISTANCE ==========")

# print("Distance:", distance)

# if distance < 0:
#     print(">>> GEOMETRY SAYS: PENETRATING <<<")
# elif distance == 0:
#     print(">>> GEOMETRY SAYS: TOUCHING <<<")
# else:
#     print(">>> GEOMETRY SAYS: SEPARATED <<<")

# # ---------------------------------------------------------
# # 8. Explicit collision generation
# # ---------------------------------------------------------
# print("\n========== mj_collision() ==========")

# mujoco.mj_collision(model, data)

# print("Total contacts:", data.ncon)

# # ---------------------------------------------------------
# # 9. Print EVERY contact
# # ---------------------------------------------------------
# found = False

# for i in range(data.ncon):

#     c = data.contact[i]

#     g1 = c.geom1
#     g2 = c.geom2

#     b1 = model.geom_bodyid[g1]
#     b2 = model.geom_bodyid[g2]

#     print(
#         f"Contact {i}: "
#         f"geom {g1} (body {b1}) <-> "
#         f"geom {g2} (body {b2}) "
#         f"dist={c.dist:.6f}"
#     )

#     if (
#         (g1 == ground_id and g2 == sphere_id)
#         or
#         (g1 == sphere_id and g2 == ground_id)
#     ):
#         found = True

# print("\n========== RESULT ==========")

# if found:
#     print(">>> SUCCESS: GROUND <-> BOB CONTACT GENERATED <<<")
# else:
#     print(">>> FAILURE: NO GROUND <-> BOB CONTACT <<<")

# # ---------------------------------------------------------
# # 10. Final collision masks
# # ---------------------------------------------------------
# print("\n========== FINAL MASKS ==========")

# print(
#     "Sphere:",
#     "contype =", model.geom_contype[sphere_id],
#     "conaffinity =", model.geom_conaffinity[sphere_id]
# )

# print(
#     "Ground:",
#     "contype =", model.geom_contype[ground_id],
#     "conaffinity =", model.geom_conaffinity[ground_id]
# )

# print("\n========== TEST 10 COMPLETE ==========\n")
# 
import mujoco
import numpy as np
import re
import tempfile
import os

XML_PATH = r"C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml"

print("\n========== TEST 12: XML COLLISION VS RUNTIME COLLISION ==========")

# ---------------------------------------------------------
# 1. Read original XML
# ---------------------------------------------------------
with open(XML_PATH, "r", encoding="utf-8") as f:
    xml = f.read()

# ---------------------------------------------------------
# 2. Change ONLY pendulum collision masks in XML
# ---------------------------------------------------------
old = '''contype="0" conaffinity="0"'''

new = '''contype="1" conaffinity="1"'''

count = xml.count(old)

print("Number of contype=0/conaffinity=0 occurrences:", count)

xml_modified = xml.replace(old, new)

# ---------------------------------------------------------
# 3. Save temporary XML
# ---------------------------------------------------------
temp_xml = os.path.join(
    os.path.dirname(XML_PATH),
    "g1_pendulum_TEST12.xml"
)

with open(temp_xml, "w", encoding="utf-8") as f:
    f.write(xml_modified)

print("Temporary XML:", temp_xml)

# ---------------------------------------------------------
# 4. Load MODIFIED XML
# ---------------------------------------------------------
model = mujoco.MjModel.from_xml_path(temp_xml)
data = mujoco.MjData(model)

# ---------------------------------------------------------
# 5. Find IDs
# ---------------------------------------------------------
ground_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "ground"
)

bob_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "pendulum_bob"
)

bob_geoms = np.where(
    model.geom_bodyid == bob_body_id
)[0]

sphere_id = None

for gid in bob_geoms:
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
        sphere_id = gid
        break

print("\n========== COMPILED VALUES ==========")

print("Ground ID:", ground_id)
print("Bob body:", bob_body_id)
print("Bob geoms:", bob_geoms)
print("Sphere ID:", sphere_id)

print(
    "Sphere contype:",
    model.geom_contype[sphere_id]
)

print(
    "Sphere conaffinity:",
    model.geom_conaffinity[sphere_id]
)

print(
    "Ground contype:",
    model.geom_contype[ground_id]
)

print(
    "Ground conaffinity:",
    model.geom_conaffinity[ground_id]
)

# ---------------------------------------------------------
# 6. Put robot into penetrating position
# ---------------------------------------------------------
data.qpos[:] = model.key_qpos[0]
data.qvel[:] = 0

data.qpos[2] -= 1.10

mujoco.mj_forward(model, data)

# ---------------------------------------------------------
# 7. Check sphere
# ---------------------------------------------------------
print("\n========== POSITION ==========")

print(
    "Sphere world position:",
    data.geom_xpos[sphere_id]
)

# ---------------------------------------------------------
# 8. Geometry distance
# ---------------------------------------------------------
fromto = np.zeros(6)

distance = mujoco.mj_geomDistance(
    model,
    data,
    ground_id,
    sphere_id,
    1.0,
    fromto
)

print("\n========== DISTANCE ==========")

print("Distance:", distance)

# ---------------------------------------------------------
# 9. Collision
# ---------------------------------------------------------
print("\n========== COLLISION ==========")

mujoco.mj_collision(model, data)

print("Total contacts:", data.ncon)

found = False

for i in range(data.ncon):

    c = data.contact[i]

    if (
        (c.geom1 == ground_id and c.geom2 == sphere_id)
        or
        (c.geom1 == sphere_id and c.geom2 == ground_id)
    ):

        found = True

        print("\n>>> GROUND <-> BOB CONTACT FOUND <<<")
        print("Contact index:", i)
        print("Distance:", c.dist)
        print("Position:", c.pos)

if not found:
    print("\n>>> NO GROUND <-> BOB CONTACT <<<")

# ---------------------------------------------------------
# 10. Clean up
# ---------------------------------------------------------
try:
    os.remove(temp_xml)
except Exception:
    pass

print("\n========== TEST 12 COMPLETE ==========\n")