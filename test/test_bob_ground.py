import mujoco
import numpy as np

XML_PATH = r"C:\Users\Asyncronix\Downloads\Asyncronix_Intern\g1-pendulum-balance\unitree_g1\g1_pendulum.xml"


# ---------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------
model = mujoco.MjModel.from_xml_path(XML_PATH)

# Find pendulum bob body
bob_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "pendulum_bob"
)

print("bob_body_id =", bob_body_id)


# ---------------------------------------------------------
# 2. Enable collision for ALL bob geoms
# ---------------------------------------------------------
for geom_id in range(model.ngeom):
    if model.geom_bodyid[geom_id] == bob_body_id:
        model.geom_contype[geom_id] = 1
        model.geom_conaffinity[geom_id] = 1

        print(
            "Enabled collision:",
            "geom_id =", geom_id,
            "type =", model.geom_type[geom_id],
            "size =", model.geom_size[geom_id]
        )


# ---------------------------------------------------------
# 3. Create simulation data
# ---------------------------------------------------------
data = mujoco.MjData(model)

# Reset to stand
mujoco.mj_resetDataKeyframe(model, data, 0)

# Forward simulation once
mujoco.mj_forward(model, data)


# ---------------------------------------------------------
# 4. Find bob sphere geom
# ---------------------------------------------------------
bob_sphere_id = 6
ground_id = 0

print("\nGround geom =", ground_id)
print("Bob sphere geom =", bob_sphere_id)


# ---------------------------------------------------------
# 5. Check initial bob position
# ---------------------------------------------------------
print("\nInitial bob position:")
print(data.geom_xpos[bob_sphere_id])


# ---------------------------------------------------------
# 6. Move bob downward manually
# ---------------------------------------------------------
#
# We directly modify the bob body's position.
# This puts the sphere through the ground.
#
# Since bob_body is attached to pendulum joints,
# we modify the body's joint configuration instead.
#

print("\nTesting contacts...")
print("--------------------------------")


# Try several bob heights
# ---------------------------------------------------------
# TEST: Can we move the pendulum bob?
# ---------------------------------------------------------

print("\n========== BOB MOVEMENT TEST ==========")

print("Number of joints:", model.njnt)

for j in range(model.njnt):
    print(
        j,
        "name =", mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            j
        ),
        "type =", model.jnt_type[j],
        "qposadr =", model.jnt_qposadr[j]
    )


print("\nInitial bob position:")
mujoco.mj_forward(model, data)
print(data.geom_xpos[6])


# Try every joint that belongs to the pendulum body
# ---------------------------------------------------------
# TEST 2: Sweep pendulum pitch and check bob-ground contact
# ---------------------------------------------------------

# print("\n========== TEST 2: PITCH SWEEP ==========")

# ground_id = 0
# bob_sphere_id = 6

# # Find pendulum pitch joint
# pitch_joint = mujoco.mj_name2id(
#     model,
#     mujoco.mjtObj.mjOBJ_JOINT,
#     "pendulum_pitch_joint"
# )

# pitch_qadr = model.jnt_qposadr[pitch_joint]

# print("Pitch joint ID  :", pitch_joint)
# print("Pitch qpos addr :", pitch_qadr)

# print("\nAngle(deg)    Bob X       Bob Z       Contact?")
# print("-----------------------------------------------")


# # Sweep from -45 degrees to +45 degrees
# for angle_deg in np.linspace(-45, 45, 19):

#     # Reset model to stand
#     mujoco.mj_resetDataKeyframe(model, data, 0)

#     # Set pendulum pitch angle
#     angle_rad = np.deg2rad(angle_deg)
#     data.qpos[pitch_qadr] = angle_rad

#     # Recalculate geometry and contacts
#     mujoco.mj_forward(model, data)

#     # Bob sphere world position
#     bob_pos = data.geom_xpos[bob_sphere_id]

#     found_contact = False
#     contact_force = 0.0

#     # Check every contact
#     for i in range(data.ncon):

#         contact = data.contact[i]

#         g1 = contact.geom1
#         g2 = contact.geom2

#         # Specifically check:
#         # ground geom 0 <-> bob sphere geom 6
#         if (g1 == ground_id and g2 == bob_sphere_id) or \
#            (g1 == bob_sphere_id and g2 == ground_id):

#             found_contact = True

#             force = np.zeros(6)
#             mujoco.mj_contactForce(model, data, i, force)

#             contact_force = np.linalg.norm(force[:3])

#     print(
#         f"{angle_deg:8.1f}    "
#         f"{bob_pos[0]:8.3f}    "
#         f"{bob_pos[2]:8.3f}    "
#         f"{'YES  force=' + str(round(contact_force, 2)) + 'N' if found_contact else 'NO'}"
#     )


# print("\n==========================================")
# print("TEST 2 COMPLETE")
# print("==========================================")
# ---------------------------------------------------------
# TEST 3: Move entire robot downward and test
#          ground (geom 0) <-> bob sphere (geom 6)
# ---------------------------------------------------------

print("\n========== TEST 3: BOB-GROUND COLLISION ==========")

ground_id = 0
bob_sphere_id = 6

# Floating base joint
floating_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "floating_base_joint"
)

floating_qadr = model.jnt_qposadr[floating_joint]

print("Floating base joint ID  :", floating_joint)
print("Floating base qpos addr :", floating_qadr)

# Reset to normal standing pose
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Store original base position
original_x = data.qpos[floating_qadr + 0]
original_y = data.qpos[floating_qadr + 1]
original_z = data.qpos[floating_qadr + 2]

print("\nOriginal base position:")
print(
    f"x={original_x:.3f}, "
    f"y={original_y:.3f}, "
    f"z={original_z:.3f}"
)

print("\nBase Z shift    Bob Z       Contact?")
print("---------------------------------------")


# Move the whole robot progressively downward
for shift in [0.0, -0.5, -0.8, -1.0, -1.05, -1.10, -1.15, -1.20]:

    # Reset to standing pose
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # Move floating base downward
    data.qpos[floating_qadr + 2] = original_z + shift

    # Recalculate kinematics and contacts
    mujoco.mj_forward(model, data)

    # Bob sphere world position
    bob_pos = data.geom_xpos[bob_sphere_id]
    bob_z = bob_pos[2]

    found_contact = False
    contact_force = 0.0

    # Check all contacts
    for i in range(data.ncon):

        contact = data.contact[i]

        g1 = contact.geom1
        g2 = contact.geom2

        # Specifically check ground geom 0 <-> bob sphere geom 6
        if (g1 == ground_id and g2 == bob_sphere_id) or \
           (g1 == bob_sphere_id and g2 == ground_id):

            found_contact = True

            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)

            contact_force = np.linalg.norm(force[:3])

    if found_contact:
        contact_text = f"YES  force={contact_force:.2f}N"
    else:
        contact_text = "NO"

    print(
        f"{shift:8.2f} m    "
        f"{bob_z:8.3f} m    "
        f"{contact_text}"
    )


print("\n==============================================")
print("TEST 3 COMPLETE")
print("==============================================")