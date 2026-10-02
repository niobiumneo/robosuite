"""Declarative overlay for the TIAGo Pro MJCF package and the robosuite derivation.

Everything that is a *decision* (names, classes, gains, collision policy, home / park poses, camera
parameters, gripper recipe) lives here; the machinery lives in ``convert_tiago_pro.py``.
Numbers marked [PAL] are taken from the PAL Robotics repositories (mj_tags.xacro files, URDF limits);
numbers marked [CHOSEN] are decisions of this port and are documented in the package README.
"""

# --------------------------------------------------------------------------------------------------
# Stage 1: xacro -> URDF
# --------------------------------------------------------------------------------------------------
MUJOCO_VERSION = "3.5.0"  # the importer / mj_saveLastXML output is version sensitive; pinned to robomimic's env

# xacro arguments pinned for reproducibility (everything else keeps its PAL default)
XACRO_ARGS = {
    "camera_model": "realsense-d435",  # PAL default d435i points at a mesh that does not exist in the clone
    "ft_sensor_left": "no-ft-sensor",
    "ft_sensor_right": "no-ft-sensor",
    "end_effector_left": "pal-pro-gripper",
    "end_effector_right": "pal-pro-gripper",
    "end_effector_teleop_left": "no-end-effector",
    "end_effector_teleop_right": "no-end-effector",
    "tool_changer_left": "True",
    "tool_changer_right": "True",
    "arm_type_left": "tiago-pro",
    "arm_type_right": "tiago-pro",
    "wrist_model_left": "spherical-wrist",
    "wrist_model_right": "spherical-wrist",
    "has_wrist_camera": "False",
    "has_teleop_arms": "False",
    "calibration_tool": "False",
    "limits_v2": "False",
}
URDF_TOP = "robots/tiago_pro.urdf.xacro"  # relative to the tiago_pro_description package

# package name -> (clone directory under --pal-root, package sub-directory)
PAL_PACKAGES = {
    "tiago_pro_description": ("tiago_pro_robot", "tiago_pro_description"),
    "pal_sea_arm_description": ("pal_sea_arm", "pal_sea_arm_description"),
    "omni_base_description": ("omni_base_robot", "omni_base_description"),
    "tiago_pro_head_description": ("tiago_pro_head_robot", "tiago_pro_head_description"),
    "pal_urdf_utils": ("pal_urdf_utils", None),
    "pal_pro_gripper_description": ("pal_pro_gripper", "pal_pro_gripper_description"),
}
GIT_CLONES = ["tiago_pro_robot", "pal_sea_arm", "omni_base_robot", "tiago_pro_head_robot", "pal_urdf_utils",
              "pal_pro_gripper"]
GRIPPER_LICENSE = "pal_pro_gripper/LICENSE.txt"  # Apache-2.0, byte identical to menagerie's pal_tiago_dual/LICENSE

# URDF sanity counts after stripping gazebo/ros2_control/transmission (asserted)
EXPECTED_URDF_LINKS = 79
EXPECTED_URDF_JOINTS = 78

# package directory (category) of the meshes, keyed by the package they come from
MESH_CATEGORY = {
    "omni_base_description": {"meshes/base": "base", "meshes/wheels": "wheels"},
    "tiago_pro_description": {"meshes/torso": "torso"},
    "tiago_pro_head_description": {"meshes": "head"},
    "pal_sea_arm_description": {"meshes/arm_tiago_pro": "arm"},
    "pal_pro_gripper_description": {"meshes": "gripper"},
}
# third-party meshes that are NOT redistributed (policy: license unknown / not Apache-2.0); their geoms are dropped
EXCLUDED_MESHES = {"sick_tim551"}

# --------------------------------------------------------------------------------------------------
# Stage 2: package
# --------------------------------------------------------------------------------------------------
MODEL_NAME = "tiago_pro"
PACKAGE_DIRNAME = "pal_tiago_pro"
ROOT_BODY = "base_footprint"
ROOT_Z = 0.002  # metres above the floor at z=0 (wheels touch at 0)
FREEJOINT = "reference"
MESH_MAXHULLVERT = 64

# collision policy ---------------------------------------------------------------------------------
# bodies that keep their collision geoms (everything else: visual only)
COLLISION_BODIES = {
    "base_link",  # 3 boxes
    "wheel_front_right_link", "wheel_front_left_link", "wheel_rear_right_link", "wheel_rear_left_link",
    "torso_base_link", "torso_lift_link", "torso_head_link", "head_1_link", "head_2_link",
    *[f"arm_{s}_{i}_link" for s in ("left", "right") for i in range(1, 8)],
    *[f"gripper_{s}_{n}" for s in ("left", "right")
      for n in ("base_link", "inner_finger_left_link", "inner_finger_right_link", "outer_finger_left_link",
                "outer_finger_right_link", "fingertip_left_link", "fingertip_right_link")],
}
# individual collision meshes that are dropped even on a kept body (non-convex / behind the robot)
DROP_COLLISION_MESHES = {"laptop_tray_collision"}

# finger contact parameters copied from robosuite's Panda gripper (panda_gripper.xml) [CHOSEN]
FINGER_GEOM = {"friction": "1 0.005 0.0001", "condim": "4", "solref": "0.02 1"}
PAD_GEOM = {"friction": "2 0.05 0.0001", "condim": "4", "solref": "0.01 0.5"}

# joint classes: (damping, frictionloss) are the URDF values; classes only factor them
JOINT_CLASSES = {
    "arm": {"damping": "1", "frictionloss": "1"},
    "wheel": {"damping": "1", "frictionloss": "2"},
    "head": {"damping": "0.5", "frictionloss": "1"},
    "torso": {"damping": "1000"},
}
ARM_JOINT_PREFIX = "arm_"
# gripper joints: PAL's own MuJoCo recipe (pal_pro_gripper_description/mujoco/mj_tags.xacro)
GRIPPER_FINGER_FORCE = 8.0  # [PAL] actuatorfrcrange on every gripper joint
GRIPPER_FINGER_DAMPING = 1.5  # [PAL] modify_element on gripper_finger_joint
GRIPPER_KP = 100  # [PAL] position actuator
GRIPPER_FINGER_RANGE = (0.0, 0.07)  # [PAL]; 0.0 = CLOSED, 0.07 = OPEN (tiago_pro_motions yaml)
GRIPPER_OPEN, GRIPPER_CLOSED = 0.07, 0.0
GRIPPER_EQ = dict(solref="0.005 1", solimp="0.95 0.99 0.001")  # [PAL]
# PAL's closed-loop `connect` constraints are NOT emitted by default: their anchors are only consistent near the closed
# pose and fight the mimic joint equalities (up to ~4 deg / 2 mm at the fingertip when open). The URDF kinematics are
# fully determined by the mimic equalities. Set True to reproduce PAL's recipe verbatim.
GRIPPER_CONNECT = False
GRIPPER_CONNECT_ANCHOR = {"right": "0.014 -0.0015 0", "left": "0.014 -0.001 0"}  # [PAL] anchor_r / anchor_l
# gripper frame: z = approach (PAL's grasping_frame has x = approach, we use z like robosuite's grip_site)
GRASP_SITE_POS = (0.0, 0.0, 0.157157)  # [PAL] gripper_*_grasping_frame_joint

# arm actuator gains of the position variant [PAL pal_sea_arm mj_tags.xacro]
ARM_KP = [1000, 1000, 1000, 500, 1000, 1000, 500]
ARM_KV = [50, 60, 50, 20, 20, 20, 20]
HEAD_KP = 2000  # [PAL]
TORSO_KP = 50000  # [CHOSEN] (as menagerie pal_tiago_dual)
WHEEL_KV, WHEEL_CTRLRANGE = 500, 100  # [PAL]
ARM_VELOCITY_KV = 100  # [CHOSEN] velocity variant

# contact excludes: parent/child pairs are automatic, these are the extra always-touching pairs ---------
CONTACT_EXCLUDES = [
    ("torso_base_link", "torso_head_link"),
    ("torso_base_link", "torso_lift_link"),
    ("torso_base_link", "base_link"),
    *[("torso_base_link", f"arm_{s}_1_link") for s in ("left", "right")],
    *[("torso_lift_link", f"arm_{s}_1_link") for s in ("left", "right")],
    *[("torso_head_link", f"arm_{s}_1_link") for s in ("left", "right")],
    *[(f"arm_{s}_5_link", f"arm_{s}_7_link") for s in ("left", "right")],
    *[(f"arm_{s}_7_link", f"gripper_{s}_base_link") for s in ("left", "right")],
    ("base_link", "wheel_front_right_link"), ("base_link", "wheel_front_left_link"),
    ("base_link", "wheel_rear_right_link"), ("base_link", "wheel_rear_left_link"),
]
# gripper-internal pairs from PAL's mj_gripper_contact + the base/finger pairs of the four bar
GRIPPER_CONTACT_EXCLUDES = [
    ("inner_finger_{s}_link", "outer_finger_{s}_link"),
    ("fingertip_{s}_link", "outer_finger_{s}_link"),
    ("base_link", "inner_finger_{s}_link"),
    ("base_link", "fingertip_{s}_link"),
    ("inner_finger_left_link", "inner_finger_right_link"),
    ("fingertip_left_link", "fingertip_right_link"),
    ("outer_finger_left_link", "outer_finger_right_link"),
]

# keyframe `home` (stowed / PAL home poses, manipulation init lives in robosuite) ------------------------
HOME_TORSO = 0.1
HOME_HEAD = (0.0, 0.0)
HOME_ARM = {
    "left": [0.36, -1.83, 0.47, -2.35, 0.0, -1.2, 0.0],  # [PAL] tiago_pro `home` motion
    "right": [-0.36, -1.83, -0.47, -2.35, 0.0, -1.2, 0.0],
}

# head camera (PAL head.urdf.xacro suggests fovy 42 for a 848x480 stream; RGB FOV of a D435 is ~69x42 deg)
HEAD_CAMERA_FRAME = "head_front_camera_color_eye_frame"  # optical frame (z forward, x right, y down)
HEAD_CAMERA_FOVY = 42

# --------------------------------------------------------------------------------------------------
# Stage 3: robosuite
# --------------------------------------------------------------------------------------------------
ROBOT_RIGHT_MODEL = "tiago_pro_right"
# dedicated left-arm park pose (see find_park_pose.py) -- NOT PAL's stowed pose, which sits in front of the base
LEFT_PARK = [0.83, -2.203, -1.766, -1.753, -1.543, -1.393, -0.46]  # found by find_park_pose.py
PARK_TORSO = 0.35  # baked torso lift = joint upper limit [CHOSEN by tools/tiago_pro/find_init_qpos.py scan: best reach of every CAMI task]
PARK_HEAD = (0.0, -0.5)  # head_2 < 0 looks down (robotview sees the table)
PARK_LEFT_GRIPPER = GRIPPER_CLOSED

# robosuite wrist camera (CAMI-compat parameter, not a hardware replica): on the approach axis side, fovy like Panda
EYE_IN_HAND_FOVY = 75
EYE_IN_HAND_POS = (0.05, 0.0, 0.105)  # in the right_hand (= gripper base) frame: beside the housing (+-0.038), just behind the finger tips so both finger tips are in view
EYE_IN_HAND_TILT_DEG = -12.0  # viewing direction tilted toward the tool axis so the finger tips and the grasp point are in view
