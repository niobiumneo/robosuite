# tools/tiago_pro: TIAGo Pro -> MuJoCo menagerie package -> robosuite

Developer tooling (not packaged). Converts the PAL Robotics TIAGo Pro xacro description into

1. `robosuite/models/assets/robots/tiago_pro/pal_tiago_pro/`: a self-contained, menagerie-conformant MJCF package
   (LICENSE Apache-2.0, README, CHANGELOG, `tiago_pro*.xml`, `scene_*.xml`, `assets/`), and
2. the robosuite files derived from it: `robots/tiago_pro/robot_right.xml` (right arm only, everything else frozen,
   fully flattened) and `grippers/pal_pro_gripper.xml` (+ `grippers/meshes/pal_pro_gripper/`).

Generated files are never edited by hand.

## Regenerate

Inputs: the cloned PAL repositories (pinned in `sources.lock`: tiago_pro_robot, pal_sea_arm, omni_base_robot,
tiago_pro_head_robot, pal_urdf_utils, pal_pro_gripper), python with `xacro` (2.x), `lxml`, `numpy`, and **mujoco==3.5.0**
(asserted: the URDF importer output is version dependent; 3.5.0 is robomimic's pin).

```bash
PAL=~/pal-robotics            # directory holding the 6 clones
PYTHONPATH=/path/to/mujoco-3.5.0 python tools/tiago_pro/convert_tiago_pro.py --pal-root $PAL        # regenerate
PYTHONPATH=/path/to/mujoco-3.5.0 python tools/tiago_pro/convert_tiago_pro.py --pal-root $PAL --check # byte-diff vs committed
PYTHONPATH=/path/to/mujoco-3.5.0 python tools/tiago_pro/convert_tiago_pro.py --pal-root $PAL --update-lock   # after moving a clone
PYTHONPATH=/path/to/mujoco-3.5.0 python tools/tiago_pro/convert_tiago_pro.py --stage1 --out /tmp/s1  # xacro->URDF->MJCF only
```
(`pip install --target /path/to/mujoco-3.5.0 mujoco==3.5.0` gives the pinned MuJoCo without touching your environment.)
`--format-with` (default `~/google-deepmind/mujoco_menagerie/format_xml.py`) formats XML in menagerie style; the committed
files are formatted, so `--check` needs the same formatter.

Verification (no xacro / PAL clones needed, any MuJoCo >= 3.3):

```bash
python tools/tiago_pro/verify_tiago_pro.py                       # package, robot_right.xml, gripper acceptance checks
tools/tiago_pro/check_menagerie.sh                               # menagerie CI replica on a temporary copy (clone untouched)
python tools/tiago_pro/find_park_pose.py                         # search for overlay.LEFT_PARK
```

## Files

- `convert_tiago_pro.py` drives stage 1 (xacro -> URDF -> MJCF), stage 2 (overlay -> package) and stage 3 (package -> robosuite).
- `overlay.py` every decision as data (classes, collision policy, gains, excludes, poses, cameras, gripper recipe).
- `shim/ament_index_python` resolves `$(find pkg)` offline (`TIAGO_PKG_PATHS`), `stubs/` the stand-in for realsense2_description.
- `sources.lock` upstream commits and tool versions.

## Known deviations from the real robot

- Real PAL Pro gripper (no surrogate); PAL's closed-loop `connect` constraints are not emitted (mimic equalities only).
- `realsense2_description` is stubbed: head camera frames only (about 2 cm). Wrist camera (`eye_in_hand`) is a CAMI-compat
  parameter (fovy 75 like Panda, beside the gripper housing), not a hardware replica.
- Omni wheels are isotropic cylinders (menagerie package only: the base cannot strafe). In robosuite the base is frozen.
- Laser meshes (`sick_tim551`) are not redistributed (license); laser bodies remain as massive frames.
- Torso, head and the left arm (parked, gripper closed) are frozen in `robot_right.xml` (`overlay.PARK_*`, `LEFT_PARK`).

## robosuite integration (stage B): `TiagoProRight`, `PalProGripper`

Hand-written python (constants cross-checked against the generated XML by `tests/test_robots/test_tiago_pro_consistency.py`):
`robosuite/models/robots/manipulators/tiago_pro_right_robot.py`, `robosuite/models/grippers/pal_pro_gripper.py`,
`robosuite/controllers/config/robots/default_tiagoproright.json`. Registered as `TiagoProRight` (`WheeledRobot` +
`NoActuationBase`) and `PalProGripper`. Action space: 7 (OSC_POSE delta 6 + gripper 1) on NutAssemblySquare / ToolHang;
**6 on Wipe**, which forces `WipingGripper` (dof 0) and never instantiates the PAL gripper.

- Gripper: PAL `finger_joint` 0 = CLOSED, 0.07 = OPEN, robosuite action +1 = close. The GRIP controller maps `a` linearly
  onto ctrlrange [0, 0.07] (-1 -> 0), so `format_action` returns `1 - 2*closure`; `current_action` is the closure fraction in
  [0, 1] (robosuite resets it to zeros = open, consistent with `init_qpos`). 8 joints (`init_qpos` has 8 values), `dof` 1.
- Controller JSON mirrors Panda's OSC_POSE block (kp 150, damping 1, output_max [.05 .05 .05 .5 .5 .5], base frame) except
  `uncouple_pos_ori: false`. With Panda's `true` the coupled inertia of this arm + 0.81 kg gripper makes a +x command drift
  4.5 cm in z and tilt the wrist by 12 deg in 20 steps (Panda: 1.5 cm / 3.7 deg); with `false` 1.2 cm / 2.9 deg.
- `right_center` (the OSC "base" frame site) is generated with the robot-base orientation (x forward, y left, z up): the PAL
  arm mount frame is rotated, and an unaligned site made `+x` move the eef in a wrong direction.

### Reach study (`find_init_qpos.py`) and the baked constants

Kinematic IK scans (`find_init_qpos.py scan [--yaw]`, `init`, `park`), targets from the CAMI tasks (NutAssemblySquare
nut handle + peg 1, ToolHang frame/tool grasp + stand), approach axis within 10 deg of straight down, joint margin >= 0.03 rad
(0.1 rad with `--yaw`: 4 gripper yaws), IK to 1 cm:

| torso H | nut grasp | ToolHang grasp | ToolHang stand | peg 1 (carry height) |
|---|---|---|---|---|
| 0.25 | 100% | 100% | 100% | 0% |
| 0.30 | 100% | 100% | 99% | 0% |
| **0.35** | **100%** | **100%** | **100%** | **6% (12% at base y = 0.25)** |

(base y = 0.159, gap 0.02 m; without the yaw requirement the peg region is 28% (tilt <= 10 deg) to 38% (tilt <= 45 deg) at H = 0.35.)
**The peg-1 region of NutAssemblySquare (x = 0.23 m) is NOT reachable with a vertical approach from a base that has to stay
outside the table box**; nut grasp, lift, ToolHang grasp and the stand region are. Baked: `PARK_TORSO = 0.35` (joint limit),
`INIT_QPOS = [-2.5965, -1.3022, 1.0465, -1.8428, 2.2129, 1.4721, 0.8143]` (grip site 0.15-0.17 m above the table, approach axis
0.7-2.4 deg from vertical (<= 5 deg with the default init noise), fingers open along world y like Panda), base offsets:
table `(-(L/2 + 0.358 + 0.02), 0.159, 0)` (base collision extent x +-0.358, y +-0.248; the arm mount sits at y = -0.159 so the
shoulder is on the table centre line), bins `(-0.58, 0.059, 0)`, empty `(-0.665, 0.159, 0)`; `top_offset (0, 0, 1)`,
`_horizontal_radius 0.5`. The left arm (park pose `LEFT_PARK`) stays >= 0.20 m from table / floor / objects over 30 resets of
NutAssemblySquare and ToolHang (`find_init_qpos.py park`), and 0 of 42 right-arm IK solutions over the task grids touch the parked
left arm (closest 0.24 m). `EYE_IN_HAND_*` retuned: pos (0.05, 0, 0.105), tilt -12 deg (a pose at z = 0.07 / -27 deg looks into the
housing: 41% robot pixels; now 7-9% at 84x84, nut and both finger tips in view; `tests/test_robots/test_tiago_pro_camera.py`).

### Known limitations (measured, see the tests)

- F/T: gripper + fingers weigh 0.81 kg (7.97 N at the wrist sensor). Bias-corrected, a wrist tilt of 20 deg gives 2.7 N (45 deg: 6.1 N)
  without contact; free-space oscillation at full action amplitude reaches 11.6 N (Panda 4.6 N), half amplitude 4.1 N. With the reset-time
  bias of `NutAssembly._reset_internal` a motionless robot reads 2.6 N (Panda 1.9 N). The 10 N threshold needs re-tuning for the real data.
- `robot0_gripper_qpos` is 8-D (Panda: 2-D); robomimic infers the shape from the dataset.
- State replay in `augment_dataset_with_force.py` is not faithful (stale `data.ctrl`); re-stepping actions is (see `scripted_collect_smoke.py`).
- Throughput: 54 env.step/s vs 60 for Panda (control_freq 20, same machine).
- **MuJoCo 3.5.0 (robomimic's pin): force/torque sensors were frozen at the reset-time reading** under robosuite's default
  `lite_physics` stepping (`mj_step1` + `mj_step2` leave the lazy `flg_rnepost` flag set, so acceleration-stage sensors are not
  re-evaluated; `mj_step` and 3.3.0/3.14 are fine). This affects Panda too (pre-existing). `MjSim.step2` now clears the flag
  (no-op on versions without it); `test_ft_sensor_is_refreshed_by_env_step` covers it.
