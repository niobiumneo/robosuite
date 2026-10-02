# TIAGo Pro assets

Everything in this directory is **generated** by `tools/tiago_pro/convert_tiago_pro.py` from the PAL Robotics TIAGo Pro
description (Apache-2.0), pinned in `tools/tiago_pro/sources.lock`. Do not edit by hand; edit
`tools/tiago_pro/overlay.py` and re-run the converter.

- `pal_tiago_pro/`: self-contained MuJoCo-menagerie-style package (full robot, position/velocity/motor variants,
  scenes, meshes). It can be copied verbatim into a menagerie checkout.
- `robot_right.xml`: robosuite robot model derived from the package: only the 7 right arm joints are free
  (torque motors); free joint, wheels, torso, head, left arm and left gripper are frozen by baking their pose into the
  body transforms (masses are kept). The right gripper is mounted by robosuite (`right_hand` is an empty body).
  Fully flattened (no `<default>`/`class`), mesh files are the package's (`pal_tiago_pro/assets/...`).
- `../../grippers/pal_pro_gripper.xml` (+ `../../grippers/meshes/pal_pro_gripper/`): the real PAL Pro gripper
  re-rooted as a robosuite gripper (sensors `force_ee` / `torque_ee` on `ft_frame`).

The real `pal_pro_gripper_description` is used (no surrogate). `realsense2_description` is stubbed (head camera frames only).
Licensing: PAL-derived assets are Apache-2.0 (see `pal_tiago_pro/LICENSE`); the rest of the repository is MIT.
