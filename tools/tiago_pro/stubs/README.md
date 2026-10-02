# STUB: realsense2_description

`realsense2_description` (Intel realsense-ros) is referenced by the PAL head xacro but is not part of the PAL repositories,
so a hand-written stand-in with the usual realsense frame names (`*_link`, `*_depth_frame`, `*_color_optical_frame`, ...)
is used. Frame structure follows realsense-ros; **numeric offsets are not verified** (about 2 cm). It only affects the head
camera frames (`head_front_camera_*`) and the one box that stands in for the camera body. It is not used by the CAMI
wrist camera. Replace it with the real package (add it to `PAL_PACKAGES` in `overlay.py`) to remove the uncertainty.

The real `pal_pro_gripper_description` (github.com/pal-robotics/pal_pro_gripper) is used, not stubbed.
