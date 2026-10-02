"""
Best-effort render check of the TiagoProRight wrist camera (``robot0_eye_in_hand_image``, the CAMI image key).

Needs an OpenGL context: runs in a subprocess under ``xvfb-run`` with ``MUJOCO_GL=glx`` (software Mesa is enough) and is
skipped when that is not available. Checks at 84x84 (the CAMI resolution): the image is not constant, the robot's own
geoms occupy < 20% of the pixels at init and while the gripper surrounds the nut handle, and the nut is in view at the
grasp pose. Frames can be saved for a visual diff against Panda with ``TIAGO_CAMERA_FRAMES=<dir>``.

$ pytest -s tests/test_robots/test_tiago_pro_camera.py
"""
import os
import shutil
import subprocess
import sys

import pytest

import robosuite

REPO = os.path.abspath(os.path.join(os.path.dirname(robosuite.__file__), ".."))
TOOLS = os.path.join(REPO, "tools", "tiago_pro")

SCRIPT = r"""
import logging, os, sys
import numpy as np, mujoco
import robosuite as suite
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)
sys.path.insert(0, %(tools)r)
import scripted_collect_smoke as S_

env = suite.make("NutAssemblySquare", robots=["TiagoProRight"], has_renderer=False, has_offscreen_renderer=True,
                 use_camera_obs=True, camera_names=["robot0_eye_in_hand"], camera_heights=84, camera_widths=84,
                 camera_segmentations="element", control_freq=20, initialization_noise=None, ignore_done=True)
m = env.sim.model._model

def fractions(obs):
    seg = obs["robot0_eye_in_hand_segmentation_element"][::-1, :, 0]
    rob = nut = 0
    for g in np.unique(seg):
        if 0 <= g < m.ngeom:
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            rob += (seg == g).sum() if b.startswith(("robot0", "gripper0")) else 0
            nut += (seg == g).sum() if "Nut" in b else 0
    return rob / seg.size, nut / seg.size

obs = env.reset()
img0 = obs["robot0_eye_in_hand_image"]
assert img0.shape == (84, 84, 3) and img0.std() > 1.0, "image constant"
r0, n0 = fractions(obs)
pol = S_.NutGraspPolicy(env, press=False)
for _ in range(300):
    obs, *_ = env.step(pol(obs))
    if pol.phase == "close" and pol.count == 1:
        break
img1 = obs["robot0_eye_in_hand_image"]
r1, n1 = fractions(obs)
print("RESULT init robot_frac=%%.3f grasp robot_frac=%%.3f nut_frac=%%.3f" %% (r0, r1, n1))
assert pol.phase == "close", pol.phase
assert r0 < 0.2 and r1 < 0.2, (r0, r1)
assert n1 > 0.02, n1
assert np.abs(img1.astype(float) - img0.astype(float)).mean() > 1.0
d = os.environ.get("TIAGO_CAMERA_FRAMES")
if d:
    import imageio
    os.makedirs(d, exist_ok=True)
    imageio.imwrite(os.path.join(d, "eye_in_hand_init.png"), img0[::-1])
    imageio.imwrite(os.path.join(d, "eye_in_hand_grasp.png"), img1[::-1])
env.close()
"""


def test_wrist_camera_renders_under_xvfb():
    if shutil.which("xvfb-run") is None:
        pytest.skip("xvfb-run not available: wrist camera not rendered (no GL context)")
    if not os.path.isdir(TOOLS):
        pytest.skip("tools/tiago_pro is not part of the installed package")
    env = dict(os.environ, MUJOCO_GL="glx", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join([REPO] + [p for p in sys.path if p])
    proc = subprocess.run(
        ["xvfb-run", "-a", sys.executable, "-c", SCRIPT % {"tools": TOOLS}],
        env=env, capture_output=True, text=True, timeout=300,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0 and ("GLX" in out or "OpenGL" in out or "display" in out.lower()) and "AssertionError" not in out:
        pytest.skip("no usable OpenGL context under xvfb: " + out[-300:])
    assert proc.returncode == 0, out[-2000:]
    print("\n" + [l for l in out.splitlines() if l.startswith("RESULT")][0])
