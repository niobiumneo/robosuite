#!/usr/bin/env python
"""Acceptance checks for the committed TIAGo Pro assets (no xacro / PAL clones needed, any MuJoCo >= 3.3).

  python tools/tiago_pro/verify_tiago_pro.py [--assets robosuite/models/assets]

Checks the menagerie package (scenes load, sizes, sensors, statics, '../'), robot_right.xml (joints, actuators, frozen
poses == package forward kinematics, mass), pal_pro_gripper.xml (elements, sensors last, mimic equalities) and that the
robosuite loader (MujocoXML, flat defaults) builds the merged robot + gripper. Exit code 1 on any FAIL.
"""
import argparse
import os
import re
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
fails = []


def check(cond, msg):
    print(("  PASS: " if cond else "  FAIL: ") + msg)
    if not cond:
        fails.append(msg)


def name(m, t, i):
    return mujoco.mj_id2name(m, t, i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.path.join(REPO, "robosuite", "models", "assets"))
    a = ap.parse_args()
    tiago = os.path.join(a.assets, "robots", "tiago_pro")
    pkg = os.path.join(tiago, "pal_tiago_pro")
    print("mujoco", mujoco.__version__)

    # ------------------------------------------------------------------ package
    print("== package")
    neqs, models = set(), {}
    for v in ("position", "velocity", "motor"):
        m = mujoco.MjModel.from_xml_path(os.path.join(pkg, "scene_%s.xml" % v))
        models[v] = m
        neqs.add(m.neq)
        check(m.nq == 44 and m.nv == 43 and m.nu == 23 and m.nsensor == 2 and m.nkey == 1,
              "scene_%s: nq=%d nv=%d nu=%d nsensor=%d nkey=%d" % (v, m.nq, m.nv, m.nu, m.nsensor, m.nkey))
    check(neqs == {14}, "neq == 14 (7 mimic equalities per gripper) in every scene (got %s)" % neqs)
    n_dots = sum(open(os.path.join(pkg, f)).read().count("../") for f in os.listdir(pkg) if f.endswith(".xml"))
    check(n_dots == 0, "no '../' in package XML")
    sizes = [os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(pkg) for f in fs]
    check(sum(sizes) <= 8_000_000 and max(sizes) <= 1_000_000,
          "package size %.2f MB, largest file %d kB" % (sum(sizes) / 1e6, max(sizes) / 1e3))
    m = models["motor"]
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qvel[:] = 0
    d.qacc[:] = 0
    mujoco.mj_inverse(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_right_base_link")
    mg = m.body_subtreemass[b] * 9.81
    f = np.linalg.norm(d.sensordata[:3])
    check(abs(f - mg) / mg < 0.02, "statics (mj_inverse, qacc=0): |F|=%.4f N vs m*g=%.4f N (distal mass %.4f kg)"
          % (f, mg, m.body_subtreemass[b]))
    mujoco.mj_forward(m, d)
    check(d.ncon == 0, "no contacts at the home keyframe (ncon=%d)" % d.ncon)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    check(abs(m.geom_pos[floor][2]) < 1e-12, "floor at z=0")

    # ------------------------------------------------------------------ robot_right
    print("== robot_right.xml")
    txt = open(os.path.join(tiago, "robot_right.xml")).read()
    check(re.search(r"class=|childclass|<default", txt) is None, "flattened: no <default>, class= or childclass")
    r = mujoco.MjModel.from_xml_path(os.path.join(tiago, "robot_right.xml"))
    joints = [name(r, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(r.njnt)]
    check(joints == ["arm_right_%d_joint" % i for i in range(1, 8)], "joint list == arm_right_1..7_joint (in order)")
    check(r.nu == 7 and r.na == 0 and np.allclose(r.actuator_ctrlrange[:, 1], [43, 43, 26, 26, 26, 26, 26]),
          "nu=7, na=0, ctrlrange == URDF efforts %s" % r.actuator_ctrlrange[:, 1].tolist())
    check(all(r.actuator_trntype == 0) and all(r.actuator_gaintype == 0) and all(r.actuator_biastype == 0),
          "all actuators are plain <motor>")
    bad = [s for s in joints + [name(r, mujoco.mjtObj.mjOBJ_ACTUATOR, u) for u in range(r.nu)]
           if any(k in s for k in ("torso", "mobile", "head", "leg"))]
    check(not bad, "no joint/actuator name contains torso/mobile/head/leg")
    hand = mujoco.mj_name2id(r, mujoco.mjtObj.mjOBJ_BODY, "right_hand")
    check(hand >= 0 and not any(r.geom_bodyid == hand) and r.body_mass[hand] == 0
          and not any(r.body_parentid == hand), "body 'right_hand' exists and is empty")
    cams = [name(r, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in range(r.ncam)]
    check(set(cams) == {"eye_in_hand", "robotview"}, "cameras %s" % cams)
    # frozen FK == package FK
    rng = np.random.default_rng(0)
    pm = mujoco.MjModel.from_xml_path(os.path.join(pkg, "tiago_pro.xml"))
    pd, rd = mujoco.MjData(pm), mujoco.MjData(r)
    sys.path.insert(0, HERE)
    import overlay as ov  # noqa: E402

    def pj(n):
        return pm.jnt_qposadr[mujoco.mj_name2id(pm, mujoco.mjtObj.mjOBJ_JOINT, n)]

    worst = 0.0
    nchecked = 0
    for k in range(20):
        q = np.array([rng.uniform(r.jnt_range[j, 0], r.jnt_range[j, 1]) for j in range(7)])
        pd.qpos[:] = pm.qpos0
        pd.qpos[2] = 0.0  # robot_right sits on the floor
        pd.qpos[pj("torso_lift_joint")] = ov.PARK_TORSO
        pd.qpos[pj("head_1_joint")], pd.qpos[pj("head_2_joint")] = ov.PARK_HEAD
        for i, v in enumerate(ov.LEFT_PARK):
            pd.qpos[pj("arm_left_%d_joint" % (i + 1))] = v
        pd.qpos[pj("gripper_left_finger_joint")] = ov.PARK_LEFT_GRIPPER  # closed: every mimic joint is 0
        for i in range(7):
            pd.qpos[pj("arm_right_%d_joint" % (i + 1))] = q[i]
        mujoco.mj_forward(pm, pd)
        rd.qpos[:] = q
        mujoco.mj_forward(r, rd)
        for bi in range(1, r.nbody):
            n = name(r, mujoco.mjtObj.mjOBJ_BODY, bi)
            pn = "arm_right_tool_link" if n == "right_hand" else n
            pb = mujoco.mj_name2id(pm, mujoco.mjtObj.mjOBJ_BODY, pn)
            if pb < 0:
                continue
            worst = max(worst, np.abs(pd.xpos[pb] - rd.xpos[bi]).max())
            nchecked += 1
    check(worst < 1e-6, "FK of %d body poses (20 random right-arm qpos) equals package FK, max error %.2e m"
          % (nchecked, worst))
    gm = float(pm.body_subtreemass[mujoco.mj_name2id(pm, mujoco.mjtObj.mjOBJ_BODY, "gripper_right_base_link")])
    check(abs(r.body_mass.sum() + gm - pm.body_mass.sum()) < 1e-9,
          "mass: robot_right %.6f + right gripper %.6f == package %.6f" % (r.body_mass.sum(), gm, pm.body_mass.sum()))

    # ------------------------------------------------------------------ gripper
    print("== pal_pro_gripper.xml")
    from lxml import etree

    gx = etree.parse(os.path.join(a.assets, "grippers", "pal_pro_gripper.xml")).getroot()
    check(re.search(r"class=|childclass|<default", open(os.path.join(a.assets, "grippers", "pal_pro_gripper.xml")).read()) is None,
          "flattened: no <default>, class= or childclass")
    bodies = {b.get("name") for b in gx.iter("body")}
    sites = {s.get("name") for s in gx.iter("site")}
    check("eef" in bodies and "right_gripper" in bodies, "bodies right_gripper and eef exist")
    check({"grip_site", "grip_site_cylinder", "ee_x", "ee_y", "ee_z", "ft_frame"} <= sites, "sites %s" % sorted(sites))
    sens = [s.get("name") for s in gx.find("sensor")]
    check(sens == ["force_ee", "torque_ee"] and list(gx)[-1].tag == "sensor", "sensors %s are the last element" % sens)
    check(len(gx.find("actuator")) == 1, "nu == 1")
    try:
        sys.path.insert(0, REPO)
        from robosuite.models.base import MujocoXML
        from robosuite.models.grippers.gripper_model import GripperModel

        class _G(GripperModel):
            def __init__(self, f, idn="0_right"):
                super().__init__(f, idn=idn)

            def format_action(self, action):
                return action

            @property
            def init_qpos(self):
                return np.zeros(8)

        g = _G(os.path.join(a.assets, "grippers", "pal_pro_gripper.xml"))
        rob = MujocoXML(os.path.join(tiago, "robot_right.xml"))
        rob.get_model()
        check(True, "robosuite MujocoXML loads robot_right.xml (flat defaults)")
        rob.merge(g, merge_body="right_hand")
        mm = rob.get_model()
        sn = [name(mm, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(mm.nsensor)]
        check(sn == ["gripper0_right_force_ee", "gripper0_right_torque_ee"], "merged sensor names %s" % sn)
        check(mm.nu == 8 and mm.nq == 15 and mm.neq == 7, "merged nq=%d nu=%d neq=%d" % (mm.nq, mm.nu, mm.neq))
        # grasp site FK: merged model vs package, random right arm + finger opening
        md = mujoco.MjData(mm)
        worst = 0.0
        gj = lambda n: mm.jnt_qposadr[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, "gripper0_right_" + n)]
        for k in range(20):
            q = np.array([rng.uniform(r.jnt_range[j, 0], r.jnt_range[j, 1]) for j in range(7)])
            fo = rng.uniform(0, 0.07)
            md.qpos[:7] = q
            pd.qpos[:] = pm.qpos0
            pd.qpos[2] = 0.0
            for i in range(7):
                pd.qpos[pj("arm_right_%d_joint" % (i + 1))] = q[i]
            pd.qpos[pj("torso_lift_joint")] = ov.PARK_TORSO
            pd.qpos[pj("gripper_right_finger_joint")] = fo
            md.qpos[gj("finger_joint")] = fo
            for e in range(pm.neq):
                if pm.eq_type[e] == 2 and name(pm, mujoco.mjtObj.mjOBJ_JOINT, pm.eq_obj2id[e]).startswith("gripper_right"):
                    jn = name(pm, mujoco.mjtObj.mjOBJ_JOINT, pm.eq_obj2id[e])
                    pd.qpos[pj(jn)] = fo / pm.eq_data[e, 1]
                    md.qpos[gj(jn[len("gripper_right_"):])] = fo / pm.eq_data[e, 1]
            # package torso baked in robot_right at PARK_TORSO; left side irrelevant for the right chain
            mujoco.mj_forward(pm, pd)
            mujoco.mj_forward(mm, md)
            ps = pd.site_xpos[mujoco.mj_name2id(pm, mujoco.mjtObj.mjOBJ_SITE, "gripper_right_grasping_frame_Z")]
            ms = md.site_xpos[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")]
            worst = max(worst, np.abs(ps - ms).max())
            if k == 0:
                print("   sample grasp site", ps.round(5), ms.round(5), "(dx from default %.3f)" % np.linalg.norm(ps - pm.site_pos[0]))
        check(worst < 1e-6, "grasp site FK (merged robosuite model vs package, random arm + finger), max error %.2e" % worst)
    except ImportError as e:  # robosuite not importable: skip
        print("  SKIP robosuite loader checks:", e)
    print("\nRESULT:", "PASS" if not fails else "%d FAIL(s)" % len(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
