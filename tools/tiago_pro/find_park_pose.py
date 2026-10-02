#!/usr/bin/env python
"""Search a dedicated left-arm PARK pose for the frozen left arm of robot_right.xml (result -> overlay.LEFT_PARK).

Criteria (evaluated at torso 0.0 and 0.35 m with the left gripper closed): no contact of the left arm/gripper with anything,
clearance >= 4 cm to every non-adjacent geom, every collision vertex at x <= 0.25 m (inside the base footprint, which
reaches x=0.36), y >= 0.12 (left of the body, away from the right arm's work volume), joints >= 0.15 rad from their limits,
smallest bounding-box volume. The best candidates are additionally swept against 300 random right-arm configurations
around the PAL home pose (fraction of configurations that touch the parked arm is printed).

  python tools/tiago_pro/find_park_pose.py [--samples 20000] [--seed 1]
"""
import argparse
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "..", "robosuite", "models", "assets", "robots", "tiago_pro", "pal_tiago_pro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(os.path.join(PKG, "scene_motor.xml"))
    d = mujoco.MjData(m)
    J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    B = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
    lg = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_type[g] == 7 and ("arm_left" in B(g) or "gripper_left" in B(g))]
    lgs = set(lg)
    verts = {g: m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:][:m.mesh_vertnum[m.geom_dataid[g]]][::4] for g in lg}
    lj = [J("arm_left_%d_joint" % i) for i in range(1, 8)]
    rj = [J("arm_right_%d_joint" % i) for i in range(1, 8)]
    lo, hi = (np.array([m.jnt_range[j, k] for j in lj]) for k in (0, 1))
    rlo, rhi = (np.array([m.jnt_range[j, k] for j in rj]) for k in (0, 1))
    excl = set(m.exclude_signature.tolist())
    rng = np.random.default_rng(a.seed)

    def pose(q, torso):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[m.jnt_qposadr[J("torso_lift_joint")]] = torso
        for j, v in zip(lj, q):
            d.qpos[m.jnt_qposadr[j]] = v
        for j in range(m.njnt):  # closed left gripper: every left gripper joint is 0
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("gripper_left"):
                d.qpos[m.jnt_qposadr[j]] = 0.0
        mujoco.mj_forward(m, d)

    def extent():
        P = np.vstack([(d.geom_xmat[g].reshape(3, 3) @ verts[g].T).T + d.geom_xpos[g] for g in lg])
        return P.min(0), P.max(0)

    def clearance():
        others = [g for g in range(m.ngeom) if m.geom_contype[g] and g not in lgs and m.geom_bodyid[g] != 0]
        worst, ft = 1.0, np.zeros(6)
        for g in lg:
            for o in others:
                b1, b2 = m.geom_bodyid[g], m.geom_bodyid[o]
                if b1 == b2 or m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1:
                    continue
                if (min(b1, b2) << 16) + max(b1, b2) in excl:
                    continue
                worst = min(worst, mujoco.mj_geomDistance(m, d, g, o, 0.3, ft))
        return worst

    cands = []
    for _ in range(a.samples):
        q = rng.uniform(lo, hi)
        if np.min(np.minimum(q - lo, hi - q)) < 0.15:
            continue
        ok, mn, mx = True, np.full(3, 9.0), np.full(3, -9.0)
        for t in (0.0, 0.35):
            pose(q, t)
            if any((c.geom1 in lgs) or (c.geom2 in lgs) for c in d.contact[: d.ncon]):
                ok = False
                break
            lo_, hi_ = extent()
            mn, mx = np.minimum(mn, lo_), np.maximum(mx, hi_)
        if ok and mx[0] <= 0.25 and mn[1] >= 0.12:
            cands.append((float(np.prod(mx - mn)), q, mn, mx))
    cands.sort(key=lambda c: c[0])
    print("%d candidates; best by bounding-box volume (clearance >= 4 cm, right-arm sweep):" % len(cands))
    shown = 0
    for vol, q, mn, mx in cands[:60]:
        cl = min((pose(q, t), clearance())[1] for t in (0.0, 0.35))
        if cl < 0.04:
            continue
        hit = 0
        for _ in range(300):
            rq = np.clip(np.array([-0.36, -1.83, -0.47, -2.35, 0.0, -1.2, 0.0]) + rng.uniform(-0.9, 0.9, 7), rlo, rhi)
            pose(q, 0.2)
            for j, v in zip(rj, rq):
                d.qpos[m.jnt_qposadr[j]] = v
            mujoco.mj_forward(m, d)
            hit += any(((c.geom1 in lgs) != (c.geom2 in lgs)) and any(k in B(c.geom2 if c.geom1 in lgs else c.geom1)
                                                                       for k in ("arm_right", "gripper_right")) for c in d.contact[: d.ncon])
        print("vol %.3f clear %.3f right-arm hit %.3f  q=%s  bbox %s..%s" % (vol, cl, hit / 300, np.round(q, 3).tolist(), mn.round(2), mx.round(2)))
        shown += 1
        if shown >= 8:
            break


if __name__ == "__main__":
    main()
