#!/usr/bin/env python
"""Reach study for TiagoProRight (stage B): torso height, base placement, init_qpos.

The torso is frozen at generation time (overlay.PARK_TORSO), so one value serves every task. The study is kinematic:
IK (damped least squares, joint limits, free yaw about the approach axis) of the gripper `grip_site` onto target grids
derived from the CAMI tasks, from a TiagoProRight env that is built once. A different torso height H only translates
the arm mount vertically (the torso joint is a vertical slide), so it is emulated by shifting target heights by
(PARK_TORSO - H); a different base placement (x gap to the table edge, y offset) shifts the targets in the base frame.

Target sets (world frame; z relative to the table top):
  nut_grasp   NutAssemblySquare square-nut handle region (x -0.17..-0.065, y 0.06..0.225)  z +0.025
  peg         carry the nut over peg1 (x 0.16..0.30, y 0.03..0.17)                         z +0.15, +0.20
  th_grasp    ToolHang frame / tool grasp region (x -0.1..0.07, y -0.28..-0.16)            z +0.03
  th_stand    ToolHang assembly region above the stand (x -0.15..0, y -0.08..0.08)         z +0.15, +0.25
(Wipe is not scanned: it forces WipingGripper (dof 0, action_dim 6) with another tool length, see the README.)

Reachable = IK to within 1 cm with joint limits (>= 0.03 rad margin) and the approach axis within TILT degrees of
straight down; reported for TILT in (10, 45).

  python tools/tiago_pro/find_init_qpos.py scan   [--torso 0 .1 .2 .3 .35] [--y ..] [--yaw]   # torso / base placement table
  python tools/tiago_pro/find_init_qpos.py init   [--torso H --gap G --y Y]                # choose init_qpos
  python tools/tiago_pro/find_init_qpos.py park   [--resets 30]                             # left-arm park clearance to table / objects
"""
import argparse
import itertools
import logging
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SITE = "gripper0_right_grip_site"
ARM_JOINTS = ["robot0_arm_right_%d_joint" % i for i in range(1, 8)]
DOWN = np.array([0.0, 0.0, -1.0])
TABLE_L = 0.8  # NutAssembly / ToolHang table length (x)
TARGETS = {  # name -> (task, x range, y range, z above table top list)
    "nut_grasp": ("NutAssemblySquare", (-0.17, -0.065), (0.06, 0.225), (0.025,)),
    "peg": ("NutAssemblySquare", (0.16, 0.30), (0.03, 0.17), (0.15, 0.20)),
    "th_grasp": ("ToolHang", (-0.10, 0.07), (-0.28, -0.16), (0.03,)),
    "th_stand": ("ToolHang", (-0.15, 0.0), (-0.08, 0.08), (0.15, 0.25)),
}
TABLE_TOP = {"NutAssemblySquare": 0.82, "ToolHang": 0.80}


def grid(name, n=4):
    task, xr, yr, zs = TARGETS[name]
    pts = [(x, y, TABLE_TOP[task] + z) for x in np.linspace(*xr, n) for y in np.linspace(*yr, n) for z in zs]
    return np.array(pts)


class Kin:
    """Kinematics of the right arm inside a built TiagoProRight env (no dynamics)."""

    def __init__(self):
        import mujoco

        import robosuite as suite
        from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

        ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)
        self.mujoco = mujoco
        self.env = suite.make("NutAssemblySquare", robots="TiagoProRight", has_renderer=False,
                              has_offscreen_renderer=False, use_camera_obs=False, control_freq=20)
        self.env.reset()
        self.m = self.env.sim.model._model
        self.d = mujoco.MjData(self.m)
        self.base0 = np.array(self.env.robots[0].base_pos)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, SITE)
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in jid])
        self.lo = self.m.jnt_range[jid, 0].copy()
        self.hi = self.m.jnt_range[jid, 1].copy()
        self.torso0 = None

    def fk(self, q):
        self.d.qpos[:] = self.env.sim.data._data.qpos
        self.d.qpos[self.qadr] = q
        self.mujoco.mj_kinematics(self.m, self.d)
        self.mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def jac(self):
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        self.mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
        return jp[:, self.dadr], jr[:, self.dadr]

    def ik(self, target, q0, w_ori, yaw=None, iters=120, margin=0.03):
        """DLS IK. Position + approach axis (weight w_ori); with `yaw` (rad) the full orientation is constrained
        (approach down, x axis of the site rotated by yaw about world z)."""
        q = np.clip(q0, self.lo + margin, self.hi - margin)
        lam = 1e-2
        Rd = None
        if yaw is not None:
            c, s = np.cos(yaw), np.sin(yaw)
            Rd = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]])  # z down, x at `yaw`
        for _ in range(iters):
            p, R = self.fk(q)
            ep = target - p
            Jp, Jr = self.jac()
            if Rd is None:
                ea = np.cross(R[:, 2], DOWN)
                J = np.vstack([Jp, w_ori * Jr])
                e = np.concatenate([ep, w_ori * ea])
            else:
                eR = 0.5 * (np.cross(R[:, 0], Rd[:, 0]) + np.cross(R[:, 1], Rd[:, 1]) + np.cross(R[:, 2], Rd[:, 2]))
                J = np.vstack([Jp, w_ori * Jr])
                e = np.concatenate([ep, w_ori * eR])
            if np.linalg.norm(ep) < 2e-3 and np.linalg.norm(e[3:]) / max(w_ori, 1e-9) < 0.02:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(len(e)), e)
            n = np.abs(dq).max()
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo + margin, self.hi - margin)
        p, R = self.fk(q)
        tilt = np.degrees(np.arccos(np.clip(R[:, 2] @ DOWN, -1, 1)))
        if Rd is not None:  # report the full orientation error (deg) as `tilt`
            tilt = np.degrees(np.arccos(np.clip((np.trace(Rd.T @ R) - 1) / 2, -1, 1)))
        return q, np.linalg.norm(target - p), tilt

    def clearance(self, q, other=None):
        """Smallest signed distance between the right arm / gripper collision geoms and every other collision geom
        (torso, base, left arm, table, ...) at arm configuration `q`; pairs that are excluded or parent/child are skipped."""
        mj = self.mujoco
        m, d = self.m, self.d
        d.qpos[:] = self.env.sim.data._data.qpos
        d.qpos[self.qadr] = q
        mj.mj_forward(m, d)
        name = lambda g: mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        mine = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_group[g] != 1 and
                (name(g).startswith("robot0_arm_right") or name(g).startswith("gripper0_right"))]
        mine_set = set(mine)
        others = [g for g in range(m.ngeom) if m.geom_contype[g] and g not in mine_set and (other is None or other in name(g))]
        excl = set(m.exclude_signature.tolist())
        ft = np.zeros(6)
        worst = (1.0, "")
        for g in mine:
            for o in others:
                b1, b2 = m.geom_bodyid[g], m.geom_bodyid[o]
                if b1 == b2 or m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1:
                    continue
                if (min(b1, b2) << 16) + max(b1, b2) in excl:
                    continue
                if not (m.geom_contype[g] & m.geom_conaffinity[o] or m.geom_contype[o] & m.geom_conaffinity[g]):
                    continue
                dist = mj.mj_geomDistance(m, d, g, o, 0.5, ft)
                if dist < worst[0]:
                    worst = (dist, "%s <-> %s" % (name(g), name(o)))
        return worst

    def solve_point(self, target, seeds, w_list=(3.0, 1.0, 0.3, 0.1)):
        """Best (pos_err<1cm) solution with the smallest tilt over seeds and orientation weights."""
        best = (1e9, 1e9, None)
        for q0 in seeds:
            for w in w_list:
                q, pe, tilt = self.ik(target, q0, w)
                if pe < 0.01 and tilt < best[1]:
                    best = (pe, tilt, q)
                if pe < 0.01 and tilt < 3.0:
                    return best
        return best


def make_seeds(kin, rng, n=6):
    nom = np.array([0.0, 0.3, 0.0, 0.8, 0.0, 0.5, 0.0])
    home = np.array([-0.36, -1.83, -0.47, -2.35, 0.0, -1.2, 0.0])  # PAL `home` motion (right arm)
    seeds = [np.clip(nom, kin.lo, kin.hi), np.clip(home, kin.lo, kin.hi)]
    seeds += [np.clip(home + rng.uniform(-0.5, 0.5, 7), kin.lo, kin.hi) for _ in range(max(0, (n - 2) // 2))]
    return seeds + [rng.uniform(kin.lo + 0.2, kin.hi - 0.2) for _ in range(n - len(seeds))]


def scan_one(args):
    torso, gap, yoff, park_torso, yaw = args
    kin = _KIN
    rng = np.random.default_rng(0)
    seeds = make_seeds(kin, rng, 10 if yaw else 6)
    base_c = np.array([-(TABLE_L / 2 + 0.358 + gap), yoff, 0.0])
    res = {}
    for name in TARGETS:
        pts = grid(name)
        shifted = pts - base_c + kin.base0 + np.array([0, 0, park_torso - torso])
        tilts = []
        if yaw:  # full 6-DoF: approach straight down with 4 gripper yaws, joint margin 0.1 rad
            for t in shifted:
                for yw in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
                    ok = False
                    for q0 in seeds:
                        _, pe, err = kin.ik(t, q0, 3.0, yaw=yw, iters=250, margin=0.1)
                        if pe < 0.01 and err < 3.0:
                            ok = True
                            break
                    tilts.append(0.0 if ok else 999.0)
        else:
            for t in shifted:
                pe, tilt, _ = kin.solve_point(t, seeds)
                tilts.append(tilt if pe < 0.01 else 999.0)
        tilts = np.array(tilts)
        res[name] = ((tilts <= 10).mean(), (tilts <= 45).mean())
    return (torso, gap, yoff), res


_KIN = None


def _init_worker():
    global _KIN
    _KIN = Kin()


def cmd_scan(a):
    from overlay import PARK_TORSO

    cfgs = list(itertools.product(a.torso, a.gap, a.y, [PARK_TORSO], [a.yaw]))
    with mp.Pool(a.workers, initializer=_init_worker) as pool:
        out = pool.map(scan_one, cfgs, chunksize=1)
    base_front = 0.358
    print("base collision front extent x = %.3f m (y +-0.248); table half length %.2f" % (base_front, TABLE_L / 2))
    print("reachable fractions: tilt<=10deg / tilt<=45deg   (nut_grasp  peg  th_grasp  th_stand)")
    print("torso  gap    y  | " + " | ".join("%-11s" % n for n in TARGETS))
    for (torso, gap, yoff), res in out:
        row = " | ".join("%4.2f / %4.2f" % res[n] for n in TARGETS)
        print("%5.2f %5.2f %5.2f | %s" % (torso, gap, yoff, row))
    return out


def cmd_init(a):
    """Choose init_qpos: gripper straight down above the middle of the work area, 0 deg yaw (fingers open along the world
    y axis like Panda's), away from joint limits, and best *local* coverage: the fraction of the task target grids
    (nut_grasp, th_grasp, th_stand) that IK reaches when started from that pose (what OSC deltas have to do), with the
    approach within 5 deg of down."""
    from overlay import PARK_TORSO

    kin = Kin()
    base_c = np.array([-(TABLE_L / 2 + 0.358 + a.gap), a.y, 0.0])
    shift = lambda P: P - base_c + kin.base0 + np.array([0, 0, PARK_TORSO - a.torso])
    rng = np.random.default_rng(a.seed)
    target = shift(np.array([[a.x, a.cy, 0.82 + a.height]]))[0]
    test = np.vstack([shift(grid(n, 3)) for n in ("nut_grasp", "th_grasp", "th_stand")])
    cands = []
    for q0 in make_seeds(kin, rng, a.seeds):
        q, pe, err = kin.ik(target, q0, 3.0, yaw=0.0, iters=400, margin=0.15)
        if pe > 2e-3 or err > 3.0:
            continue
        if any(np.abs(q - c[2]).max() < 0.05 for c in cands):
            continue
        cands.append([0.0, 0.0, q])
    for c in cands:
        q = c[2]
        ok = 0
        for t in test:
            _, pe, tilt = kin.ik(t, q, 3.0, iters=250, margin=0.05)
            ok += (pe < 0.01) and (tilt < 5.0)
        kin.fk(q)
        Jp, _ = kin.jac()
        c[0] = ok / len(test)
        c[1] = np.sqrt(np.linalg.det(Jp @ Jp.T))
        c.append(kin.clearance(q))
    cands.sort(key=lambda c: -(c[0] + 2.0 * c[1] + 2.0 * min(c[3][0], 0.1)))
    print("%d distinct IK solutions for the init target %s (base frame shift applied)" % (len(cands), np.round(target, 3)))
    for cov, manip, q, (clr, pair) in cands[:10]:
        print("coverage %.2f manip %.3f margin %.2f clearance %.3f (%s) q=%s" % (
            cov, manip, np.min(np.minimum(q - kin.lo, kin.hi - q)), clr, pair, np.round(q, 4).tolist()))
    return cands


def left_arm_clearance(task, n_resets=30, seed=0):
    """Minimum distance (m) from the parked left arm + left gripper to every non-robot geom (table, nuts, pegs, tools, ...)
    over `n_resets` resets of `task`; and the smallest left-arm to right-arm distance at the init pose."""
    import mujoco

    import robosuite as suite
    from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

    ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)
    env = suite.make(task, robots="TiagoProRight", has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, control_freq=20, seed=seed)
    m = env.sim.model._model
    d = env.sim.data._data
    body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
    left = [g for g in range(m.ngeom) if m.geom_contype[g] and ("arm_left" in body(g) or "gripper_left" in body(g))]
    world = [g for g in range(m.ngeom) if m.geom_contype[g] and not body(g).startswith(("robot0", "gripper0"))
             and m.geom_bodyid[g] != 0 or (m.geom_bodyid[g] == 0 and m.geom_contype[g])]
    ft = np.zeros(6)
    worst = 1.0
    for _ in range(n_resets):
        env.reset()
        for g in left:
            for o in world:
                worst = min(worst, mujoco.mj_geomDistance(m, d, g, o, 1.0, ft))
    env.close()
    return worst


def cmd_park(a):
    """Left-arm park pose checks: clearance to table/objects over the reset distributions (> 5 cm required)."""
    ok = True
    for task in ("NutAssemblySquare", "ToolHang"):
        dist = left_arm_clearance(task, a.resets)
        print("%-18s min distance left arm/gripper -> table / floor / objects over %d resets: %.3f m" % (task, a.resets, dist))
        ok &= dist > 0.05
    print("PASS" if ok else "FAIL (< 5 cm)")
    # right arm over its working set vs the left arm / torso / base (IK solutions ignore collisions: report only)
    kin = Kin()
    rng = np.random.default_rng(0)
    seeds = make_seeds(kin, rng, 6)
    base_c = np.array([-(TABLE_L / 2 + 0.358 + 0.02), 0.159, 0.0])
    n = bad = left_bad = 0
    worst = left_worst = (1.0, "")
    for name in TARGETS:
        for t in grid(name, 3):
            tt = t - base_c + kin.base0
            pe, tilt, q = kin.solve_point(tt, seeds)
            if q is None:
                continue
            n += 1
            c = kin.clearance(q)
            bad += c[0] < 0.0
            worst = min(worst, c)
            cl = kin.clearance(q, other="_left")
            left_bad += cl[0] < 0.0
            left_worst = min(left_worst, cl)
    print("right-arm IK solutions over the reachable task grids: %d; %d touch another link / the table (IK ignores collisions: "
          "arbitrary elbow), smallest signed distance %.3f m (%s)" % (n, bad, worst[0], worst[1]))
    print("  of these, collisions with the parked LEFT arm / gripper: %d (smallest distance %.3f m: %s)" % (left_bad, left_worst[0], left_worst[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--torso", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.35])
    s.add_argument("--gap", type=float, nargs="+", default=[0.02])
    s.add_argument("--y", type=float, nargs="+", default=[0.0, 0.159])
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--yaw", action="store_true", help="require full orientation (approach down, 4 gripper yaws)")
    s = sub.add_parser("park")
    s.add_argument("--resets", type=int, default=30)
    s = sub.add_parser("init")
    s.add_argument("--torso", type=float, default=0.35)
    s.add_argument("--gap", type=float, default=0.02)
    s.add_argument("--y", type=float, default=0.159)
    s.add_argument("--height", type=float, default=0.15, help="eef height above the table top")
    s.add_argument("--x", type=float, default=-0.10)
    s.add_argument("--cy", type=float, default=0.0, help="world y of the init target")
    s.add_argument("--seeds", type=int, default=60)
    s.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    {"scan": cmd_scan, "init": cmd_init, "park": cmd_park}[a.cmd](a)


if __name__ == "__main__":
    main()
