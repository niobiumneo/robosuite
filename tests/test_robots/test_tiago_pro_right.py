"""
Behavioural validation of the TiagoProRight robot (TIAGo Pro right arm + PAL Pro gripper, everything else frozen):

$ pytest -s tests/test_robots/test_tiago_pro_right.py

CAMI-relevant checks on NutAssemblySquare: action space, Panda-compatible observation keys, frozen links, OSC tracking,
gripper sign convention, F/T sensor sanity (bias taken AFTER the arm has settled, wrist tilt sweep, free-space motion),
init pose, self-contacts and step rate relative to Panda. Measured numbers are printed (use ``-s``).
"""
import collections
import logging
import time

import mujoco
import numpy as np
import pytest

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)

ROBOT = "TiagoProRight"
FT_THRESHOLD = 10.0  # N, the CAMI contact threshold (robomimic dataset.py / augment_dataset_with_force.py)
HOLD = np.array([0, 0, 0, 0, 0, 0, -1.0])  # zero arm motion, gripper open


def make_env(task="NutAssemblySquare", robot=ROBOT, **kwargs):
    config = dict(
        env_name=task,
        robots=robot,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        initialization_noise=None,
        hard_reset=False,
    )
    config.update(kwargs)
    return suite.make(**config)


@pytest.fixture(scope="module")
def env():
    e = make_env()
    e.reset()
    yield e
    e.close()


def name_of(m, kind, i):
    return mujoco.mj_id2name(m, kind, i) or ""


def sensor(env, which="force"):
    m, d = env.sim.model._model, env.sim.data._data
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gripper0_right_%s_ee" % which)
    return d.sensordata[m.sensor_adr[sid] : m.sensor_adr[sid] + 3].copy()


def _is_robot_geom(m, g):
    return name_of(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]).startswith(("robot0", "gripper0"))


def robot_external_contacts(env):
    """Contacts between robot0/gripper0 geoms and anything that is not the robot (table, objects, arena)."""
    m, d = env.sim.model._model, env.sim.data._data
    return [
        (name_of(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), name_of(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))
        for c in d.contact[: d.ncon]
        if _is_robot_geom(m, c.geom1) != _is_robot_geom(m, c.geom2)
    ]


def eef_site_pos(env):
    return env.robots[0].sim.data.get_site_xpos(env.robots[0].gripper["right"].important_sites["grip_site"]).copy()


def settle(env, n=30):
    for _ in range(n):
        env.step(HOLD)


# ------------------------------------------------------------------------------------------------------------------
# interface
# ------------------------------------------------------------------------------------------------------------------
def test_action_dim_is_7_and_obs_keys_match_panda(env):
    assert env.action_dim == 7  # OSC_POSE delta (6) + gripper (1), like Panda
    obs = env.reset()
    panda = make_env(robot="Panda")
    panda_obs = panda.reset()
    panda.close()
    for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "force"):
        assert key in obs, key
    # every Panda observation key exists with the same shape, except the gripper qpos / qvel (8 joints vs 2) and the
    # derived proprio vector that concatenates them; extra keys (robot0_base_*) are additions, not renames
    for key, val in panda_obs.items():
        assert key in obs, "missing Panda key %s" % key
        if key in ("robot0_gripper_qpos", "robot0_gripper_qvel"):
            assert obs[key].shape == (8,) and val.shape == (2,)
        elif key != "robot0_proprio-state":
            assert obs[key].shape == val.shape, key
    assert all(np.all(np.isfinite(v)) for v in obs.values())


def test_sensors_and_cameras(env):
    env.reset()
    m = env.sim.model._model
    names = [name_of(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)]
    assert names == ["gripper0_right_force_ee", "gripper0_right_torque_ee"]
    cams = [name_of(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
    assert "robot0_eye_in_hand" in cams and "robot0_robotview" in cams  # (camera obs need GL: not rendered here)
    settle(env)
    assert np.all(np.isfinite(sensor(env, "force"))) and np.all(np.isfinite(sensor(env, "torque")))


def test_frozen_links_constant_under_random_actions(env):
    env.reset()
    m = env.sim.model._model
    arm1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_arm_right_1_link")

    def under_arm1(b):
        while b > 0:
            if b == arm1:
                return True
            b = m.body_parentid[b]
        return False

    frozen = [
        b
        for b in range(1, m.nbody)
        if name_of(m, mujoco.mjtObj.mjOBJ_BODY, b).startswith("robot0") and not under_arm1(b)
    ]
    names = {name_of(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in frozen}
    # base, wheels, torso (lift, head), head, the whole left arm and the left gripper are all frozen
    for expected in (
        "robot0_torso_lift_link",
        "robot0_head_1_link",
        "robot0_arm_left_7_link",
        "robot0_wheel_front_left_link",
    ):
        assert expected in names, expected
    d = env.sim.data._data
    pos0, quat0 = d.xpos[frozen].copy(), d.xquat[frozen].copy()
    rng = np.random.default_rng(0)
    for _ in range(100):
        env.step(rng.uniform(-1, 1, env.action_dim))
    d = env.sim.data._data
    assert np.abs(d.xpos[frozen] - pos0).max() < 1e-9
    assert np.abs(d.xquat[frozen] - quat0).max() < 1e-9
    # and there are no joints other than the 7 right arm joints + the gripper (+ objects)
    robot_joints = [name_of(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    assert [j for j in robot_joints if j.startswith("robot0")] == ["robot0_arm_right_%d_joint" % i for i in range(1, 8)]


# ------------------------------------------------------------------------------------------------------------------
# control
# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_osc_delta_moves_eef_along_axis(env, axis):
    """+x/+y/+z delta of 1.0 for 20 steps (1 s). The OSC goal is re-anchored on the current pose each step, so the
    achieved displacement is a tracking-limited velocity; the per-step command is 0.05 m (1.0 m in total)."""
    obs = env.reset()
    p0, q0 = obs["robot0_eef_pos"].copy(), obs["robot0_eef_quat"].copy()
    a = HOLD.copy()
    a[axis] = 1.0
    for _ in range(20):
        obs, *_ = env.step(a)
    dp = obs["robot0_eef_pos"] - p0
    rot = np.degrees(np.linalg.norm(T.quat2axisangle(T.quat_multiply(obs["robot0_eef_quat"], T.quat_inverse(q0)))))
    lateral = np.delete(dp, axis)
    print(
        "\nOSC %s: moved %s m in 1 s (%.3f m/step of 0.05 commanded), lateral drift %s, orientation drift %.1f deg"
        % ("xyz"[axis], np.round(dp, 3), dp[axis] / 20, np.round(lateral, 3), rot)
    )
    assert dp[axis] >= 0.1
    assert np.abs(lateral).max() < 0.03
    assert rot < 10.0


def test_zero_action_holds_pose_200_steps(env):
    env.reset()
    p0 = eef_site_pos(env)
    for _ in range(200):
        env.step(HOLD)
    assert np.linalg.norm(eef_site_pos(env) - p0) < 5e-3


def _finger(env):
    return float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint")])


def test_gripper_sign_and_monotonic(env):
    """robosuite: +1 closes, -1 opens. PAL: finger_joint 0 = closed, 0.07 = open."""
    env.reset()
    a = HOLD.copy()
    q = [_finger(env)]
    assert q[0] > 0.065  # starts open (init_qpos)
    a[-1] = 1.0
    for _ in range(20):  # 1 s
        env.step(a)
        q.append(_finger(env))
    q = np.array(q)
    assert q[-1] < 0.02, q  # closed within 1 s (fingertips meet slightly above the 0.0 joint limit)
    assert np.all(np.diff(q) < 1e-3), "closing not monotonic"
    ctrl_adr = env.robots[0]._ref_joint_gripper_actuator_indexes["right"][0]
    assert env.sim.data.ctrl[ctrl_adr] == pytest.approx(0.0, abs=1e-9)  # closed command is ctrl 0
    a[-1] = -1.0
    q2 = [q[-1]]
    for _ in range(20):
        env.step(a)
        q2.append(_finger(env))
    q2 = np.array(q2)
    assert q2[-1] > 0.065, q2
    assert np.all(np.diff(q2) > -1e-3), "opening not monotonic"
    assert env.sim.data.ctrl[ctrl_adr] == pytest.approx(0.07, abs=1e-9)
    # action 0 holds the current command
    a[-1] = 1.0
    for _ in range(3):
        env.step(a)
    c = env.sim.data.ctrl[ctrl_adr]
    a[-1] = 0.0
    env.step(a)
    assert env.sim.data.ctrl[ctrl_adr] == pytest.approx(c)


# ------------------------------------------------------------------------------------------------------------------
# force / torque sensor
# ------------------------------------------------------------------------------------------------------------------
def test_ft_bias_taken_after_settle_vs_reset_time(env):
    """NutAssembly takes the bias right after _reset_internal (a transient, not the weight). Report how large the
    offset is on this robot and on Panda, and require it to stay well under the 10 N contact threshold."""
    res = {}
    for robot in (ROBOT, "Panda"):
        e = env if robot == ROBOT else make_env(robot=robot)
        e.reset()
        bias_reset = np.asarray(e._bias_F_sensor, dtype=np.float64).copy()
        settle(e, 40)
        f1 = sensor(e)
        settle(e, 20)
        f2 = sensor(e)
        assert np.linalg.norm(f1 - f2) < 1e-3, "force not stationary after 40 zero-action steps"
        res[robot] = (np.linalg.norm(f1), np.linalg.norm(f1 - bias_reset))
        if robot != ROBOT:
            e.close()
    print(
        "\nF/T bias: %s |F_settled| %.2f N, motionless |F_settled - F_reset_bias| %.2f N | Panda %.2f N, %.2f N "
        "(threshold %.0f N)" % (ROBOT, *res[ROBOT], *res["Panda"], FT_THRESHOLD)
    )
    # statics: the wrist sensor carries the gripper (+ fingers): m*g
    m = env.sim.model._model
    bodies = [b for b in range(m.nbody) if name_of(m, mujoco.mjtObj.mjOBJ_BODY, b).startswith("gripper0")]
    mass = m.body_mass[bodies].sum()
    assert res[ROBOT][0] == pytest.approx(mass * 9.81, rel=0.02)
    assert res[ROBOT][1] < 0.5 * FT_THRESHOLD  # >= 50% margin even with the reset-time bias


def test_ft_wrist_tilt_sweep_free_space(env):
    """Gravity re-projection: after a wrist tilt by theta the (settled) force changes by ~2 m g sin(theta/2) with no
    contact. Yaw about the approach axis must not matter."""
    env.reset()
    settle(env, 30)
    obs, *_ = env.step(HOLD)
    R0 = T.quat2mat(obs["robot0_eef_quat"])
    F0 = sensor(env)

    def rotate(Rt, n=60):
        for _ in range(n):
            o, *_ = env.step(HOLD)
            err = T.quat2axisangle(T.mat2quat(Rt @ T.quat2mat(o["robot0_eef_quat"]).T))
            a = HOLD.copy()
            a[3:6] = np.clip(err / 0.5, -1, 1)
            env.step(a)

    table = {}
    for name, axis, degs in (
        ("tilt_x", [1, 0, 0], (10, 20, 30, 45)),
        ("tilt_y", [0, 1, 0], (10, 20, 30, 45)),
        ("yaw", [0, 0, 1], (45, 90)),
    ):
        for deg in degs:
            for sgn in (1, -1):
                env.reset()
                settle(env, 30)
                rotate(T.quat2mat(T.axisangle2quat(np.array(axis) * np.radians(sgn * deg))) @ R0)
                settle(env, 30)
                assert not robot_external_contacts(env)
                table[(name, sgn * deg)] = np.linalg.norm(sensor(env) - F0)

    def worst(names, lim):
        return max(v for (n, d), v in table.items() if n in names and abs(d) <= lim)

    tilts = ("tilt_x", "tilt_y")
    print(
        "\nwrist sweep, max |F - F_settled_at_init| without contact: tilt<=20deg %.2f N, <=30deg %.2f N, <=45deg %.2f N, "
        "yaw<=90deg %.2f N (10 N threshold)" % (worst(tilts, 20), worst(tilts, 30), worst(tilts, 45), worst(("yaw",), 90))
    )
    # task-relevant tilts (<= 20 deg) must stay below 30% of the threshold; yaw is irrelevant for a vertical approach
    assert worst(tilts, 20) < 0.3 * FT_THRESHOLD
    assert worst(("yaw",), 90) < 0.5


def test_ft_free_space_fast_motion(env):
    """Scripted free-space motion (no contact): bias-corrected |F|. Reports Panda's value for the same script."""

    def run(e, scale):
        e.reset()
        bias_reset = np.asarray(e._bias_F_sensor, dtype=np.float64).copy()
        settle(e, 20)
        bias = sensor(e)
        m_settled = m_reset = 0.0
        for i in range(200):
            a = np.zeros(7)
            a[0], a[1], a[2] = np.sin(i * 0.5), np.cos(i * 0.4), 0.5 * np.sin(i * 0.3)
            a[3:6] = 0.5 * np.sin(i * 0.35 + np.array([0.0, 1.0, 2.0]))
            a[:6] *= scale
            a[6] = -1.0
            e.step(a)
            f = sensor(e)
            m_settled = max(m_settled, np.linalg.norm(f - bias))
            m_reset = max(m_reset, np.linalg.norm(f - bias_reset))
            assert not robot_external_contacts(e)
        return m_settled, m_reset

    panda = make_env(robot="Panda")
    out = {s: (run(env, s), run(panda, s)) for s in (0.5, 1.0)}
    panda.close()
    for s, (tp, pa) in out.items():
        print(
            "\nfree-space motion at %.0f%% action amplitude: max |F - F_settled| / |F - F_reset_bias| = %.2f / %.2f N "
            "(%s) vs %.2f / %.2f N (Panda), threshold %.0f N" % (100 * s, *tp, ROBOT, *pa, FT_THRESHOLD)
        )
    # moderate speed: >= 50% margin to the threshold with the settled bias. Full-amplitude oscillation exceeds it on this
    # (heavier) gripper; that is reported above, not asserted: see tools/tiago_pro/README.md
    assert out[0.5][0][0] < 0.5 * FT_THRESHOLD
    assert all(np.isfinite(v) for tp, pa in out.values() for v in tp + pa)


# ------------------------------------------------------------------------------------------------------------------
# init pose, contacts, speed
# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("task,table_top", [("NutAssemblySquare", 0.82), ("ToolHang", 0.80), ("Lift", 0.8)])
def test_init_pose(task, table_top):
    e = make_env(task)
    try:
        e.reset()
        assert e.action_dim == 7
        assert not robot_external_contacts(e), robot_external_contacts(e)
        p = eef_site_pos(e)
        assert 0.0 < p[2] - table_top < 0.2, p  # eef within 0.2 m above the table top (no init noise)
        m, d = e.sim.model._model, e.sim.data._data
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")
        tilt = np.degrees(np.arccos(np.clip(-d.site_xmat[sid].reshape(3, 3)[2, 2], -1, 1)))
        assert tilt < 5.0, tilt  # approach axis straight down (the "default" init noise adds a few degrees)
        obs = e.reset()
        for _ in range(20):
            obs, *_ = e.step(HOLD)
        assert all(np.all(np.isfinite(v)) for v in obs.values())
    finally:
        e.close()


def test_init_pose_with_default_init_noise_has_no_contacts_and_small_tilt():
    e = make_env("NutAssemblySquare", initialization_noise="default")
    try:
        for _ in range(5):
            e.reset()
            assert not robot_external_contacts(e)
            m, d = e.sim.model._model, e.sim.data._data
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")
            tilt = np.degrees(np.arccos(np.clip(-d.site_xmat[sid].reshape(3, 3)[2, 2], -1, 1)))
            assert tilt < 10.0, tilt
    finally:
        e.close()


def test_wipe_builds_with_wiping_gripper_and_is_6d():
    """Wipe forces WipingGripper (dof 0): the robot builds, but the action space is 6-D, not 7-D (PalProGripper is
    never instantiated there), and the sensor/geometry come from the Panda-flange WipingGripper."""
    e = make_env("Wipe")
    try:
        obs = e.reset()
        assert e.action_dim == 6
        assert not robot_external_contacts(e)
        for _ in range(5):
            obs, *_ = e.step(np.zeros(6))
        assert all(np.all(np.isfinite(v)) for v in obs.values())
    finally:
        e.close()


def test_no_always_on_self_contacts():
    """Random arm configurations within joint limits: no robot-robot geom pair is in contact in (almost) every sample,
    and the init pose has no self contacts (contact excludes of the generator are sufficient)."""
    e = make_env("Lift")
    try:
        e.reset()
        m = e.sim.model._model
        d = mujoco.MjData(m)
        d.qpos[:] = e.sim.data._data.qpos
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot0_arm_right_%d_joint" % i) for i in range(1, 8)]
        adr = [m.jnt_qposadr[j] for j in jid]
        lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
        rng = np.random.default_rng(0)
        n = 500
        count = collections.Counter()
        for _ in range(n):
            d.qpos[adr] = rng.uniform(lo, hi)
            mujoco.mj_forward(m, d)
            pairs = {
                tuple(
                    sorted(
                        (
                            name_of(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1]),
                            name_of(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom2]),
                        )
                    )
                )
                for c in d.contact[: d.ncon]
                if _is_robot_geom(m, c.geom1) and _is_robot_geom(m, c.geom2)
            }
            count.update(pairs)
        worst = count.most_common(1)[0] if count else (None, 0)
        print("\nmost frequent self-contact pair over %d random configurations: %s (%.1f%%)" % (n, worst[0], 100 * worst[1] / n))
        always_on = [p for p, c in count.items() if c / n >= 0.5]
        assert not always_on, always_on
        e.reset()
        d_ = e.sim.data._data
        assert not [c for c in d_.contact[: d_.ncon] if _is_robot_geom(m, c.geom1) and _is_robot_geom(m, c.geom2)]
    finally:
        e.close()


def _steps_per_second(robot, n=150):
    e = make_env(robot=robot)
    e.reset()
    rng = np.random.default_rng(0)
    for _ in range(20):
        e.step(rng.uniform(-1, 1, e.action_dim))
    t = time.perf_counter()
    for _ in range(n):
        e.step(rng.uniform(-1, 1, e.action_dim))
    rate = n / (time.perf_counter() - t)
    e.close()
    return rate


def test_step_rate_relative_to_panda():
    # best of two to be robust against machine load; the absolute rate is machine dependent (relative to Panda here)
    tp = max(_steps_per_second(ROBOT) for _ in range(2))
    pa = max(_steps_per_second("Panda") for _ in range(2))
    print("\nenv.step/s at control_freq 20: %s %.1f, Panda %.1f (ratio %.2f)" % (ROBOT, tp, pa, tp / pa))
    assert tp / pa > 0.5


@pytest.mark.parametrize("robot", [ROBOT, "Panda"])
def test_ft_sensor_is_refreshed_by_env_step(robot):
    """MuJoCo 3.5.0: mj_step1/mj_step2 (robosuite lite_physics) left acceleration-stage sensors (force/torque) at their
    last mj_forward value (the reset-time reading) because the lazy `flg_rnepost` flag stayed set; MjSim.step2 clears it.
    For a motionless robot the reading after env.step must equal the reading after a fresh forward()."""
    e = make_env(robot=robot)
    try:
        e.reset()
        reset_reading = sensor(e)
        settle(e, 30)
        stepped = sensor(e)
        e.sim.forward()
        forwarded = sensor(e)
        assert np.abs(stepped - forwarded).max() < 1e-3, (stepped, forwarded)
        assert np.abs(stepped - reset_reading).max() > 0.3  # the reset-time reading is a transient
    finally:
        e.close()
