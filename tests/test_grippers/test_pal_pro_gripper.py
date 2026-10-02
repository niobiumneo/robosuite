"""
Tests for the PAL Pro gripper (TIAGo Pro): interface contract, action sign convention, and a real grasp+lift.

PAL convention: finger_joint 0 = CLOSED, 0.07 = OPEN. robosuite convention: action +1 = close, -1 = open.
"""
import mujoco
import numpy as np

from robosuite.models.grippers import GRIPPER_MAPPING, GripperTester, PalProGripper


def _tester(box_half=0.02, low=-0.04):
    gripper = PalProGripper()
    tester = GripperTester(
        gripper=gripper,
        pos="0 0 0.3",
        quat="0 0 1 0",
        gripper_low_pos=low,
        gripper_high_pos=0.01,
        box_size=[box_half] * 3,
        render=False,
    )
    tester.start_simulation()
    return tester


def _finger(tester):
    return float(tester.sim.data.qpos[tester.sim.model.get_joint_qpos_addr("gripper0_finger_joint")])


def test_registered_and_interface():
    assert GRIPPER_MAPPING["PalProGripper"] is PalProGripper
    g = PalProGripper()
    assert g.dof == 1
    assert len(g.actuators) == 1
    # one value per joint (8: commanded finger_joint + 7 mimic joints), dof is the actuator count
    assert len(g.joints) == 8
    assert len(g.init_qpos) == len(g.joints)
    assert g.init_qpos[0] == 0.07  # open
    for name in ("force_ee", "torque_ee"):
        assert name in g.important_sensors
    for name in ("grip_site", "grip_cylinder", "ee_x", "ee_y", "ee_z"):
        assert name in g.important_sites
    # important geoms exist in the xml
    names = {e.get("name") for e in g.root.iter("geom")}
    for group in g.important_geoms.values():
        assert group, "empty important geom group"
        assert all(n in names for n in group), group


def test_init_qpos_is_consistent_with_mimic_equalities():
    tester = _tester()
    sim, g = tester.sim, tester.gripper
    adr = [sim.model.get_joint_qpos_addr(j) for j in g.joints]
    # gripper starts at init_qpos? the xml default is 0, so set it and look at the equality residuals
    sim.data.qpos[adr] = g.init_qpos
    sim.forward()
    m, d = sim.model._model, sim.data._data
    eq = d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    assert eq.sum() >= 7
    assert np.abs(d.efc_pos[eq]).max() < 1e-3, np.abs(d.efc_pos[eq]).max()


def test_format_action_sign_and_range():
    g = PalProGripper()
    assert np.all(g.current_action == 0)  # robosuite resets to zeros: open
    # open at reset: normalized command +1 maps to ctrl 0.07 (top of ctrlrange)
    assert np.allclose(g.format_action(np.array([-1.0])), 1.0)
    prev = 1.0
    for _ in range(20):  # closing: command decreases monotonically to -1 (ctrl 0.0 = closed)
        out = float(g.format_action(np.array([1.0]))[0])
        assert out <= prev + 1e-12 and -1.0 <= out <= 1.0
        prev = out
    assert np.isclose(prev, -1.0)
    for _ in range(20):  # opening
        out = float(g.format_action(np.array([-1.0]))[0])
        assert out >= prev - 1e-12
        prev = out
    assert np.isclose(prev, 1.0)
    # action 0 holds
    g.format_action(np.array([1.0]))
    a = g.format_action(np.array([0.0]))
    assert np.allclose(a, g.format_action(np.array([0.0])))


def test_close_open_monotonic_in_sim():
    tester = _tester()
    tester.gripper_z_is_low = False
    # the xml default is qpos 0 (= closed); robots apply init_qpos (open) at reset, so do the same here
    adr = [tester.sim.model.get_joint_qpos_addr(j) for j in tester.gripper.joints]
    tester.sim.data.qpos[adr] = tester.gripper.init_qpos
    tester.sim.forward()
    q = []
    tester.gripper_is_closed = True
    for _ in range(300):
        tester.step()
        q.append(_finger(tester))
    q = np.array(q)
    assert q[0] > 0.06  # starts open
    assert q[-1] < 0.02, q[-1]  # closed on nothing (fingertips meet slightly above the 0.0 joint limit)
    assert np.all(np.diff(q) < 1e-3), "closing is not monotonic"
    tester.gripper_is_closed = False
    q2 = []
    for _ in range(300):
        tester.step()
        q2.append(_finger(tester))
    q2 = np.array(q2)
    assert q2[-1] > 0.065, q2[-1]
    assert np.all(np.diff(q2) > -1e-3), "opening is not monotonic"
    # ctrl range in the model is the physical finger range (0 closed .. 0.07 open)
    cr = tester.sim.model.actuator_ctrlrange[tester.gripper_actuator_ids][0]
    assert np.allclose(cr, [0.0, 0.07])


def test_pal_pro_grasps_and_lifts_box():
    """The stock GripperTester.loop test_y check is vacuous (compares absolute height), so check the lift explicitly."""
    tester = _tester(box_half=0.02, low=-0.04)
    plan = [(False, False), (True, False), (True, True), (False, True)]
    rest_z = None
    for low, closed in plan:
        tester.gripper_z_is_low, tester.gripper_is_closed = low, closed
        for _ in range(400):
            tester.step()
        if (low, closed) == (True, False):
            rest_z = float(tester.sim.data.body_xpos[tester.object_id][2])
    lift = float(tester.sim.data.body_xpos[tester.object_id][2]) - rest_z
    finger = _finger(tester)
    assert lift > 0.01, "box not lifted: %.4f" % lift
    # fingers stopped on the box (4 cm wide), not at the closed limit
    assert 0.015 < finger < 0.06, finger
