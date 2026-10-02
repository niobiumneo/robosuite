#!/usr/bin/env python
"""Scripted data-collection smoke test for TiagoProRight (no human input device, no GL).

Builds NutAssemblySquare with ``robots=[ROBOT]`` and the robot's default JSON controller, drives a scripted right-arm
approach -> descend -> close -> lift on the square-nut handle (ground-truth nut pose, P control in the OSC delta action
space), records it with the fork's ``DataCollectionWrapperWithFT`` and writes it with ``gather_demonstrations_as_hdf5``
(both from ``robosuite/scripts/collect_human_demonstrations_force_vision.py``, i.e. the code path the human collection
uses). Then verifies the hdf5 and re-steps the stored actions.

  python tools/tiago_pro/scripted_collect_smoke.py [--robot TiagoProRight] [--out DIR] [--episodes 1]

What is (not) reproducible from a stored demo, measured here:
  * ``states`` (flattened sim state [time, qpos, qvel]) + 6 trailing F/T columns recorded live (x1000 scaled, with the
    wrapper's own bias: first reading + EMA on ncon == 0).
  * robomimic's ``augment_dataset_with_force.py`` replays by ``set_state_from_flattened(states[t]); forward()`` and reads
    the sensor. ``data.ctrl`` is NOT part of the state, so the replayed reading uses a stale ctrl (the last one the env
    set), and the force sensor of the distal subtree depends on qacc, i.e. on ctrl. State replay therefore does NOT
    reproduce the live force to 1e-6: this script reports the state-replay error and the label flip rate instead.
  * Re-stepping the stored ACTIONS from ``states[0]`` is what is reproducible (same physics, same controller state
    after ``reset_to``): the script reports max |F_restep - F_live|.
"""
import argparse
import importlib.util
import json
import logging
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)
FT_SCALE = 1000.0  # DataCollectionWrapperWithFT stores force x 1000
THRESHOLD = 10.0  # N


def load_collect_module():
    path = os.path.join(os.path.dirname(suite.__file__), "scripts", "collect_human_demonstrations_force_vision.py")
    spec = importlib.util.spec_from_file_location("collect_human_demonstrations_force_vision", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class NutGraspPolicy:
    """approach above the handle -> align yaw -> descend -> close -> lift. Uses ground-truth nut pose from the env."""

    def __init__(self, env, hover=0.10, press=True):
        self.env, self.hover, self.press = env, hover, press
        self.phase, self.count = "approach", 0
        sim = env.sim
        self.handle = lambda: np.array(sim.data.get_site_xpos("SquareNut_handle_site"))
        self.nut = lambda: np.array(sim.data.get_body_xpos(env.nuts[0].root_body))

    def __call__(self, obs):
        eef = np.array(self.env.robots[0].sim.data.get_site_xpos(self.env.robots[0].gripper["right"].important_sites["grip_site"]))
        handle, nut = self.handle(), self.nut()
        R = T.quat2mat(obs["robot0_eef_quat"])
        finger_axis = R[:, 1]  # the grip site y axis is the finger opening direction
        phi = np.arctan2(finger_axis[1], finger_axis[0])
        d = handle[:2] - nut[:2]
        psi = np.arctan2(d[1], d[0]) + np.pi / 2  # fingers close across the handle
        yaw_err = (wrap_angle(psi - phi) + np.pi / 2) % np.pi - np.pi / 2
        a = np.zeros(7)
        a[6] = -1.0
        a[5] = np.clip(yaw_err / 0.5 * 1.5, -1, 1)
        xy_err = handle[:2] - eef[:2]
        k = 1.0 / 0.05 * 0.6  # 60% of the per-step motion limit per metre of error
        if self.phase == "approach":
            a[:2] = np.clip(k * xy_err, -1, 1)
            a[2] = np.clip(k * (handle[2] + self.hover - eef[2]), -1, 1)
            if np.linalg.norm(xy_err) < 0.008 and abs(yaw_err) < 0.08 and abs(eef[2] - handle[2] - self.hover) < 0.01:
                self.phase = "descend"
        elif self.phase == "descend":
            a[:2] = np.clip(k * xy_err, -1, 1)
            a[2] = np.clip(k * (handle[2] - eef[2] - 0.004), -0.5, 0.5)
            if eef[2] - handle[2] < 0.012:
                self.phase, self.count = "close", 0
        elif self.phase == "close":
            a[6] = 1.0
            self.count += 1
            if self.count >= 15:
                self.phase, self.count = ("press" if self.press else "lift"), 0
        elif self.phase == "press":  # push the closed gripper/nut onto the table: a deliberate contact for the F/T sensor
            a[6] = 1.0
            a[2] = -1.0
            self.count += 1
            if self.count >= 10:
                self.phase, self.count = "lift", 0
        elif self.phase == "lift":
            a[6] = 1.0
            a[2] = 0.6
            self.count += 1
            if self.count >= 20:
                self.phase = "done"
        else:
            a[6] = 1.0
        return a


def make_env(robot, controller_config, seed=0):
    return suite.make(
        env_name="NutAssemblySquare",
        robots=[robot],
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        reward_shaping=True,
        control_freq=20,
        initialization_noise=None,
        seed=seed,
    )


def collect(robot, out_dir, episodes=1, max_steps=250, seed=0):
    mod = load_collect_module()
    controller_config = load_composite_controller_config(controller=None, robot=robot)
    config = {"env_name": "NutAssemblySquare", "robots": [robot], "controller_configs": controller_config}
    env_info = json.dumps(config)  # exactly what the human collection script stores
    tmp = tempfile.mkdtemp(prefix="tiago_collect_")
    env = mod.DataCollectionWrapperWithFT(make_env(robot, controller_config, seed), tmp)
    stats = []
    try:
        for ep in range(episodes):
            obs = env.reset()
            policy = NutGraspPolicy(env.env)
            nut_z0 = policy.nut()[2]
            n_steps = peak = 0
            for t in range(max_steps):
                obs, _, _, _ = env.step(policy(obs))
                n_steps += 1
                if env.action_infos:
                    peak = max(peak, float(np.linalg.norm(env.action_infos[-1]["ee_force"]["robot0_right"])))
                if policy.phase == "done":
                    break
            stats.append(dict(steps=n_steps, phase=policy.phase, nut_lift=float(policy.nut()[2] - nut_z0), peak_live_dF_N=peak))
            env.close()  # flushes the episode (as collect_human_trajectory does)
        mod.gather_demonstrations_as_hdf5(tmp, out_dir, env_info, save_only_success=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return os.path.join(out_dir, "demo.hdf5"), stats


def verify(path, robot):
    """Checks of the produced hdf5 and the reproducibility analysis. Returns a dict of numbers; raises on failure."""
    out = {}
    with h5py.File(path, "r") as f:
        grp = f["data"]
        env_args = json.loads(grp.attrs["env_args"])
        robots = env_args["env_kwargs"]["robots"]
        assert (robots == robot) or (robots == [robot]), robots
        assert env_args["env_name"] == "NutAssemblySquare"
        assert env_args["env_kwargs"]["controller_configs"]["body_parts"]["right"]["type"] == "OSC_POSE"
        demos = sorted(k for k in grp.keys() if k.startswith("demo_"))
        assert demos
        d = grp[demos[0]]
        actions, states = d["actions"][()], d["states"][()]
        assert actions.ndim == 2 and actions.shape[1] == 7, actions.shape
        n_ft = int(d.attrs["state_dim_ft"])
        assert n_ft == 6 and states.shape[1] == int(d.attrs["state_dim_original"]) + 6
        assert states.shape[0] == actions.shape[0]
        ft = states[:, -6:]
        assert np.all(np.isfinite(ft))
        model_xml = d.attrs["model_file"]
        out.update(T=int(actions.shape[0]), state_dim=int(states.shape[1]), action_dim=int(actions.shape[1]),
                   peak_live_ft_F=float(np.abs(ft[:, :3]).max() / FT_SCALE))
        live_F = ft[:, :3] / FT_SCALE  # bias-corrected, scaled back to N
        sd = int(d.attrs["state_dim_original"])
        states0 = states[:, :sd]

    # ---- the stored XML reloads (mesh paths are re-rooted by edit_model_xml) from a different cwd
    cwd = os.getcwd()
    tmpd = tempfile.mkdtemp(prefix="tiago_reload_")
    try:
        os.chdir(tmpd)
        cfg = dict(env_args["env_kwargs"])
        env = suite.make(**cfg, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True,
                         reward_shaping=True, control_freq=20)
        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(states0[0])
        env.sim.forward()
        out["xml_reload_ok"] = True

        m, dta = env.sim.model._model, env.sim.data._data
        import mujoco

        fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gripper0_right_force_ee")
        adr = m.sensor_adr[fid]

        def sensor_f():
            return np.array(env.sim.data._data.sensordata[adr : adr + 3], dtype=np.float64)

        # (1) what augment_dataset_with_force.py does: set the stored state, forward, read the sensor
        replay = np.zeros((len(states0), 3))
        for t in range(len(states0)):
            env.sim.set_state_from_flattened(states0[t])
            env.sim.forward()
            replay[t] = sensor_f()
        # bias as augment_dataset_with_force.py uses it: the reading at t = 0
        replay_bias = replay - replay[0]

        # (2) re-step the stored actions from the stored initial state, exactly as robomimic's reset_to does
        #     (reset_from_xml_string + set_state_from_flattened + forward): this IS reproducible
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(states0[0])
        env.sim.forward()
        restep = np.zeros((len(states0), 3))
        arm_adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot0_arm_right_%d_joint" % i)] for i in range(1, 8)]
        arm_err = 0.0
        for t in range(len(actions)):
            env.step(actions[t])
            restep[t] = sensor_f()
            if t + 1 < len(states0):  # the stored state at index t+1 is the state reached after action t
                ref = states0[t + 1][1 : 1 + env.sim.model._model.nq]
                arm_err = max(arm_err, float(np.abs(env.sim.data._data.qpos[arm_adr] - ref[arm_adr]).max()))
        env.close()
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmpd, ignore_errors=True)

    # ---- compare, all in N and bias-corrected with the convention of the producer:
    #   live   : wrapper bias = the reading after the first step (ncon never 0 -> no EMA update), post-step state t
    #   restep : same convention, post-step reading after re-applying action t
    #   replay : augment_dataset_with_force.py: set_state(states[t]); forward(); bias = reading at t = 0 (pre-action state)
    mag = lambda x: np.linalg.norm(x, axis=1)
    restep_bias = restep - restep[0]
    live_bias = live_F
    n = len(actions)
    replay_post = replay[1:] - replay[0]  # replay reading of the state reached after action t, augment's bias convention
    out["restep_max_arm_joint_err_rad"] = arm_err
    out["live_max_dF_N"] = float(mag(live_bias).max())
    out["restep_max_dF_N"] = float(mag(restep_bias).max())
    out["restep_vs_live_max_abs_dF_N"] = float(np.abs(restep_bias - live_bias).max())
    out["state_replay_vs_live_max_abs_dF_N"] = float(np.abs(replay_post - live_bias[: n - 1]).max())
    out["state_replay_vs_live_median_abs_dF_N"] = float(np.median(np.abs(replay_post - live_bias[: n - 1])))
    out["label_rate_live"] = float((mag(live_bias) > THRESHOLD).mean())
    out["label_flip_rate_state_replay_vs_live"] = float(((mag(replay_post) > THRESHOLD) != (mag(live_bias[: n - 1]) > THRESHOLD)).mean())
    out["label_flip_rate_restep_vs_live"] = float(((mag(restep_bias) > THRESHOLD) != (mag(live_bias) > THRESHOLD)).mean())
    out["contact_step_exists_live"] = bool((mag(live_bias) > THRESHOLD).any())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="TiagoProRight")
    ap.add_argument("--out", default=None, help="output directory for demo.hdf5 (default: a temp dir, removed afterwards)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out_dir = a.out or tempfile.mkdtemp(prefix="tiago_demo_")
    os.makedirs(out_dir, exist_ok=True)
    try:
        path, stats = collect(a.robot, out_dir, a.episodes, seed=a.seed)
        print("episodes:", stats)
        res = verify(path, a.robot)
        for k, v in res.items():
            print("  %-42s %s" % (k, v))
    finally:
        if a.out is None:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
