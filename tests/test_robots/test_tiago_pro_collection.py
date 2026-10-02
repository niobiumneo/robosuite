"""
Scripted data collection with TiagoProRight through the fork's own collection code path
(``DataCollectionWrapperWithFT`` + ``gather_demonstrations_as_hdf5`` of
``robosuite/scripts/collect_human_demonstrations_force_vision.py``), without a human input device or a GL context.

$ pytest -s tests/test_robots/test_tiago_pro_collection.py

Documents what is and is not reproducible from a stored demo (see tools/tiago_pro/scripted_collect_smoke.py):
re-stepping the stored actions is; the state replay of ``augment_dataset_with_force.py`` (set_state + forward, stale
``data.ctrl``) is not, so its force/label error is reported, not asserted to be zero.
"""
import importlib.util
import os

import pytest

h5py = pytest.importorskip("h5py")

import robosuite

REPO = os.path.abspath(os.path.join(os.path.dirname(robosuite.__file__), ".."))
TOOL = os.path.join(REPO, "tools", "tiago_pro", "scripted_collect_smoke.py")


@pytest.fixture(scope="module")
def smoke():
    if not os.path.exists(TOOL):
        pytest.skip("tools/tiago_pro is not part of the installed package")
    spec = importlib.util.spec_from_file_location("scripted_collect_smoke", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiago_demo(smoke, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("tiago_demo"))
    path, stats = smoke.collect("TiagoProRight", out, episodes=1)
    return path, stats, smoke


def test_scripted_grasp_lifts_the_nut_and_touches(tiago_demo):
    _, stats, _ = tiago_demo
    s = stats[0]
    assert s["phase"] == "done"
    assert s["nut_lift"] > 0.03, s  # the PAL gripper grasped the handle and lifted the nut
    assert s["peak_live_dF_N"] > 10.0, s  # pressing on the table/nut is visible above the 10 N threshold


def test_hdf5_content_and_reproducibility(tiago_demo):
    path, _, smoke = tiago_demo
    res = smoke.verify(path, "TiagoProRight")  # asserts: env_args robots, actions (T, 7), 6 trailing F/T columns, XML reload
    print("\n" + "\n".join("  %-42s %s" % kv for kv in res.items()))
    assert res["action_dim"] == 7 and res["xml_reload_ok"]
    # re-stepping the stored actions reproduces the trajectory bit-exactly and the force up to solver-warmstart noise
    assert res["restep_max_arm_joint_err_rad"] < 1e-9
    assert res["restep_vs_live_max_abs_dF_N"] < 1.0
    assert res["label_flip_rate_restep_vs_live"] == 0.0
    assert res["contact_step_exists_live"]
    # the state replay used by augment_dataset_with_force.py is NOT faithful: reported (no 1e-6 claim), but never better
    # than re-stepping
    assert res["state_replay_vs_live_max_abs_dF_N"] > res["restep_vs_live_max_abs_dF_N"]


def test_hdf5_force_columns_match_live_wrapper(tiago_demo):
    path, _, _ = tiago_demo
    import numpy as np

    with h5py.File(path, "r") as f:
        d = f["data/demo_1"]
        assert json_robots(f) in ("TiagoProRight", ["TiagoProRight"])
        assert d["actions"].shape[1] == 7
        assert d.attrs["state_dim_ft"] == 6
        ft = d["states"][:, -6:]
        assert np.all(np.isfinite(ft)) and np.abs(ft[:, :3]).max() > 0
        assert json_keys(d) == ["robot0_right"]  # same key as the Panda runs


def json_robots(f):
    import json

    return json.loads(f["data"].attrs["env_args"])["env_kwargs"]["robots"]


def json_keys(d):
    import json

    return json.loads(d.attrs["ft_keys_order"])


def test_panda_collection_still_produces_valid_hdf5(smoke, tmp_path):
    """Backwards compatibility: the unchanged Panda path still yields a valid demo (the scripted grasp itself is only
    tuned for the PAL gripper, so nothing is asserted about its success)."""
    import numpy as np

    path, _ = smoke.collect("Panda", str(tmp_path), episodes=1, max_steps=60)
    with h5py.File(path, "r") as f:
        d = f["data/demo_1"]
        assert d["actions"].shape[1] == 7
        assert d["states"].shape[1] == d.attrs["state_dim_original"] + 6
        assert np.all(np.isfinite(d["states"][()]))
        assert json_robots(f) in ("Panda", ["Panda"])
