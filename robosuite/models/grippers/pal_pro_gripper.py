"""
PAL Robotics PAL Pro gripper (TIAGo Pro): one commanded prismatic joint driving a four-bar finger mechanism.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class PalProGripper(GripperModel):
    """
    PAL Pro gripper of the TIAGo Pro.

    The XML is generated from PAL's description by ``tools/tiago_pro/convert_tiago_pro.py`` (do not edit it by hand).
    It has 8 joints (the commanded ``finger_joint`` plus 7 mimic joints tied to it by joint equalities, so
    ``len(init_qpos) == len(joints) == 8``) but a single actuator, so ``dof == 1``.

    PAL convention: ``finger_joint`` = 0 is CLOSED and 0.07 is OPEN, while the robosuite convention is that an action
    of +1 closes and -1 opens. The position actuator has ctrlrange [0, 0.07] and the GRIP controller maps a normalized
    command ``a`` linearly onto it (-1 -> 0.0 = closed, +1 -> 0.07 = open), so :meth:`format_action` emits ``-closure``.

    ``current_action`` is the *closure fraction* in [0, 1]: robosuite resets it to zeros at every episode reset, which
    here means "open", consistent with ``init_qpos`` (unlike Panda, where zeros means half open).

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/pal_pro_gripper.xml"), idn=idn)

    def format_action(self, action):
        """
        Maps a continuous action into a normalized actuator command, integrating at ``speed`` per step.
        +1 => closing (finger_joint -> 0.0), -1 => opening (finger_joint -> 0.07).

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), 0.0, 1.0)
        # normalized actuator command in [-1, 1]: closure 0 -> +1 (open, ctrl 0.07), closure 1 -> -1 (closed, ctrl 0.0)
        return 1.0 - 2.0 * self.current_action

    @property
    def init_qpos(self):
        # [finger_joint, inner_left, outer_left, fingertip_left, finger_right, inner_right, outer_right, fingertip_right]
        # fully open, consistent with the joint equalities (see tests/test_robots/test_tiago_pro_consistency.py)
        return np.array([0.07, -0.5796, 0.5796, -0.5796, 0.0154, -0.5796, 0.5796, -0.5796])

    @property
    def speed(self):
        # closure fraction per control step: 10 steps (0.5 s at 20 Hz) from open to closed (Panda: 0.2 on a [-1, 1] range)
        return 0.1

    @property
    def dof(self):
        return 1

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["inner_finger_left_collision", "outer_finger_left_collision", "fingertip_left_collision"],
            "right_finger": [
                "inner_finger_right_collision",
                "outer_finger_right_collision",
                "fingertip_right_collision",
            ],
            "left_fingerpad": ["fingertip_left_collision"],
            "right_fingerpad": ["fingertip_right_collision"],
        }
