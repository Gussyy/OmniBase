"""The robot model against the robot: measured link positions, not the model's own opinion.

Every other check in this repository compares the chain with itself -- forward kinematics
against inverse, the export against the solver -- and all of them passed for a day while the
tool point sat on the gripper's servo and the jaws were assumed to reach along an axis they do
not reach along. This is the check that would have failed: the simulator's gripper_base and
fingertip-frame positions, read back after writing joint angles into the articulation
(so101-scene/scripts/joint_convention_check.py), against what the chain says for the same
angles. If you change a robot's joints, tool or approach, re-measure and replace the table;
do not fit the table to the model.

Run: python tests/test_robots.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import omnibase as ob  # noqa: E402
from omnibase.chain import Chain  # noqa: E402
from omnibase.robots import SO101_APPROACH, SO101_JOINTS, SO101_TOOL  # noqa: E402

# (name, joint angles the simulator actually held in degrees, gripper_base origin, gripper_frame_link
# origin) -- both relative to the robot's root, metres, Isaac Sim so101_full asset, 2026-09-09.
SO101_SIM = [
    ("zero", [-0.0, 0.0, 0.0, 0.0, 0.0], [0.413, -0.0, 0.234], [0.413, 0.008, 0.136]),
    ("pan+45", [42.6, 0.0, 0.0, 0.0, 0.0], [0.314, -0.254, 0.234], [0.319, -0.247, 0.136]),
    ("lift-45", [-0.0, -42.7, -0.7, -0.1, -0.0], [0.239, -0.0, 0.439], [0.306, 0.008, 0.368]),
    ("elbow+45", [-0.0, 0.4, 42.6, 0.2, 0.0], [0.332, -0.0, 0.017], [0.265, 0.008, -0.055]),
    ("wflex+45", [-0.0, 0.3, 0.5, 40.1, 0.0], [0.37, -0.0, 0.114], [0.305, 0.008, 0.04]),
    ("wroll+45", [0.0, 0.0, 0.1, 0.0, 39.3], [0.413, -0.0, 0.234], [0.413, 0.068, 0.163]),
    ("ready", [-0.0, -64.9, 81.6, 26.3, 0.0], [0.242, -0.0, 0.032], [0.175, 0.008, -0.039]),
]
FINGER_LEN = 0.098          # gripper_base -> gripper_frame_link along the fingers, measured


def _worst(chain, approach):
    """Worst tool-point distance (m) and approach-axis angle (deg) over the table."""
    wp = wa = 0.0
    for _, deg, base, tip in SO101_SIM:
        base, tip = np.asarray(base), np.asarray(tip)
        fingers = (tip - base) / np.linalg.norm(tip - base)
        tool_sim = base + np.linalg.norm(chain.tool) * fingers        # the same distance out along the real fingers
        tool_chain, M = chain.fk(np.radians(np.asarray(deg, dtype=float))[None])
        wp = max(wp, float(np.linalg.norm(tool_chain[0] - tool_sim)))
        aim = M[0] @ np.asarray(approach, dtype=float)
        wa = max(wa, float(np.degrees(np.arccos(np.clip(aim @ fingers, -1.0, 1.0)))))
    return wp, wa


def test_the_chain_agrees_with_the_simulator():
    """Tool point within 12 mm and jaws within 8 degrees of the measured fingers, every pose.

    The fingertip frame is not exactly on the finger axis (it sits 8 mm off), so 8 degrees is
    the measurement's own slack, not the model's. 90 is what the wrong frame gives.
    """
    wp, wa = _worst(ob.so101(), SO101_APPROACH)
    assert wp < 0.012, f"tool point {1000 * wp:.0f} mm from the simulator's fingers at some pose"
    assert wa < 8.0, f"jaws aimed {wa:.0f} deg from the simulator's fingers at some pose"


def test_the_check_has_teeth():
    """The frame this replaced -- tool along -y, jaws along +y -- must fail it by a mile."""
    old = Chain(SO101_JOINTS, tool=(0.0, -0.0748, 0.0), name="so101-old")
    wp, wa = _worst(old, (0.0, 1.0, 0.0))
    assert wp > 0.05 and wa > 60.0, f"a wrong frame passed: {1000 * wp:.0f} mm, {wa:.0f} deg"


def test_the_terminal_origin_is_the_gripper_base():
    """Zero tool offset must land on gripper_base itself: that is what the table is relative to."""
    bare = Chain(SO101_JOINTS, tool=(0.0, 0.0, 0.0), name="so101-bare")
    for _, deg, base, _tip in SO101_SIM:
        pos, _ = bare.fk(np.radians(np.asarray(deg, dtype=float))[None])
        d = float(np.linalg.norm(pos[0] - np.asarray(base)))
        assert d < 0.006, f"terminal origin {1000 * d:.0f} mm from gripper_base"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("\nall robot checks passed")
