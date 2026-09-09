"""Built-in arms, and how to describe your own.

Only one arm ships here, because only one has been checked against a running simulator. Adding
yours is a dozen lines: read the joint origins, rotations, axes and stops out of whatever
describes it, and hand them to :class:`~omnibase.chain.Chain`.

``Joint.from_usd`` takes exactly what a USD ``PhysicsRevoluteJoint`` prim carries
(``physics:localPos0``, ``physics:localRot0`` as (w, x, y, z), ``physics:axis``, and the limits
in degrees). ``Joint.from_urdf`` takes a URDF joint's ``origin xyz``, ``origin rpy``, ``axis``
and limits in radians. Both end up in the same place.

**Check it before you trust it.** A chain with a plausible-looking wrong number does not fail,
it quietly reports that your data is unreachable. The test suite here round-trips random joint
configurations through forward and inverse kinematics; do at least that much for a new arm, and
if you have the robot or a simulator, compare against poses it reports for known joint angles.
"""
from __future__ import annotations

import numpy as np

from .chain import Chain, Joint

#: SO-ARM101 with the parallel gripper, as Isaac Sim's SO-ARM101-FULL asset describes it.
#:
#: Validated against 52,666 logged (joint_pos, end-effector pose) pairs out of a running Isaac
#: Sim at 0.002 mm and 0.081 degrees. Note this does NOT agree with the SO-101 URDFs floating
#: around -- they disagree with the asset about the wrist by 120 mm -- so if you are driving a
#: real SO-101 rather than this simulation, measure yours.
SO101_JOINTS = [
    Joint.from_usd((0.0388353, -8.97657e-9, 0.0624),
                   (-1.7605304e-12, -1.3267949e-6, 1.0, 1.3267949e-6), "z",
                   -109.99988, 109.99988, "shoulder_pan"),
    Joint.from_usd((-0.0303992, -0.0182778, -0.0542),
                   (-0.49999815, 0.5, 0.5, 0.50000185), "z",
                   -100.000046, 100.000046, "shoulder_lift"),
    Joint.from_usd((-0.11257, -0.028, 0.0),
                   (-0.70710546, 0.0, 0.0, -0.7071081), "z",
                   -96.829865, 96.829865, "elbow_flex"),
    Joint.from_usd((-0.1349, 0.0052, 0.0),
                   (-0.70710546, 0.0, 0.0, 0.7071081), "z",
                   -94.99984, 94.99984, "wrist_flex"),
    Joint.from_usd((0.0, -0.181, 0.018),
                   (9.38184e-7, -0.7071081, 9.381874e-7, 0.70710546), "y",
                   -157.21103, 162.78934, "wrist_roll"),
]
#: The chain ends at the wrist_roll joint's frame, not at the gripper body's, and the two are not
#: the same frame. Measured in the simulator by writing joint angles and reading link positions
#: (so101-scene/scripts/joint_convention_check.py): in the terminal frame the fingertip frame
#: sits 98 mm along -z, the gripper servo -- the housing's top -- along -y, and the jaws open
#: along -/+x. In the gripper BODY's own frame (the one a recording of this gripper reports, the
#: one its wrist camera is placed in) the jaws reach along +y and the housing top is +z. This is
#: that body frame expressed in the terminal frame: columns are the body's x, y, z. It is its
#: own inverse. Every recording is brought through it on the way to the solver (data._apply_frame).
SO101_BODY_IN_CHAIN = ((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0))

#: The point between the jaws, in the chain's terminal frame -- not the wrist flange. Getting this
#: wrong shifts every reachability answer by however far out it is. 75 mm along the fingers, where
#: the pads meet; the fingertip frame is at 98.
SO101_TOOL = (0.0, 0.0, -0.0748)

#: The direction the jaws REACH, in the terminal frame: -z, see SO101_BODY_IN_CHAIN.
#:
#: This was (0, 1, 0) for a day, and that day's exports pointed the real fingers 90 degrees from
#: the human's. The +y was right -- for the gripper body's frame, which is what the recording
#: that "confirmed" it was reporting -- and wrong for the frame the solver actually works in.
#: A sign or axis error here does not fail loudly: the sweep is just as feasible, the export's
#: own forward-kinematics check passes, and the policy trained on it cannot grasp anything.
#: Measure it in the simulator, against link positions, not against a recording.
SO101_APPROACH = (0.0, 0.0, -1.0)


def so101():
    """The SO-ARM101 arm with its parallel gripper."""
    return Chain(SO101_JOINTS, tool=SO101_TOOL, name="so101")


BUILTIN = {"so101": so101}


def load(name):
    """A built-in arm by name."""
    if name not in BUILTIN:
        raise KeyError(f"no built-in arm {name!r}; have {sorted(BUILTIN)}. "
                       "Build your own with omnibase.chain.Chain -- see this module's docstring.")
    return BUILTIN[name]()


def describe(chain):
    """A short human summary of a chain, for checking you typed it in correctly."""
    lines = [f"{chain.name}: {chain.n} joints, tool offset "
             f"{np.round(chain.tool, 4).tolist()} m"]
    for j, (lo, hi) in zip(chain.joints, chain.limits):
        lines.append(f"  {j.name or '?':<16} axis {j.axis}  "
                     f"origin {np.round(j.origin, 4).tolist()}  "
                     f"limits [{np.degrees(lo):+7.1f}, {np.degrees(hi):+7.1f}] deg")
    reach = float(np.linalg.norm(chain.fk(np.zeros(chain.n))[0][0]))
    lines.append(f"  tool at all-zeros sits {reach * 1000:.0f} mm from the base origin")
    return "\n".join(lines)
