"""Checks on reading a recording whose numbers do not mean what this library's mean.

A hand-held rig writes its poses in its own axes, in whichever order it felt like writing three
Euler angles, measured at whatever point it happened to be tracking. None of that is in the
file. These build a recording where the answer is known -- a synthetic arm reaching down and
grasping on a tilted table, written out through a deliberately awkward convention -- and check
that the fit recovers it.

The ones that matter are the two that could pass while being wrong: a convention search that
picks by something the data cannot see, and a levelling that agrees with itself rather than with
the table.

Run with ``python -m pytest`` or ``python tests/test_frames.py``.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import omnibase as ob  # noqa: E402
from omnibase.data import _axes, _body_from, _upright, approach_axis, contacts, fit_level  # noqa: E402

try:
    import pandas as pd
except ImportError:                                        # pragma: no cover
    pd = None

#: The recording's own frame: up is not z, the fingers are not along -z, and the angles are
#: written intrinsic Z-Y-X. Nothing a reader could guess.
TRUE_UP = np.array([0.18, -0.10, 0.978])
TRUE_UP /= np.linalg.norm(TRUE_UP)
TRUE_FINGERS = np.array([0.30, 0.05, -0.95])
TRUE_FINGERS /= np.linalg.norm(TRUE_FINGERS)
TRUE_EULER = "ZYX"
TRUE_TCP = 0.09


def _episode(seed, cycles=3, per=14, spread=0.05):
    """One take: three pick-and-places on a tilted table. Poses of the TRACKER, not the jaws.

    Three rather than one, because two contact points do not determine a plane: centre them and
    they span a single direction, leaving the normal free to rotate about it. Real recordings
    have four to six, which is why the fit works on them and would not on a tidier fake.
    """
    rng = np.random.default_rng(seed)
    up, fing = TRUE_UP, TRUE_FINGERS
    ax = np.cross(up, [1.0, 0.0, 0.0])
    ax /= np.linalg.norm(ax)
    ay = np.cross(up, ax)
    table = rng.normal(0, 0.02, 3) @ np.stack([ax, ay, up])      # the rig moves between takes

    def spot():
        """Somewhere on the table, an object's half-height above it."""
        return (table + rng.uniform(-0.16, 0.16) * ax + rng.uniform(-0.16, 0.16) * ay
                + (0.03 + rng.uniform(0.0, spread)) * up)

    keys, hold = [], []
    for _ in range(cycles):
        grab, drop = spot(), spot()
        keys += [grab + 0.20 * up, grab, grab, drop + 0.20 * up, drop, drop]
        hold += [0, 0, 1, 1, 1, 0]
    pos, grip = [], []
    for i in range(len(keys) - 1):
        for t in np.linspace(0, 1, per, endpoint=False):
            pos.append(keys[i] * (1 - t) + keys[i + 1] * t)
            grip.append(0.05 if hold[i] else 1.0)
    pos, grip = np.array(pos), np.array(grip)

    # The jaws point at the table, roughly, and roll freely about their own axis.
    quats = []
    for p in pos:
        want = -up + rng.normal(0, 0.12, 3)
        want /= np.linalg.norm(want)
        v, c = np.cross(fing, want), float(fing @ want)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        M = np.eye(3) if np.linalg.norm(v) < 1e-9 else np.eye(3) + K + K @ K / (1 + c)
        quats.append(R.from_matrix(R.from_rotvec(want * rng.uniform(-np.pi, np.pi)).as_matrix()
                                   @ M).as_quat())
    quats = np.array(quats)
    # What is written down is the TRACKER, a fixed distance back along the fingers from the jaws.
    tracker = pos - R.from_quat(quats).as_matrix() @ (TRUE_TCP * fing)
    return np.column_stack([tracker, R.from_quat(quats).as_euler(TRUE_EULER), grip]), pos


def _dataset(root, episodes=24, cycles=3, spread=0.05):
    """A v2.1 dataset in the rig's own dialect: x,y,z,roll,pitch,yaw,gripper."""
    root = Path(root)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "fps": 20, "total_episodes": episodes,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {"action": {"dtype": "float32", "shape": [7], "names": {"motors": names}}},
    }), encoding="utf-8")
    d = root / "data" / "chunk-000"
    d.mkdir(parents=True, exist_ok=True)
    truth = {}
    for e in range(episodes):
        a, jaws = _episode(e, cycles, spread=spread)
        truth[e] = jaws
        pd.DataFrame({"action": list(a), "episode_index": np.full(len(a), e, dtype=np.int64),
                      "frame_index": np.arange(len(a))}).to_parquet(
            d / f"episode_{e:06d}.parquet")
    return root, truth


def _skip():
    if pd is None:
        print("  (skipped: pandas not installed)")
        return True
    return False


def test_axis_spec_refuses_a_mirror():
    """A left-handed spec is a bug that looks like a left hand. It must not be accepted."""
    assert np.allclose(_axes("x,-z,y") @ np.array([0, 0, 1.0]), [0, -1, 0])
    for bad in ("x,y,y", "x,y,-z", "x,y", "x,y,q"):
        try:
            _axes(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} is not a rotation and was accepted")


def test_upright_takes_up_to_z():
    for v in ([0, 0, 1.0], [0, 0, -1.0], [0.2, -0.1, 0.97], [1.0, 0, 0]):
        v = np.array(v, dtype=float) / np.linalg.norm(v)
        L = _upright(v)
        assert np.allclose(L @ v, [0, 0, 1], atol=1e-9), f"{v} did not land on +z"
        assert np.allclose(L @ L.T, np.eye(3), atol=1e-9) and np.linalg.det(L) > 0


def test_euler_columns_read_as_poses():
    """roll/pitch/yaw is a pose like any other -- and `left_wrist_roll` still is not."""
    if _skip():
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        root, _ = _dataset(tmp / "rig", episodes=2, cycles=1)
        raw = np.stack(pd.read_parquet(
            root / "data/chunk-000/episode_000000.parquet")["action"].to_numpy())
        ep = ob.load(root, episode=0, euler=TRUE_EULER)
        h = ep.hands[0]
        assert h.source == "pose" and h.grip is not None and len(h.pos) == len(raw)
        # what load() hands back is the chain's terminal frame: the recorded body turned by BODY
        want = (R.from_euler(TRUE_EULER, raw[:, 3:6]) * R.from_matrix(ob.data.BODY)).as_quat()
        assert np.allclose(np.abs((h.quat * want).sum(1)), 1.0, atol=1e-9)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_fit_recovers_a_frame_nobody_told_it():
    """The whole point: up, the fingers, the tool offset and the angle order, from the data.

    Every one of these is wrong by default and none is written down anywhere in the file.
    """
    if _skip():
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        root, truth = _dataset(tmp / "rig", episodes=24)
        lvl = fit_level(root, frame=None, eulers=("xyz", "zyx", "XYZ", "ZYX"),
                        tcps=(0.0, 0.03, 0.06, 0.09, 0.12), sample=24)
        assert lvl["euler"] == TRUE_EULER, f"read the angles as {lvl['euler']}, not {TRUE_EULER}"
        up_err = np.degrees(np.arccos(abs(float(np.array(lvl["up"]) @ TRUE_UP))))
        assert up_err < 5, f"up is {up_err:.1f} deg out"
        fing_err = np.degrees(np.arccos(abs(float(np.array(lvl["approach"]) @ TRUE_FINGERS))))
        assert fing_err < 10, f"the finger axis is {fing_err:.1f} deg out"
        # NOT the tool offset -- see test_the_tool_offset_is_only_as_sharp_as_the_objects.
        assert lvl["aligned"] > 0.85, f"consistency only {lvl['aligned']}"
        assert lvl["below"] < 0.03, f"{100 * lvl['below']:.1f}% of frames under the table"

        # And the levelled poses must be the jaws, on a table at zero, pointing down to grasp.
        ep = ob.load(root, episode=3, level=lvl)
        h = ep.hands[0]
        ev = contacts(h.grip, closing=True)
        assert (h.pos[:, 2] > -0.03).mean() > 0.98, "the hand goes under the table"
        assert abs(np.median(h.pos[ev, 2])) < 0.05, "grasps are not happening at the surface"
        jaw = R.from_quat(h.quat).as_matrix() @ np.asarray(ob.robots.SO101_APPROACH, dtype=float)
        down = np.degrees(np.arccos(np.clip(-jaw[ev, 2], -1, 1)))
        assert np.median(down) < 30, f"jaws {np.median(down):.0f} deg off the table at a grasp"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_wrong_angle_order_collapses():
    """The convention search has to separate, not merely prefer.

    If every candidate scored about the same it would be choosing at random while reporting a
    confident number. Most do collapse -- read the angles in a genuinely different order and no
    single body direction points at the table at every grasp. Two of the four do not collapse
    against each other: intrinsic Z-Y-X and extrinsic z-y-x describe nearly the same rotation
    for angles like these, so the winner beats its near-twin by a little and the other two by a
    lot. That is the real shape of the answer, so it is what is asserted."""
    if _skip():
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        root, _ = _dataset(tmp / "rig", episodes=16)
        scores = {}
        for eu in ("xyz", "zyx", "XYZ", "ZYX"):
            scores[eu] = fit_level(root, frame=None, eulers=(eu,), tcps=(0.09,),
                                   sample=16)["aligned"]
        assert max(scores, key=scores.get) == TRUE_EULER, f"picked the wrong one: {scores}"
        assert sum(v < 0.5 for v in scores.values()) >= 2,             f"nothing collapsed, so the statistic is not discriminating: {scores}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_tool_offset_is_only_as_sharp_as_the_objects():
    """The one thing the fit cannot promise, stated as a test rather than as a footnote.

    The tool offset is found by asking which offset makes grasps taken at different wrist angles
    land on the same surface. That works when the things being grasped are the same height, and
    stops working when they are not: a contact happens at table height plus half an object, so a
    box of assorted objects writes 5 cm of scatter over an effect worth about 1 cm. Both halves
    are checked here, because a method whose limits are only described drifts back into being
    trusted past them.
    """
    if _skip():
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        tcps = (0.0, 0.03, 0.06, 0.09, 0.12, 0.15)

        # Same object every time: the offset is the only thing left that can flatten the grasps.
        root, _ = _dataset(tmp / "same", episodes=20, cycles=3, spread=0.0)
        got = fit_level(root, frame=None, eulers=(TRUE_EULER,), tcps=tcps, sample=20)
        assert abs(got["tcp"] - TRUE_TCP) <= 0.03, \
            f"with objects all one height the offset should come out: {got['tcp']} vs {TRUE_TCP}"

        # Assorted objects: it does not, and that is the documented ceiling, not a regression.
        root, _ = _dataset(tmp / "mixed", episodes=20, cycles=3, spread=0.05)
        loose = fit_level(root, frame=None, eulers=(TRUE_EULER,), tcps=tcps, sample=20)
        assert loose["flat"] > got["flat"], "assorted objects should scatter the contacts more"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_grasps_and_releases_are_told_apart():
    g = np.array([1, 1, 1, 0.1, 0.1, 0.1, 1, 1, 0.1, 1])
    assert contacts(g, closing=True).tolist() == [3, 8]
    assert contacts(g, closing=False).tolist() == [6, 9]
    assert contacts(g).tolist() == [3, 6, 8, 9]
    assert len(contacts(np.ones(20))) == 0, "a jaw that never moves has no contacts"


def test_body_rotation_is_a_rotation():
    B = _body_from([0.3, 0.05, -0.95], [0.1, 0.9, 0.2])
    assert np.allclose(B @ B.T, np.eye(3), atol=1e-9) and np.linalg.det(B) > 0
    # our +y must be the finger axis, in the rig's coordinates
    f = np.array([0.3, 0.05, -0.95]) / np.linalg.norm([0.3, 0.05, -0.95])
    assert np.allclose(B[1], f, atol=1e-9)


def test_approach_axis_needs_grasps_not_guesses():
    """With no contacts there is nothing to measure, and it must say so rather than invent one."""
    M = np.tile(np.eye(3), (10, 1, 1))
    axis, q = approach_axis([(np.zeros((10, 3)), M, np.ones(10))], [0, 0, 1.0])
    assert axis is None and q == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall frame checks passed")
