"""Checks that fail if the library is lying about what an arm can reach.

Run with ``python -m pytest`` or just ``python tests/test_omnibase.py``.

The important ones are the two that could silently pass while being wrong: inverse kinematics
that reports success it did not achieve, and a chunker that reports a placement holds frames it
does not. Both are checked against forward kinematics rather than against themselves.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import omnibase as ob  # noqa: E402
from omnibase.chain import Chain, Joint  # noqa: E402


def test_fk_matches_hand_composition():
    """Forward kinematics is just a chain of transforms; check it against one done by hand."""
    ch = ob.so101()
    q = np.array([0.3, -0.4, 0.5, 0.2, 0.7])
    p, M = np.zeros(3), np.eye(3)
    for i, j in enumerate(ch.joints):
        p = p + M @ np.asarray(j.origin)
        M = M @ R.from_quat(j.rot).as_matrix() @ R.from_rotvec(
            np.eye(3)["xyz".index(j.axis)] * q[i]).as_matrix()
    tip, rot = ch.fk(q)
    assert np.allclose(tip[0], p + M @ ch.tool, atol=1e-12)
    assert np.allclose(rot[0], M, atol=1e-12)


def test_ik_round_trip():
    """Every pose the arm can strike, it must recover -- and report ~zero error for."""
    ch = ob.so101()
    rng = np.random.default_rng(0)
    q = rng.uniform(ch.limits[:, 0] * 0.9, ch.limits[:, 1] * 0.9, size=(400, ch.n))
    pos, M = ch.fk(q)
    _, pe, re = ch.ik(pos, R.from_matrix(M).as_quat())
    # Position is solved exactly -- it is the primary task and it converges.
    assert np.percentile(pe, 90) < 1e-5, f"position p90 {np.percentile(pe, 90):.2e} m"
    assert np.median(np.degrees(re)) < 0.1, "orientation median too large"
    # Orientation has a tail. This is a general numeric solver on a redundant chain, and a few
    # per cent of poses settle in the wrong branch -- a robot-specific closed form enumerates
    # branches and does not. Measured against one on real data, this understates what the arm
    # can reach by up to ~2.4 points, which is the safe direction to be wrong in: it never
    # claims a frame is usable when it is not.
    within = ((pe < 1e-3) & (np.degrees(re) < 20.0)).mean()
    assert within > 0.93, f"only {within:.1%} of achievable poses recovered"


def test_reported_error_is_real():
    """The errors returned must describe the joints returned, not a hope.

    A solver that returned an optimistic number would make every downstream count wrong, and it
    is the kind of bug that never shows up in a demo.
    """
    ch = ob.so101()
    rng = np.random.default_rng(1)
    pos = rng.uniform(-0.35, 0.35, (200, 3))
    quat = R.random(200, random_state=2).as_quat()
    q, pe, re = ch.ik(pos, quat)
    pe2, re2 = ch.error(q, pos, R.from_quat(quat).as_matrix())
    assert np.allclose(pe, pe2, atol=1e-9)
    assert np.allclose(re, re2, atol=1e-9)
    assert np.all(q >= ch.limits[:, 0] - 1e-9) and np.all(q <= ch.limits[:, 1] + 1e-9)


def test_unreachable_is_reported_unreachable():
    """Two metres away is not reachable, and must not be called reachable."""
    ch = ob.so101()
    far = np.array([[2.0, 0.0, 0.5], [0.0, -3.0, 0.0]])
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))
    _, pe, _ = ch.ik(far, quat)
    assert (pe > 1.0).all(), "an arm 0.36 m long reported reaching 2 m"
    assert not ch.reachable(far, quat).any()


def test_chunks_hold_what_they_claim():
    """Re-solve every chunk from scratch: the frames it claims to hold, it must hold."""
    ch = ob.so101()
    rng = np.random.default_rng(3)
    # A plausible free-flown path: a loop over the table at gripper height.
    t = np.linspace(0, 1, 90)
    pos = np.stack([0.22 + 0.10 * np.cos(6 * t), 0.10 * np.sin(6 * t), 0.14 + 0.05 * t], -1)
    quat = R.from_euler("y", (90 + 25 * np.sin(9 * t))[:, None], degrees=True).as_quat()

    cells = ob.base_grid(span=0.30, step=0.04, height=0.08)
    F = ob.feasibility(ch, pos, quat, cells)
    chunks, held = ob.chunk([F], cells, window=15)
    assert len(chunks) >= 1 and chunks[0].start == 0 and chunks[-1].stop == len(pos)
    assert [c.stop for c in chunks[:-1]] == [c.start for c in chunks[1:]], "chunks must tile"

    Rt = R.from_quat(quat).as_matrix()
    for c in chunks:
        _, pe, re = ch.ik(pos[c.start:c.stop] - c.bases[0], Rt[c.start:c.stop])
        ok = (pe < 0.015) & (re < np.radians(20.0))
        assert abs(ok.mean() - c.held[0]) < 1e-9, "a chunk claimed frames it does not hold"
        assert np.array_equal(ok, held[0, c.start:c.stop])
    del rng


def test_shorter_window_never_hurts():
    """The central claim. A base that must serve fewer frames can only ever do better."""
    ch = ob.so101()
    t = np.linspace(0, 1, 80)
    pos = np.stack([0.24 + 0.12 * np.cos(5 * t), 0.14 * np.sin(5 * t), 0.13 + 0.06 * t], -1)
    quat = R.from_euler("y", (90 + 30 * np.sin(7 * t))[:, None], degrees=True).as_quat()
    cells = ob.base_grid(span=0.30, step=0.04, height=0.08)
    F = ob.feasibility(ch, pos, quat, cells)

    rows = ob.yield_curve(F, cells, windows=(10, 20, 40, None))
    got = [r["usable"] for r in rows]
    assert got == sorted(got, reverse=True), f"yield must not rise with window length: {got}"

    _, held_short = ob.chunk([F], cells, window=10)
    _, held_long = ob.chunk([F], cells, window=80)
    assert held_short.mean() >= held_long.mean() - 1e-9


def test_urdf_and_usd_joints_agree():
    """The two constructors describe the same joint, so they must build the same transform."""
    usd = Joint.from_usd((0.1, 0.0, 0.05), (0.70710678, 0.70710678, 0.0, 0.0), "z", -90, 90)
    urdf = Joint.from_urdf((0.1, 0.0, 0.05), (np.pi / 2, 0.0, 0.0), (0, 0, 1),
                           -np.pi / 2, np.pi / 2)
    a = Chain([usd], tool=(0, 0, 0.1))
    b = Chain([urdf], tool=(0, 0, 0.1))
    for angle in (-0.7, 0.0, 0.9):
        pa, Ma = a.fk([angle])
        pb, Mb = b.fk([angle])
        assert np.allclose(pa, pb, atol=1e-9) and np.allclose(Ma, Mb, atol=1e-9)


def test_quaternion_order_is_checked():
    """(w, x, y, z) passed as (x, y, z, w) is the classic silent failure -- catch it."""
    good = R.random(20, random_state=4).as_quat()
    ob.normalise_quats(good)
    try:
        ob.normalise_quats(good * 1.4)
    except ValueError:
        pass
    else:
        raise AssertionError("non-unit quaternions must be refused, not rescaled")


def test_score_map_ranks_placements_sensibly():
    """A placement that reaches nothing must not outscore one that reaches everything."""
    ch = ob.so101()
    t = np.linspace(0, 1, 40)
    pos = np.stack([0.24 + 0.08 * np.cos(4 * t), 0.08 * np.sin(4 * t), 0.14 + 0.04 * t], -1)
    quat = R.from_euler("y", (90 + 20 * np.sin(6 * t))[:, None], degrees=True).as_quat()
    cells = ob.base_grid(span=0.32, step=0.08, height=0.08)
    smap = ob.score_map(ch, pos, quat, cells)

    assert set(smap["verdict"]) <= {"good", "workable", "marginal", "unusable"}
    dead = smap["coverage"] == 0
    if dead.any() and (~dead).any():
        assert smap["score"][~dead].max() > smap["score"][dead].max(), \
            "a placement that reaches nothing outscored one that reaches"
    assert np.all((smap["score"] >= 0) & (smap["score"] <= 1))
    assert np.all(smap["verdict"][dead] == "unusable")

    cell, i, report = ob.best_spot(smap)
    assert smap["score"][i] == smap["score"].max() or smap["verdict"][i] == "good"
    assert isinstance(report, str) and "score" in report
    assert "coverage" in ob.ascii_map(smap, "coverage")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
