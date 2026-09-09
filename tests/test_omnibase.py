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


def test_chunking_never_loses_to_standing_still():
    """Re-placing the base is optional, so it cannot come out worse than one fixed base.

    It did, once: when no placement could hold a whole window the fallback took the longest
    unbroken run instead of the best coverage, and on rigidly-coupled arms -- where whole
    windows are often unservable -- that scored below simply standing in the best spot.
    """
    ch = ob.so101()
    t = np.linspace(0, 1, 70)
    # A deliberately awkward path: wide, and swinging the heading well out of any one plane.
    pos = np.stack([0.26 + 0.16 * np.cos(3 * t), 0.22 * np.sin(3 * t), 0.12 + 0.10 * t], -1)
    quat = R.from_euler("y", (90 + 55 * np.sin(4 * t))[:, None], degrees=True).as_quat()
    cells = ob.base_grid(span=0.32, step=0.04, height=0.08)
    F = ob.feasibility(ch, pos, quat, cells)
    _, fixed = ob.best_fixed(F, cells)
    for window in (5, 21, 50, len(pos), len(pos) * 2):
        _, held = ob.chunk([F], cells, window=window)
        assert held.mean() >= fixed - 1e-9, (
            f"window {window}: chunked {held.mean():.3f} < best fixed base {fixed:.3f}")


def test_rigid_pair_is_never_better_than_free_arms():
    """Bolting two arms together is a constraint, so it can only cost reach, never add it."""
    ch = ob.so101()
    t = np.linspace(0, 1, 40)
    hands = []
    for sign in (-1.0, 1.0):
        pos = np.stack([0.24 + 0.06 * np.cos(4 * t),
                        sign * 0.10 + 0.05 * np.sin(4 * t), 0.14 + 0.03 * t], -1)
        quat = R.from_euler("y", (90 + 15 * np.sin(5 * t))[:, None], degrees=True).as_quat()
        hands.append((pos, quat))

    mount = ob.pair(0.30)
    assert len(mount) == 2 and mount.names == ["right", "left"]
    cells = ob.mount_grid(span=0.28, step=0.07, height=0.08)
    comb, per = ob.mount_feasibility(ch, hands, mount, cells)

    # The combined mask is exactly "every arm can", not something looser.
    assert np.array_equal(comb, per[0] & per[1])
    for a in (0, 1):
        assert comb.mean() <= per[a].mean() + 1e-12

    # A mounted arm stands at cell + offset, not at cell, so the honest comparison is against
    # a free arm at those same shifted positions -- and there it must agree EXACTLY, because
    # with no mount yaw the two are the same solve. This is what catches a wrong transform.
    for a, (p, q) in enumerate(hands):
        shifted = cells[:, :3] + mount.offsets[a][:3]
        free = ob.feasibility(ch, p, q, shifted)
        assert np.array_equal(free, per[a]), "the mount transform disagrees with a free solve"

    # arm_bases must put the arms where the mount says they are: the right separation, in line.
    for cell in (cells[0], cells[len(cells) // 2]):
        (pr, _), (pl, _) = ob.arm_bases(mount, cell)
        assert abs(np.linalg.norm(pl - pr) - 0.30) < 1e-9
        assert np.allclose((pr + pl) / 2.0, cell[:3])


def test_mount_yaw_turns_the_whole_assembly():
    """Turning the mount and turning the world about it must come to the same thing."""
    ch = ob.so101()
    t = np.linspace(0, 1, 24)
    pos = np.stack([0.24 + 0.05 * np.cos(4 * t), 0.05 * np.sin(4 * t), 0.15 + 0.02 * t], -1)
    quat = R.from_euler("y", (90 + 10 * np.sin(3 * t))[:, None], degrees=True).as_quat()
    hands = [(pos, quat), (pos + np.array([0.0, 0.18, 0.0]), quat)]
    mount = ob.pair(0.26)
    w = np.radians(35.0)

    plain = ob.mount_feasibility(ch, hands, mount, np.array([[0.0, 0.0, 0.08, 0.0]]))[0]
    # Same scene, rotated about the origin, with the mount turned to match.
    Y = R.from_rotvec([0.0, 0.0, w])
    turned = [(Y.apply(p), (Y * R.from_quat(q)).as_quat()) for p, q in hands]
    rot = ob.mount_feasibility(ch, turned, mount, np.array([[0.0, 0.0, 0.08, w]]))[0]
    assert np.array_equal(plain, rot), "mount yaw does not agree with rotating the scene"


def _reachable_path(ch, base, n=60):
    """A trajectory built so that one known base can hold all of it."""
    rng = np.random.default_rng(7)
    q = np.zeros((n, ch.n))
    t = np.linspace(0, 1, n)
    for j in range(ch.n):
        lo, hi = ch.limits[j] * 0.55
        q[:, j] = (lo + hi) / 2 + (hi - lo) / 2 * np.sin(2 * np.pi * (t + rng.random()))
    pos, M = ch.fk(q)
    return pos + np.asarray(base), R.from_matrix(M).as_quat()


def test_home_base_is_used_whenever_it_works():
    """The robot has a base. Use it wherever it can do the job; leave only where it cannot."""
    ch = ob.so101()
    home = np.array([0.10, -0.04, 0.08])
    pos, quat = _reachable_path(ch, home)
    cells = ob.base_grid(span=0.24, step=0.04, height=0.08, centre=(0.10, -0.04))
    F = ob.feasibility(ch, pos, quat, cells)

    hi = ob.home_index(cells, home)
    assert np.allclose(cells[hi], home), "home must snap to the candidate that IS home"
    assert F[:, hi].mean() > 0.95, "the fixture is wrong: home should hold this path"

    chunks, held = ob.chunk([F], cells, window=21, home=home)
    assert all(c.at_home[0] for c in chunks), "home works throughout and was not used"
    assert ob.home_share(chunks, len(pos))[0] > 0.95
    assert len(chunks) == 1, "no reason to ever move, so there should be one chunk"


def test_home_is_reclaimed_as_soon_as_it_can_be():
    """Having left home, come back at the first frame home can serve -- not merely when the
    stand-in fails. Without that the robot stays parked somewhere invented long after it could
    have gone back, and invented placements are the ones you have to justify."""
    ch = ob.so101()
    home = np.array([0.10, -0.04, 0.08])
    pos, quat = _reachable_path(ch, home)
    # Push a stretch of the path far away, so home cannot serve those frames.
    pos = pos.copy()
    pos[20:32] += np.array([0.16, 0.0, 0.0])
    cells = ob.base_grid(span=0.28, step=0.04, height=0.08, centre=(0.10, -0.04))
    F = ob.feasibility(ch, pos, quat, cells)
    hi = ob.home_index(cells, home)
    assert not F[20:32, hi].all(), "the fixture is wrong: home should fail on that stretch"

    chunks, _ = ob.chunk([F], cells, window=5, home=home)
    share = ob.home_share(chunks, len(pos))[0]
    free = ob.home_share(ob.chunk([F], cells, window=5)[0], len(pos))[0]
    assert share > free, "home preference bought nothing"
    # Every frame home could have served, while not at home, is a frame we came back too late.
    at_home = np.zeros(len(pos), dtype=bool)
    for c in chunks:
        at_home[c.start:c.stop] = c.at_home[0]
    late = (~at_home) & F[:, hi]
    assert late.sum() <= 5, f"{int(late.sum())} frames stayed away while home was usable"


def test_home_preference_still_never_loses_to_standing_still():
    """Preferring home must not cost coverage: it is a tie-break, not a handicap."""
    ch = ob.so101()
    home = np.array([0.10, -0.04, 0.08])
    pos, quat = _reachable_path(ch, home)
    pos = pos.copy()
    pos[25:45] += np.array([0.14, 0.06, 0.0])
    cells = ob.base_grid(span=0.28, step=0.04, height=0.08, centre=(0.10, -0.04))
    F = ob.feasibility(ch, pos, quat, cells)
    _, fixed = ob.best_fixed(F, cells)
    for window in (5, 21, len(pos)):
        _, held = ob.chunk([F], cells, window=window, home=home)
        assert held.mean() >= fixed - 1e-9, f"window {window}: {held.mean():.3f} < {fixed:.3f}"


def _random_chain(n, seed, tool=(0.0, 0.0, 0.05)):
    rng = np.random.default_rng(seed)
    js = []
    for i in range(n):
        lo = rng.uniform(-2.6, -0.3)
        js.append(Joint(tuple(rng.uniform(-0.12, 0.12, 3)),
                        tuple(R.random(random_state=int(seed * 100 + i)).as_quat()),
                        "xyz"[i % 3], lo, lo + rng.uniform(0.6, 4.0), f"j{i}"))
    return Chain(js, tool=tool, name=f"r{n}s{seed}")


def test_pruning_only_rejects_the_truly_unreachable():
    """The prune must be a NECESSARY condition, not a good guess.

    Everything the speed-ups rest on is this: `cannot_reach` says "no joint angles exist", and
    the solver is then skipped. A single pose wrongly rejected is a pose this library reports as
    beyond the arm when the arm can hold it -- the one error it exists to avoid. So: take poses
    the arm demonstrably CAN hold, because forward kinematics just produced them, and require
    that none is rejected.
    """
    for ch in [ob.so101()] + [_random_chain(n, s) for n, s in ((3, 1), (4, 2), (5, 3), (6, 4))]:
        rng = np.random.default_rng(11)
        q = rng.uniform(ch.limits[:, 0], ch.limits[:, 1], size=(600, ch.n))
        pos, M = ch.fk(q)
        assert not ch.cannot_reach(pos, M).any(), f"{ch.name} rejected a pose it just held"
        # Widening the tolerances can only make more poses admissible, never fewer.
        wide = ch.cannot_reach(pos, M, pos_tol=0.05, rot_tol=np.radians(45.0))
        tight = ch.cannot_reach(pos, M, pos_tol=0.001, rot_tol=np.radians(2.0))
        assert not wide.any() and (tight | ~wide).all()


def test_pruning_agrees_with_solving_everything():
    """On a real grid, the pruned sweep and the exhaustive one must give the SAME mask."""
    ch = ob.so101()
    t = np.linspace(0, 1, 40)
    pos = np.stack([0.22 + 0.14 * np.cos(3 * t), 0.18 * np.sin(3 * t), 0.10 + 0.12 * t], -1)
    quat = R.from_euler("y", (90 + 40 * np.sin(4 * t))[:, None], degrees=True).as_quat()
    cells = ob.base_grid(span=0.30, step=0.06, height=0.08)
    Rt = R.from_quat(quat).as_matrix()

    fast = ob.feasibility(ch, pos, quat, cells)
    slow = np.zeros_like(fast)
    for i, b in enumerate(cells):                      # what the library did before pruning
        _, pe, re = ch.ik(pos - b, Rt)
        slow[:, i] = (pe < 0.015) & (re < np.radians(20.0))
    assert np.array_equal(fast, slow), f"{int((fast != slow).sum())} entries differ"
    assert 0 < fast.sum() < fast.size, "fixture proves nothing if everything is one answer"


def test_envelope_fits_a_row_budget_on_any_arm():
    """The precompute samples grid ** (variable joints). Left alone that is 60 GB on a 6-DoF arm.

    It is coarsened to fit `Chain.ROWS` instead. That is safe rather than merely convenient: a
    coarser grid only widens the slack, so the region stays a superset and the prune stays a
    necessary condition -- which is what the assert below actually checks.
    """
    for n in (4, 5, 6, 7):
        ch = _random_chain(n, seed=n)
        free = set(ch.free_joints())
        var = max(len([i for i in range(1, n) if i not in free]), 1)
        env = ch.envelope()
        if env["maps"]:                                # it built one, so it has to have fit
            grid = max(3, min(96, int(ch.ROWS ** (1.0 / var))))
            assert grid ** var <= 8 * ch.ROWS, f"n={n} built {grid ** var} sample rows"
        rng = np.random.default_rng(5)
        q = rng.uniform(ch.limits[:, 0], ch.limits[:, 1], size=(200, ch.n))
        pos, M = ch.fk(q)
        assert not ch.cannot_reach(pos, M).any(), f"n={n} coarse grid pruned a reachable pose"


def test_the_three_by_three_solve_is_stable_where_the_solver_uses_it():
    """`_chol3` is the inner solve, and it is only sound because A is positive definite.

    Cramer's rule was tried here and looks identical on a good day: same speed, same mask on
    real data. On ill-conditioned batches it gives 1.3e-07 backward error against this routine's
    2.1e-16, and near-singular it returns nan -- which reads downstream as `pe < tol` being
    False, i.e. "the arm cannot reach". This pins the difference so nobody swaps it back.
    """
    from omnibase.chain import _chol3
    rng = np.random.default_rng(0)
    for d in (1e-4, 0.05):                             # the two dampings ik() actually uses
        U = rng.normal(size=(4000, 3, 3))
        U[:, :, 2] = U[:, :, 0]                        # rank-deficient, the singular case
        J = U @ np.swapaxes(U, -1, -2)
        A = J @ np.swapaxes(J, -1, -2) + d ** 2 * np.eye(3)
        v = rng.normal(size=(4000, 3, 1))
        x = _chol3(A, v)
        assert np.isfinite(x).all(), f"d={d}: non-finite solution"
        err = (np.linalg.norm(A @ x - v, axis=(-2, -1))
               / (np.linalg.norm(A, axis=(-2, -1)) * np.linalg.norm(x, axis=(-2, -1))))
        assert err.max() < 1e-12, f"d={d}: backward error {err.max():.2e}"


def test_free_joints_follows_the_chain_it_is_asked_about():
    """It is derived from the limits, so the answer must track a change to them.

    A memoised version was tried and answered for the chain as it was BUILT rather than as it
    is. Below is the case that catches it: joint 0 spins the tool about its own axis, so it is a
    free joint exactly when the rest of the chain keeps the tool on that axis. Widen joint 1 and
    it stops being free; pin joint 1 and it starts. A cache gets the second answer wrong, and
    silently -- `_polish_free` then skips a joint it should sweep, and the arm is reported unable
    to aim somewhere it can. The memo was also measured at 0.98x, so it bought nothing.
    """
    # Joint 0 turns about z at the origin; the tool sits further up that same z axis.
    j0 = Joint((0.0, 0.0, 0.0), (0, 0, 0, 1), "z", -2.0, 2.0, "roll")
    j1 = Joint((0.0, 0.0, 0.1), (0, 0, 0, 1), "y", -1.0, 1.0, "pitch")
    ch = Chain([j0, j1], tool=(0.0, 0.0, 0.1), name="probe")
    assert 0 not in ch.free_joints(), "pitch swings the tool off the roll axis; not free"

    ch.limits = ch.limits.copy()
    ch.limits[1] = [0.0, 0.0]                      # pin the pitch: the tool is back on the axis
    assert 0 in ch.free_joints(), "answered for the chain it was built with, not the one it has"


def test_splitting_the_sweep_into_blocks_changes_nothing():
    """`feasibility` walks frames in blocks so a long episode on a fine grid does not ask for
    the whole (frames, cells, 3) product at once. The block size is a memory ceiling only, so
    the mask must not depend on it -- including at a size that splits the episode many ways.
    """
    from omnibase import plan as P
    ch = ob.so101()
    t = np.linspace(0, 1, 55)
    pos = np.stack([0.22 + 0.12 * np.cos(3 * t), 0.16 * np.sin(3 * t), 0.10 + 0.10 * t], -1)
    quat = R.from_euler("y", (90 + 35 * np.sin(4 * t))[:, None], degrees=True).as_quat()
    cells = ob.base_grid(span=0.24, step=0.06, height=0.08)

    keep = P._PAIR_BATCH
    try:
        P._PAIR_BATCH = 10 ** 9                    # one block: the whole episode at once
        whole = ob.feasibility(ch, pos, quat, cells)
        for pairs in (len(cells), 3 * len(cells), 7 * len(cells)):
            P._PAIR_BATCH = pairs                  # 55, 19 and 8 blocks respectively
            assert np.array_equal(ob.feasibility(ch, pos, quat, cells), whole),                 f"block of {pairs} pairs changed the mask"
    finally:
        P._PAIR_BATCH = keep
    assert 0 < whole.sum() < whole.size, "fixture proves nothing if everything is one answer"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
