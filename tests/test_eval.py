"""The evaluation tools have exact reference answers; check them, not the toy policies.

On a reachable hand trajectory the replay reference must miss by exactly the base shift,
the oracle by nothing, joint deltas by a little, and the ambiguity of absolute joint targets
must be about the grid half-width. These are the calibration facts the tools rest on.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import omnibase as ob  # noqa: E402
from omnibase.eval import ambiguity, grasp_window, knn_policy, probe, resolve  # noqa: E402


def _chunks(n=40):
    """A hand path the arm can hold -- forward kinematics of a smooth joint sweep, so every pose
    is reachable by construction -- standing on a base 8 cm up, jaws closing halfway."""
    chain = ob.so101()
    base = np.array([0.0, 0.0, 0.08])
    q0, q1 = np.radians([[0, 20, -30, 40, 0]]), np.radians([[25, 35, -50, 55, 10]])
    p, M = chain.fk(q0 + (q1 - q0) * np.linspace(0, 1, n)[:, None])
    pos, quat = p + base, R.from_matrix(M).as_quat()
    grip = np.r_[np.ones(n // 2), np.zeros(n - n // 2)]
    return chain, [dict(pos=pos, quat=quat, grip=grip, base=base, episode=0, hand="right", start=0)]


def test_resolve_is_the_planner_solve():
    chain, chunks = _chunks()
    S, A, ok, where = resolve(chain, chunks, (0, 0, 0))
    assert S.shape == (39, 5) and A.shape == (39, 5) and ok.all()
    assert np.allclose(chain.fk(A)[0], chunks[0]["pos"][1:] - chunks[0]["base"], atol=0.015)
    assert where[0].tolist() == [0, 0] and where[-1].tolist() == [0, 38]


def test_grasp_window_brackets_the_closing_row():
    _, chunks = _chunks(40)
    w = grasp_window(chunks, rows=3)
    assert w.sum() == 6 and w[17:23].all() and not w[:17].any()


def test_probe_references_are_exact():
    chain, chunks = _chunks()
    res = probe(chain, chunks, offsets_cm=(2, 4), knn_grid_cm=None)
    for tag, b in (("x+2", 2.0), ("y-4", 4.0)):
        row = res["rows"][tag]
        assert abs(row["replay (unshifted rows)"]["miss_cm"] - b) < 0.05
        assert row["oracle (re-solved)"]["miss_cm"] < 1e-6
        assert row["joint deltas (unshifted)"]["miss_cm"] < 0.5
    assert res["rows"]["0"]["replay (unshifted rows)"]["miss_cm"] < 1e-9


def test_knn_at_one_base_memorises_and_a_grid_does_not():
    chain, chunks = _chunks()
    res = probe(chain, chunks, offsets_cm=(4,), knn_grid_cm=(-6, -3, 0, 3, 6))
    one = np.median([res["rows"][t]["knn (one base)"]["miss_cm"] for t in ("x+4", "x-4", "y+4", "y-4")])
    grid = np.median([res["rows"][t]["knn (base grid)"]["miss_cm"] for t in ("x+4", "x-4", "y+4", "y-4")])
    assert one > 3.0 and grid < 1.5


def test_ambiguity_costs_about_the_spread_for_absolute_joints():
    chain, chunks = _chunks()
    rows = ambiguity(chain, chunks, spans_cm=(2, 4))
    for r in rows:
        assert r["frames"] > 0
        assert 0.6 * r["span_cm"] < r["absolute_cm"] < 1.6 * r["span_cm"]
        assert r["delta_cm"] < 0.4 * r["absolute_cm"]


def test_user_policy_and_likelihood_are_reported():
    chain, chunks = _chunks()

    def hold(state, where=None):
        return state                                   # never moves: misses by the step length

    def nll(state, action, where=None):
        return np.full(len(state), 1.5)
    hold.nll = nll
    res = probe(chain, chunks, {"hold": hold}, offsets_cm=(2,), knn_grid_cm=None)
    row = res["rows"]["x+2"]["hold"]
    assert row["miss_cm"] > 0.2 and row["nll"] == 1.5 and row["nll_replay"] == 1.5


def test_holds_and_pick_hands():
    from omnibase.data import holds, pick_hands
    pos = np.array([[0, 0, 0], [0, 0, 0.0005], [0, 0, 0.05], [0, 0, 0.05]])
    assert holds(pos, mm=2).tolist() == [False, True, False, True]
    assert holds(pos, grip=np.array([1, 1, 0, 0]), mm=2).tolist() == [False, True, False, True]
    assert holds(pos, grip=np.array([1, 0, 0, 0]), mm=2).tolist() == [False, False, False, True]

    class H:                       # the two attributes pick_hands reads
        def __init__(self, name): self.name = name
    class E:
        hands = [H("right"), H("left")]
    assert [h.name for h in pick_hands(E(), "left,0")] == ["left", "right"]
    try:
        pick_hands(E(), "nope"); assert False
    except ValueError:
        pass


def test_service_builds_argv_and_serves_the_page():
    try:
        from fastapi.testclient import TestClient
        from omnibase import service
    except ImportError:
        return                                          # the extra is optional
    argv = service._argv("place", {"dataset": "d", "hands": "right", "home": "-0.1,0,0", "workers": 4}, Path("out.json"))
    assert argv[3:] == ["place", "d", "--hands", "right", "--home=-0.1,0,0", "--workers", "4", "--place-out", "out.json"]
    c = TestClient(service.app)
    assert "OmniBase" in c.get("/").text and c.get("/health").json()["ok"]
    assert c.post("/jobs", json={"tool": "nope", "args": {}}).status_code == 400
    assert c.post("/jobs", json={"tool": "report", "args": {}}).status_code == 400
