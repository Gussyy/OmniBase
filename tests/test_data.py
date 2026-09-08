"""Reading LeRobot datasets: both on-disk layouts, both kinds of action column.

These build their own tiny datasets in a temp directory rather than pointing at one, so they
run for anyone who clones this. Skipped entirely if pandas is not installed -- it is an optional
extra, and the rest of the library does not need it.

Run with ``python -m pytest`` or ``python tests/test_data.py``.
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

try:
    import pandas as pd
except ImportError:                                        # pragma: no cover
    pd = None


POSE_NAMES = [f"{h}_{f}" for h in ("right", "left")
              for f in ("x", "y", "z", "qx", "qy", "qz", "qw", "jaw")]
JOINT_NAMES = [f"{h}_{j}" for h in ("right", "left")
               for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                         "wrist_roll", "gripper")]


def _rows(n, names, seed=0):
    """Plausible action rows: real quaternions if it is a pose layout, angles if not."""
    rng = np.random.default_rng(seed)
    if "right_qw" in names:
        out = np.zeros((n, len(names)))
        for h, base in (("right", 0), ("left", 8)):
            out[:, base:base + 3] = rng.uniform(0.1, 0.3, (n, 3))
            out[:, base + 3:base + 7] = R.random(n, random_state=seed + base).as_quat()
            out[:, base + 7] = rng.choice([-1.0, 1.0], n)
        return out
    return rng.uniform(-0.6, 0.6, (n, len(names)))


def _write(root, names, version, episodes=(0, 1), n=25):
    """A dataset on disk in either layout, with just enough metadata to be read."""
    root = Path(root)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    path_tpl = ("data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet" if version == "v3.0"
                else "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": version, "fps": 30, "total_episodes": len(episodes),
        "data_path": path_tpl,
        "features": {"action": {"dtype": "float32", "shape": [len(names)], "names": names}},
    }), encoding="utf-8")

    frames = []
    for e in episodes:
        a = _rows(n, names, seed=e)
        frames.append(pd.DataFrame({"action": list(a),
                                    "episode_index": np.full(n, e, dtype=np.int64),
                                    "frame_index": np.arange(n)}))
    d = root / "data" / "chunk-000"
    d.mkdir(parents=True, exist_ok=True)
    if version == "v3.0":
        pd.concat(frames, ignore_index=True).to_parquet(d / "file-000.parquet")
    else:
        for e, f in zip(episodes, frames):
            f.to_parquet(d / f"episode_{e:06d}.parquet")
    return root


def _skip():
    if pd is None:
        print("  (skipped: pandas not installed)")
        return True
    return False


def test_reads_both_layouts_with_poses():
    """v2.1 keeps one episode per file, v3.0 concatenates them. Same answer either way."""
    if _skip():
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        got = {}
        for version in ("v2.1", "v3.0"):
            root = _write(tmp / version, POSE_NAMES, version)
            ep = ob.load(root, episode=1)
            assert ep.version == version and ep.frames == 25 and ep.fps == 30
            assert [h.name for h in ep.hands] == ["right", "left"]
            for h in ep.hands:
                assert h.source == "pose" and h.pos.shape == (25, 3) and h.quat.shape == (25, 4)
                assert h.grip is not None and h.grip.shape == (25,)
                assert np.allclose(np.linalg.norm(h.quat, axis=1), 1.0)
            got[version] = ep.hands[0].pos
        assert np.allclose(got["v2.1"], got["v3.0"]), "the two layouts disagree on episode 1"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_picks_the_right_episode():
    """Episode 0 and episode 1 hold different data; asking for one must not return the other."""
    if _skip():
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        for version in ("v2.1", "v3.0"):
            root = _write(tmp / version, POSE_NAMES, version)
            a = ob.load(root, episode=0).hands[0].pos
            b = ob.load(root, episode=1).hands[0].pos
            assert not np.allclose(a, b), f"{version}: both episodes came back identical"
            try:
                ob.load(root, episode=99)
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError("a missing episode must raise, not return something")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_joint_columns_become_poses_through_fk():
    """Joint angles are usable too -- run them through the chain rather than refusing."""
    if _skip():
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        chain = ob.so101()
        root = _write(tmp / "joints", JOINT_NAMES, "v3.0")
        try:
            ob.load(root, episode=0)
        except ValueError as exc:
            assert "joint" in str(exc).lower(), "the error should say what to do about it"
        else:
            raise AssertionError("joint columns without a chain must not silently succeed")

        ep = ob.load(root, episode=0, chain=chain)
        assert sorted(h.name for h in ep.hands) == ["left", "right"]
        for h in ep.hands:
            assert h.source == "fk" and h.joints is not None
            # The poses must be this chain's own forward kinematics of those angles.
            pos, M = chain.fk(h.joints)
            assert np.allclose(h.pos, pos, atol=1e-12)
            assert np.allclose(np.abs((R.from_quat(h.quat).as_matrix() * M).sum((1, 2))), 3.0,
                               atol=1e-6)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_unusable_column_says_why():
    """A column that is neither poses nor joints must name its columns in the error."""
    if _skip():
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        root = _write(tmp / "odd", ["a", "b", "c"], "v3.0")
        try:
            ob.load(root, episode=0, chain=ob.so101())
        except ValueError as exc:
            assert "'a'" in str(exc) or "a" in str(exc)
            assert "_qx" in str(exc), "the error should say what naming it wants"
        else:
            raise AssertionError("unreadable columns must raise")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_hand_prefix_is_not_confused_by_joint_names():
    """`left_wrist_roll` is a joint of hand `left`, not a hand called `left_wrist`."""
    groups = ob.data._split(["left_wrist_roll", "left_x", "left_y", "left_z",
                             "left_qx", "left_qy", "left_qz", "left_qw"])
    assert "left" in groups and "left_wrist" not in groups
    assert set(("x", "y", "z", "qx", "qy", "qz", "qw")) <= set(groups["left"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall data checks passed")
